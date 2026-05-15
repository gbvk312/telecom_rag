"""Bedrock RAG pipeline — retrieval + generation using Knowledge Base."""
import boto3
import json
from config import AWS_REGION, LLM_MODEL_ID

bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

SYSTEM_PROMPT = """You are TelecomRAG, an expert assistant specializing in 3GPP telecommunications standards and industry whitepapers. You have deep knowledge of 3GPP Releases 15 through 18, O-RAN architecture, IMS subsystems, 5G NR, LTE, ECN, and related telecom protocols.

When answering:
1. Always cite the exact 3GPP Technical Specification number (e.g., 3GPP TS 23.501 Section 4.2.3) when referencing a standard.
2. Identify and explicitly list any dependent specifications or concepts related to the answer (these will be used to update the dependency graph).
3. If the question cannot be answered from the provided context, say: 'This query is outside the current knowledge base. Please upload the relevant 3GPP specification.'
4. Format responses with: Summary → Technical Detail → References → Related Nodes (for graph update).
5. Always respond in structured markdown."""


def get_kb_id():
    """Load Knowledge Base ID from config file."""
    import os
    config_path = os.path.join(os.path.dirname(__file__), ".kb_config")
    try:
        with open(config_path) as f:
            return json.load(f)["kb_id"]
    except FileNotFoundError:
        return None


def retrieve_from_kb(query, kb_id, top_k=4):
    """Retrieve relevant chunks from Bedrock Knowledge Base."""
    try:
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": top_k}
            },
        )
        results = []
        for item in response.get("retrievalResults", []):
            results.append({
                "text": item["content"]["text"],
                "score": item.get("score", 0),
                "source": item.get("location", {}).get("s3Location", {}).get("uri", "Unknown"),
            })
        return results
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []


def generate_response(query, context_chunks, chat_history=None):
    """Generate response using Bedrock Claude with retrieved context."""
    # Build context from retrieved chunks
    context = "\n\n---\n\n".join(
        [f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks]
    )

    # Build conversation history
    history_text = ""
    if chat_history:
        for role, msg in chat_history[-5:]:
            history_text += f"{role}: {msg}\n"

    user_message = f"""Based on the following context from 3GPP specifications and telecom whitepapers, answer the user's question.

## Retrieved Context:
{context}

## Conversation History:
{history_text}

## User Question:
{query}

Provide a comprehensive answer with specific 3GPP specification references. At the end, list "Related Nodes:" with key entities mentioned (specs, concepts, technologies) for dependency graph update."""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message}],
        "temperature": 0.3,
    })

    response = bedrock_runtime.invoke_model(
        modelId=LLM_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def query_rag(query, chat_history=None):
    """Full RAG pipeline: retrieve + generate."""
    kb_id = get_kb_id()

    if not kb_id:
        # Fallback: direct LLM without KB
        return generate_response(query, [], chat_history), []

    # Retrieve relevant chunks
    chunks = retrieve_from_kb(query, kb_id)

    # Generate response
    response = generate_response(query, chunks, chat_history)

    # Format citations
    citations = []
    for c in chunks:
        source = c["source"].split("/")[-1] if "/" in c["source"] else c["source"]
        citations.append(f"📄 {source} (score: {c['score']:.3f})")

    return response, citations
