"""Telecom RAG Assistant — Main Gradio Application."""
import os
import json
import base64
import gradio as gr
from graph_builder import (
    build_seed_graph, update_graph, render_graph,
    save_graph_json, load_graph_json, get_graph_stats,
)
from entity_extractor import extract_from_response
from bedrock_rag import query_rag, get_kb_id
from config import GRAPH_OUTPUT_PATH

# Initialize graph
G = build_seed_graph()
render_graph(G)
save_graph_json(G)

# Chat history storage
chat_history = []


def get_graph_iframe():
    """Read graph HTML and return as iframe."""
    try:
        with open(GRAPH_OUTPUT_PATH, "r") as f:
            content = f.read()
        encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        return f'<iframe src="data:text/html;base64,{encoded}" width="100%" height="600" frameborder="0" style="border:1px solid #444; border-radius:8px; background:#1a1a2e;"></iframe>'
    except FileNotFoundError:
        return "<p style='text-align:center; color:#888; padding:40px;'>Graph not yet generated. Ask a question to start!</p>"


def get_stats_text():
    """Get formatted graph statistics."""
    stats = get_graph_stats(G)
    return (
        f"**📊 Graph Stats** — "
        f"Nodes: {stats['total_nodes']} | Edges: {stats['total_edges']} | "
        f"🔵 Specs: {stats['specs']} | 🟢 Releases: {stats['releases']} | "
        f"🟠 Concepts: {stats['concepts']} | 🟣 Papers: {stats['papers']}"
    )


def respond(message, history):
    """Handle chat message — RAG query + graph update."""
    global G, chat_history

    if not message.strip():
        return ""

    # Query RAG
    response, citations = query_rag(message, chat_history)

    # Extract entities from response and update graph
    entities, relations = extract_from_response(response)
    query_entities, _ = extract_from_response(message)
    all_entities = entities + query_entities

    added_nodes, added_edges = update_graph(G, all_entities, relations)

    # Highlight query-relevant nodes
    highlight = set(e["name"] for e in all_entities)
    render_graph(G, highlight_nodes=highlight)
    save_graph_json(G)

    # Format response with citations
    full_response = response
    if citations:
        full_response += "\n\n---\n**📚 Sources:**\n" + "\n".join(citations)
    if added_nodes > 0 or added_edges > 0:
        full_response += f"\n\n*🕸️ Graph updated: +{added_nodes} nodes, +{added_edges} edges*"

    # Update internal history
    chat_history.append(("User", message))
    chat_history.append(("Assistant", response))

    return full_response


def filter_graph(filter_type):
    """Filter graph by node type."""
    global G
    type_map = {
        "All": "all",
        "3GPP Specs Only": "spec",
        "Releases Only": "release",
        "Concepts Only": "concept",
        "Whitepapers Only": "paper",
    }
    render_graph(G, filter_type=type_map.get(filter_type, "all"))
    return get_graph_iframe(), get_stats_text()


def search_node(search_term):
    """Search and highlight specific nodes."""
    global G
    if not search_term.strip():
        render_graph(G)
        return get_graph_iframe()

    matching = set()
    for node in G.nodes():
        if search_term.lower() in node.lower():
            matching.add(node)
            matching.update(G.predecessors(node))
            matching.update(G.successors(node))

    render_graph(G, highlight_nodes=matching)
    return get_graph_iframe()


def refresh_graph():
    """Refresh graph display after chat interaction."""
    return get_graph_iframe(), get_stats_text()


def upload_files(files):
    """Handle file upload — trigger re-indexing."""
    if not files:
        return "No files uploaded."

    import boto3
    from config import AWS_REGION, S3_BUCKET, S3_PREFIX

    s3 = boto3.client("s3", region_name=AWS_REGION)
    uploaded = []

    if isinstance(files, str):
        files = [files]

    for file_path in files:
        fname = os.path.basename(file_path)
        s3_key = S3_PREFIX + "uploads/" + fname
        s3.upload_file(file_path, S3_BUCKET, s3_key)
        local_path = os.path.join(os.path.dirname(__file__), "data", "uploads", fname)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(file_path, "r", errors="ignore") as src:
            content = src.read()
        with open(local_path, "w") as dst:
            dst.write(content)
        uploaded.append(fname)

    # Trigger re-ingestion
    kb_id = get_kb_id()
    if kb_id:
        try:
            config_path = os.path.join(os.path.dirname(__file__), ".kb_config")
            with open(config_path) as f:
                config = json.load(f)
            bedrock_agent = boto3.client("bedrock-agent", region_name=AWS_REGION)
            bedrock_agent.start_ingestion_job(
                knowledgeBaseId=config["kb_id"],
                dataSourceId=config["ds_id"],
            )
            return f"✅ Uploaded {len(uploaded)} files: {', '.join(uploaded)}\n🔄 Re-indexing started..."
        except Exception as e:
            return f"✅ Uploaded {len(uploaded)} files. ⚠️ Re-indexing note: {e}"

    return f"✅ Uploaded {len(uploaded)} files: {', '.join(uploaded)}"


def download_graph():
    """Download graph JSON."""
    from config import GRAPH_JSON_PATH
    save_graph_json(G)
    return GRAPH_JSON_PATH


# Build Gradio UI
with gr.Blocks(title="Telecom RAG Assistant") as app:

    gr.Markdown("# 🏢 TELECOM RAG ASSISTANT — 3GPP Knowledge + Dependency Graph\n*Powered by Amazon Bedrock Knowledge Bases | Claude 3 Sonnet | Titan Embeddings*")

    with gr.Row(equal_height=False):
        # ===== LEFT PANEL: Chat =====
        with gr.Column(scale=5):
            gr.Markdown("### 💬 Ask a Question")

            chatbot = gr.Chatbot(height=400, label="Conversation")

            with gr.Group():
                msg_input = gr.Textbox(
                    placeholder="e.g., What is network slicing in 5G? How does O-RAN relate to 3GPP?",
                    label="Your Question",
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("🚀 Send", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

            with gr.Accordion("📁 Upload Documents", open=False):
                upload_btn = gr.UploadButton(
                    "Upload 3GPP Docs / Whitepapers",
                    file_types=[".pdf", ".md", ".txt", ".docx"],
                    file_count="multiple",
                )
                upload_status = gr.Textbox(label="Status", interactive=False)

        # ===== RIGHT PANEL: Graph =====
        with gr.Column(scale=6):
            gr.Markdown("### 🕸️ Dependency Graph")
            stats_display = gr.Markdown(value=get_stats_text())
            graph_display = gr.HTML(value=get_graph_iframe())

            with gr.Row():
                search_input = gr.Textbox(
                    placeholder="Search: TS 23.501, 5G Core, NWDAF...",
                    label="🔍 Search Node",
                    scale=3,
                )
                search_btn = gr.Button("Search", scale=1)

            with gr.Row():
                filter_dropdown = gr.Dropdown(
                    choices=["All", "3GPP Specs Only", "Releases Only", "Concepts Only", "Whitepapers Only"],
                    value="All",
                    label="Filter",
                    scale=2,
                )
                download_btn = gr.DownloadButton("📥 Graph JSON", scale=1)

            gr.Markdown("**Legend:** 🔵 3GPP Spec | 🟢 Release | 🟠 Concept | 🟣 Whitepaper | 🔴 Active Query")

    # ===== Event Handlers =====

    def chat_and_update(message, history):
        """Send message, get response, update graph display."""
        if not message.strip():
            return history, "", get_graph_iframe(), get_stats_text()

        response = respond(message, history)
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, "", get_graph_iframe(), get_stats_text()

    def clear_all():
        """Clear chat and reset graph."""
        global chat_history
        chat_history = []
        render_graph(G)
        return None, "", get_graph_iframe(), get_stats_text()

    send_btn.click(
        chat_and_update,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input, graph_display, stats_display],
    )
    msg_input.submit(
        chat_and_update,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input, graph_display, stats_display],
    )
    clear_btn.click(
        clear_all,
        outputs=[chatbot, msg_input, graph_display, stats_display],
    )
    search_btn.click(search_node, inputs=[search_input], outputs=[graph_display])
    search_input.submit(search_node, inputs=[search_input], outputs=[graph_display])
    filter_dropdown.change(filter_graph, inputs=[filter_dropdown], outputs=[graph_display, stats_display])
    upload_btn.upload(upload_files, inputs=[upload_btn], outputs=[upload_status])
    download_btn.click(download_graph, outputs=[download_btn])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
