# Telecom RAG Assistant — 3GPP Knowledge Base + Dependency Graph

## Architecture
```
S3 (3GPP docs + whitepapers)
    → Bedrock Knowledge Base (auto-chunking + Titan Embeddings)
        → Bedrock Claude 3 Sonnet (generation)
            → Gradio UI (chat + interactive graph)
```

## Prerequisites
- AWS CLI configured with access to Bedrock, S3, OpenSearch Serverless
- Python 3.11+
- Bedrock model access enabled for:
  - `amazon.titan-embed-text-v2:0`
  - `anthropic.claude-3-sonnet-20240229-v1:0`

## Quick Start

### 1. Install dependencies
```bash
pip install gradio networkx pyvis spacy boto3
python -m spacy download en_core_web_sm
```

### 2. Set up AWS infrastructure (one-time)
```bash
cd /workshop/project/telecom_rag
python3.11 aws_setup.py
```
This will:
- Create S3 bucket and upload documents
- Create IAM role for Bedrock KB
- Create OpenSearch Serverless collection
- Create Bedrock Knowledge Base
- Start document ingestion (~5-10 min for 85MB)

### 3. Run the application
```bash
python3.11 main.py
```
Access at: http://localhost:7860

## Project Structure
```
telecom_rag/
├── main.py              ← Gradio UI + orchestration
├── app.py               ← Application entry point
├── aws_setup.py         ← AWS infrastructure provisioning
├── bedrock_rag.py       ← Bedrock KB retrieval + Claude generation
├── graph_builder.py     ← NetworkX graph + Pyvis rendering
├── entity_extractor.py  ← Telecom NER + relation extraction
├── config.py            ← Configuration
├── lib/                 ← Frontend libraries (vis.js, tom-select)
├── data/
│   ├── 3gpp/           ← 58 3GPP spec files (~85MB)
│   └── whitepapers/    ← 12 telecom whitepapers
├── graph_output.html    ← Generated interactive graph
├── graph.json           ← Graph data (node-link format)
└── .kb_config           ← Auto-generated KB IDs
```

## Features
- ✅ RAG with 3GPP specs via Bedrock Knowledge Bases
- ✅ Interactive dependency graph (Pyvis)
- ✅ Color-coded nodes (spec/release/concept/paper)
- ✅ Query-triggered node highlighting
- ✅ Graph search and filtering
- ✅ File upload with auto re-indexing
- ✅ Source citations
- ✅ Chat history with context

## Data
- **3GPP Specs**: 58 documents from Release 16 and Release 18
  - TS 23.501, TS 38.300, TS 23.334, TS 36.300, TS 26.114, etc.
- **Whitepapers**: 12 synthetic papers covering:
  - 5G NR Architecture, AI/ML in RAN, Network Slicing
  - O-RAN, QoS, IMS, NTN, Private 5G, Edge Computing
  - Security, V2X, 5GC SBA, NR PHY, Migration, Mission Critical

## Cost Estimate (AWS)
- Bedrock KB: ~$0.01/query (retrieval) + ~$0.003/1K tokens (Claude Sonnet)
- OpenSearch Serverless: ~$0.24/hr (OCU minimum)
- S3: Negligible for 85MB
- **Total for demo**: ~$5-10/day with moderate usage
