"""Telecom RAG Assistant — Streamlit Frontend."""
import os
import sys
import json
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_builder import (
    build_seed_graph, update_graph, render_graph,
    save_graph_json, get_graph_stats,
)
from entity_extractor import extract_from_response
from bedrock_rag import query_rag, get_kb_id
from config import GRAPH_OUTPUT_PATH, GRAPH_JSON_PATH, S3_BUCKET, S3_PREFIX, AWS_REGION

# Page config — must be first Streamlit command
st.set_page_config(
    page_title="Telecom RAG Assistant",
    page_icon="🏢",
    layout="wide",
)

# ===== Session State Initialization =====
if "graph" not in st.session_state:
    st.session_state.graph = build_seed_graph()
    render_graph(st.session_state.graph)
    save_graph_json(st.session_state.graph)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def handle_query(user_input):
    """Process user query through RAG pipeline."""
    G = st.session_state.graph

    response, citations = query_rag(user_input, st.session_state.chat_history)

    entities, relations = extract_from_response(response)
    query_entities, _ = extract_from_response(user_input)
    all_entities = entities + query_entities

    added_nodes, added_edges = update_graph(G, all_entities, relations)

    highlight = set(e["name"] for e in all_entities)
    render_graph(G, highlight_nodes=highlight)
    save_graph_json(G)

    st.session_state.chat_history.append(("User", user_input))
    st.session_state.chat_history.append(("Assistant", response))

    return response, citations, added_nodes, added_edges


# ===== HEADER =====
st.markdown("# 🏢 Telecom RAG Assistant")
st.markdown("*3GPP Knowledge Base + Dependency Graph — Powered by Amazon Bedrock*")
st.divider()

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("## 📊 Graph Stats")
    stats = get_graph_stats(st.session_state.graph)
    c1, c2 = st.columns(2)
    c1.metric("Nodes", stats["total_nodes"])
    c2.metric("Edges", stats["total_edges"])
    c3, c4 = st.columns(2)
    c3.metric("🔵 Specs", stats["specs"])
    c4.metric("🟢 Releases", stats["releases"])
    c5, c6 = st.columns(2)
    c5.metric("🟠 Concepts", stats["concepts"])
    c6.metric("🟣 Papers", stats["papers"])

    st.divider()
    st.markdown("## 🎛️ Graph Controls")

    search_term = st.text_input("🔍 Search Node", placeholder="TS 23.501, 5G Core...")
    if search_term:
        G = st.session_state.graph
        matching = set()
        for node in G.nodes():
            if search_term.lower() in node.lower():
                matching.add(node)
                matching.update(G.predecessors(node))
                matching.update(G.successors(node))
        if matching:
            render_graph(G, highlight_nodes=matching)
            st.success(f"Found {len(matching)} related nodes")
        else:
            st.warning("No matching nodes found")

    filter_type = st.selectbox(
        "Filter by Type",
        ["All", "3GPP Specs Only", "Releases Only", "Concepts Only", "Whitepapers Only"],
    )
    type_map = {"All": "all", "3GPP Specs Only": "spec", "Releases Only": "release",
                "Concepts Only": "concept", "Whitepapers Only": "paper"}
    if filter_type != "All" and not search_term:
        render_graph(st.session_state.graph, filter_type=type_map[filter_type])

    st.divider()
    st.markdown("## 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Add 3GPP specs or whitepapers",
        type=["pdf", "md", "txt", "docx"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        import boto3
        s3 = boto3.client("s3", region_name=AWS_REGION)
        for uf in uploaded_files:
            s3_key = S3_PREFIX + "uploads/" + uf.name
            s3.upload_fileobj(uf, S3_BUCKET, s3_key)
        st.success(f"✅ Uploaded {len(uploaded_files)} files!")
        kb_id = get_kb_id()
        if kb_id:
            try:
                config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".kb_config")
                with open(config_path) as f:
                    config = json.load(f)
                bedrock_agent = boto3.client("bedrock-agent", region_name=AWS_REGION)
                bedrock_agent.start_ingestion_job(
                    knowledgeBaseId=config["kb_id"], dataSourceId=config["ds_id"])
                st.info("🔄 Re-indexing started...")
            except Exception as e:
                st.warning(f"⚠️ {e}")

    st.divider()
    if os.path.exists(GRAPH_JSON_PATH):
        with open(GRAPH_JSON_PATH, "r") as f:
            graph_json = f.read()
        st.download_button("📥 Download Graph JSON", data=graph_json,
                           file_name="telecom_rag_graph.json", mime="application/json")

    st.markdown("---\n**Legend:**\n- 🔵 3GPP Spec\n- 🟢 Release\n- 🟠 Concept\n- 🟣 Whitepaper\n- 🔴 Active Query")


# ===== MAIN LAYOUT: Chat + Graph side by side =====
chat_col, graph_col = st.columns([4, 6])

# ===== CHAT COLUMN =====
with chat_col:
    st.markdown("### 💬 Chat")

    # Chat messages container
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    with st.expander("📚 Sources", expanded=False):
                        for cite in msg["citations"]:
                            st.caption(cite)

    # Chat input at bottom
    if user_input := st.chat_input("Ask about 3GPP specs, 5G, O-RAN, IMS..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate response
        with st.spinner("🔍 Searching knowledge base..."):
            response, citations, added_nodes, added_edges = handle_query(user_input)

        # Build assistant message
        assistant_content = response
        if added_nodes > 0 or added_edges > 0:
            assistant_content += f"\n\n---\n*🕸️ Graph updated: +{added_nodes} nodes, +{added_edges} edges*"

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_content,
            "citations": citations,
        })
        st.rerun()

# ===== GRAPH COLUMN =====
with graph_col:
    st.markdown("### 🕸️ Dependency Graph")
    if os.path.exists(GRAPH_OUTPUT_PATH):
        st.iframe(GRAPH_OUTPUT_PATH, height=560)
    else:
        st.info("Graph will appear after initialization or your first query.")
