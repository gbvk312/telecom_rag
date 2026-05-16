"""Telecom RAG Assistant - Main Gradio Application."""
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


def create_app():
    """Factory function that creates and returns the Gradio app.

    All mutable state is encapsulated inside this function to avoid
    global variables and the race conditions they cause when Gradio
    serves multiple concurrent users.
    """

    # Initialize graph (instance-local, not global)
    G = build_seed_graph()
    render_graph(G)
    save_graph_json(G)

    def get_graph_iframe():
        """Read graph HTML and return as iframe."""
        try:
            with open(GRAPH_OUTPUT_PATH, "r") as f:
                content = f.read()
            encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
            return (
                f'<iframe src="data:text/html;base64,{encoded}" '
                f'width="100%" height="600" frameborder="0" '
                f'style="border:1px solid #444; border-radius:8px; background:#1a1a2e;">'
                f'</iframe>'
            )
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

    def respond(message, chat_history_state):
        """Handle chat message — RAG query + graph update."""
        if not message.strip():
            return ""

        response, citations = query_rag(message, chat_history_state)

        entities, relations = extract_from_response(response)
        query_entities, _ = extract_from_response(message)
        all_entities = entities + query_entities

        added_nodes, added_edges = update_graph(G, all_entities, relations)

        highlight = set(e["name"] for e in all_entities)
        render_graph(G, highlight_nodes=highlight)
        save_graph_json(G)

        full_response = response
        if citations:
            full_response += "\n\n---\n**📚 Sources:**\n" + "\n".join(citations)
        if added_nodes > 0 or added_edges > 0:
            full_response += f"\n\n*🕸️ Graph updated: +{added_nodes} nodes, +{added_edges} edges*"

        chat_history_state.append(("User", message))
        chat_history_state.append(("Assistant", response))

        return full_response

    def filter_graph(filter_type):
        """Filter graph by node type."""
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
            # Use binary mode for all files (fixes PDF corruption)
            with open(file_path, "rb") as src:
                content = src.read()
            with open(local_path, "wb") as dst:
                dst.write(content)
            uploaded.append(fname)

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
        # Gradio State for per-session chat history (replaces global mutable state)
        chat_history_state = gr.State([])

        gr.Markdown("# 🏢 TELECOM RAG ASSISTANT — 3GPP Knowledge + Dependency Graph\n*Powered by Amazon Bedrock Knowledge Bases | Claude 3 Sonnet | Titan Embeddings*")

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                gr.Markdown("### 💬 Ask a Question")
                chatbot = gr.Chatbot(height=400, label="Conversation")
                with gr.Group():
                    msg_input = gr.Textbox(
                        placeholder="e.g., What is network slicing in 5G?",
                        label="Your Question", lines=2,
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

            with gr.Column(scale=6):
                gr.Markdown("### 🕸️ Dependency Graph")
                stats_display = gr.Markdown(value=get_stats_text())
                graph_display = gr.HTML(value=get_graph_iframe())
                with gr.Row():
                    search_input = gr.Textbox(placeholder="Search: TS 23.501, 5G Core...", label="🔍 Search Node", scale=3)
                    search_btn = gr.Button("Search", scale=1)
                with gr.Row():
                    filter_dropdown = gr.Dropdown(
                        choices=["All", "3GPP Specs Only", "Releases Only", "Concepts Only", "Whitepapers Only"],
                        value="All", label="Filter", scale=2,
                    )
                    download_btn = gr.DownloadButton("📥 Graph JSON", scale=1)
                gr.Markdown("**Legend:** 🔵 3GPP Spec | 🟢 Release | 🟠 Concept | 🟣 Whitepaper | 🔴 Active Query")

        def chat_and_update(message, history, hist_state):
            if not message.strip():
                return history, "", get_graph_iframe(), get_stats_text(), hist_state
            response = respond(message, hist_state)
            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history, "", get_graph_iframe(), get_stats_text(), hist_state

        def clear_all():
            render_graph(G)
            return None, "", get_graph_iframe(), get_stats_text(), []

        send_btn.click(chat_and_update, inputs=[msg_input, chatbot, chat_history_state], outputs=[chatbot, msg_input, graph_display, stats_display, chat_history_state])
        msg_input.submit(chat_and_update, inputs=[msg_input, chatbot, chat_history_state], outputs=[chatbot, msg_input, graph_display, stats_display, chat_history_state])
        clear_btn.click(clear_all, outputs=[chatbot, msg_input, graph_display, stats_display, chat_history_state])
        search_btn.click(search_node, inputs=[search_input], outputs=[graph_display])
        search_input.submit(search_node, inputs=[search_input], outputs=[graph_display])
        filter_dropdown.change(filter_graph, inputs=[filter_dropdown], outputs=[graph_display, stats_display])
        upload_btn.upload(upload_files, inputs=[upload_btn], outputs=[upload_status])
        download_btn.click(download_graph, outputs=[download_btn])

    return app


if __name__ == "__main__":
    server_host = os.environ.get("SERVER_HOST", "127.0.0.1")
    server_port = int(os.environ.get("SERVER_PORT", "7860"))
    app = create_app()
    app.launch(server_name=server_host, server_port=server_port, share=False)
