"""Configuration for Telecom RAG application."""
import os

AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
ACCOUNT_ID = "715001841576"

# S3
S3_BUCKET = f"telecom-rag-kb-{ACCOUNT_ID}"
S3_PREFIX = "knowledge-base-docs/"

# Bedrock
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
KB_NAME = "telecom-rag-knowledge-base"

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GRAPH_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "graph_output.html")
GRAPH_JSON_PATH = os.path.join(os.path.dirname(__file__), "graph.json")

# RAG
TOP_K = 4
