"""AWS infrastructure setup for Telecom RAG — S3 + Bedrock Knowledge Base."""
import boto3
import json
import time
import os
import sys

from config import AWS_REGION, S3_BUCKET, S3_PREFIX, ACCOUNT_ID, KB_NAME, DATA_DIR, EMBEDDING_MODEL_ID

s3 = boto3.client("s3", region_name=AWS_REGION)
bedrock_agent = boto3.client("bedrock-agent", region_name=AWS_REGION)
iam = boto3.client("iam", region_name=AWS_REGION)
oss = boto3.client("opensearchserverless", region_name=AWS_REGION)

ROLE_NAME = "TelecomRAG-BedrockKB-Role"
COLLECTION_NAME = "telecom-rag-vectors"
INDEX_NAME = "telecom-rag-index"


def create_s3_bucket():
    """Create S3 bucket for knowledge base documents."""
    try:
        if AWS_REGION == "us-east-1":
            s3.create_bucket(Bucket=S3_BUCKET)
        else:
            s3.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
            )
        print(f"✅ Created S3 bucket: {S3_BUCKET}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"✅ S3 bucket already exists: {S3_BUCKET}")
    except Exception as e:
        if "BucketAlreadyOwnedByYou" in str(e) or "BucketAlreadyExists" in str(e):
            print(f"✅ S3 bucket already exists: {S3_BUCKET}")
        else:
            raise


def upload_documents():
    """Upload all documents from data/ to S3."""
    count = 0
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if not fname.endswith((".md", ".txt", ".pdf")):
                continue
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, DATA_DIR)
            s3_key = S3_PREFIX + rel_path
            s3.upload_file(local_path, S3_BUCKET, s3_key)
            count += 1
    print(f"✅ Uploaded {count} documents to s3://{S3_BUCKET}/{S3_PREFIX}")


def create_kb_role():
    """Create IAM role for Bedrock Knowledge Base."""
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": ACCOUNT_ID},
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:bedrock:{AWS_REGION}:{ACCOUNT_ID}:knowledge-base/*"
                    },
                },
            }
        ],
    }

    try:
        resp = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role for Telecom RAG Bedrock Knowledge Base",
        )
        role_arn = resp["Role"]["Arn"]
        print(f"✅ Created IAM role: {ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}"
        # Update trust policy
        iam.update_assume_role_policy(
            RoleName=ROLE_NAME,
            PolicyDocument=json.dumps(trust_policy),
        )
        print(f"✅ IAM role already exists: {ROLE_NAME}")

    # Attach inline policy for S3 + Bedrock + OpenSearch
    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{S3_BUCKET}",
                    f"arn:aws:s3:::{S3_BUCKET}/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel"],
                "Resource": f"arn:aws:bedrock:{AWS_REGION}::foundation-model/{EMBEDDING_MODEL_ID}",
            },
            {
                "Effect": "Allow",
                "Action": ["aoss:APIAccessAll"],
                "Resource": f"arn:aws:aoss:{AWS_REGION}:{ACCOUNT_ID}:collection/*",
            },
        ],
    }
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="TelecomRAG-KB-Policy",
        PolicyDocument=json.dumps(policy_doc),
    )
    time.sleep(10)  # Wait for IAM propagation
    return role_arn


def create_opensearch_collection():
    """Create OpenSearch Serverless collection for vector search."""
    # Create encryption policy
    enc_policy = json.dumps(
        {
            "Rules": [{"ResourceType": "collection", "Resource": [f"collection/{COLLECTION_NAME}"]}],
            "AWSOwnedKey": True,
        }
    )
    try:
        oss.create_security_policy(
            name=f"{COLLECTION_NAME}-enc",
            type="encryption",
            policy=enc_policy,
        )
    except Exception as e:
        if "ConflictException" in str(type(e).__name__) or "already exists" in str(e).lower():
            pass
        else:
            raise

    # Create network policy
    net_policy = json.dumps(
        [
            {
                "Rules": [
                    {"ResourceType": "collection", "Resource": [f"collection/{COLLECTION_NAME}"]},
                    {"ResourceType": "dashboard", "Resource": [f"collection/{COLLECTION_NAME}"]},
                ],
                "AllowFromPublic": True,
            }
        ]
    )
    try:
        oss.create_security_policy(
            name=f"{COLLECTION_NAME}-net",
            type="network",
            policy=net_policy,
        )
    except Exception as e:
        if "ConflictException" in str(type(e).__name__) or "already exists" in str(e).lower():
            pass
        else:
            raise

    # Create data access policy
    data_policy = json.dumps(
        [
            {
                "Rules": [
                    {
                        "ResourceType": "index",
                        "Resource": [f"index/{COLLECTION_NAME}/*"],
                        "Permission": [
                            "aoss:CreateIndex",
                            "aoss:UpdateIndex",
                            "aoss:DescribeIndex",
                            "aoss:ReadDocument",
                            "aoss:WriteDocument",
                        ],
                    },
                    {
                        "ResourceType": "collection",
                        "Resource": [f"collection/{COLLECTION_NAME}"],
                        "Permission": [
                            "aoss:CreateCollectionItems",
                            "aoss:DescribeCollectionItems",
                            "aoss:UpdateCollectionItems",
                        ],
                    },
                ],
                "Principal": [
                    f"arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}",
                    f"arn:aws:sts::{ACCOUNT_ID}:assumed-role/vscode-server-CodeEditorInstanceBootstrapRole-81AXesWau8rB/*",
                ],
            }
        ]
    )
    try:
        oss.create_access_policy(
            name=f"{COLLECTION_NAME}-access",
            type="data",
            policy=data_policy,
        )
    except Exception as e:
        if "ConflictException" in str(type(e).__name__) or "already exists" in str(e).lower():
            # Update existing policy
            oss.update_access_policy(
                name=f"{COLLECTION_NAME}-access",
                type="data",
                policyVersion="MTY...",  # Will fail gracefully
                policy=data_policy,
            )
        else:
            raise

    # Create collection
    try:
        resp = oss.create_collection(
            name=COLLECTION_NAME,
            type="VECTORSEARCH",
        )
        collection_id = resp["createCollectionDetail"]["id"]
        print(f"✅ Created OpenSearch collection: {COLLECTION_NAME} (id: {collection_id})")
    except Exception as e:
        if "ConflictException" in str(type(e).__name__) or "already exists" in str(e).lower():
            # Get existing collection
            resp = oss.list_collections(
                collectionFilters={"name": COLLECTION_NAME}
            )
            collection_id = resp["collectionSummaries"][0]["id"]
            print(f"✅ OpenSearch collection already exists: {COLLECTION_NAME} (id: {collection_id})")
        else:
            raise

    # Wait for collection to be active
    print("⏳ Waiting for OpenSearch collection to become active...")
    for _ in range(60):
        resp = oss.batch_get_collection(ids=[collection_id])
        status = resp["collectionDetails"][0]["status"]
        if status == "ACTIVE":
            break
        time.sleep(10)

    endpoint = resp["collectionDetails"][0]["collectionEndpoint"]
    collection_arn = resp["collectionDetails"][0]["arn"]
    print(f"✅ Collection active. Endpoint: {endpoint}")
    return collection_arn, collection_id


def create_opensearch_index(collection_id):
    """Create vector index in OpenSearch Serverless."""
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth

    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        AWS_REGION,
        "aoss",
        session_token=credentials.token,
    )

    resp = oss.batch_get_collection(ids=[collection_id])
    endpoint = resp["collectionDetails"][0]["collectionEndpoint"]
    host = endpoint.replace("https://", "")

    client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300,
    )

    index_body = {
        "settings": {
            "index": {"knn": True, "knn.algo_param.ef_search": 512}
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "engine": "faiss",
                        "space_type": "l2",
                        "name": "hnsw",
                        "parameters": {"ef_construction": 512, "m": 16},
                    },
                },
                "text": {"type": "text"},
                "metadata": {"type": "text"},
            }
        },
    }

    try:
        client.indices.create(index=INDEX_NAME, body=index_body)
        print(f"✅ Created vector index: {INDEX_NAME}")
    except Exception as e:
        if "already_exists" in str(e).lower() or "resource_already_exists" in str(e).lower():
            print(f"✅ Vector index already exists: {INDEX_NAME}")
        else:
            raise


def create_knowledge_base(role_arn, collection_arn):
    """Create Bedrock Knowledge Base."""
    # Check if KB already exists
    existing = bedrock_agent.list_knowledge_bases()
    for kb in existing["knowledgeBaseSummaries"]:
        if kb["name"] == KB_NAME:
            print(f"✅ Knowledge Base already exists: {kb['knowledgeBaseId']}")
            return kb["knowledgeBaseId"]

    resp = bedrock_agent.create_knowledge_base(
        name=KB_NAME,
        roleArn=role_arn,
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": f"arn:aws:bedrock:{AWS_REGION}::foundation-model/{EMBEDDING_MODEL_ID}",
            },
        },
        storageConfiguration={
            "type": "OPENSEARCH_SERVERLESS",
            "opensearchServerlessConfiguration": {
                "collectionArn": collection_arn,
                "vectorIndexName": INDEX_NAME,
                "fieldMapping": {
                    "vectorField": "vector",
                    "textField": "text",
                    "metadataField": "metadata",
                },
            },
        },
    )
    kb_id = resp["knowledgeBase"]["knowledgeBaseId"]
    print(f"✅ Created Knowledge Base: {kb_id}")

    # Wait for KB to be active
    for _ in range(30):
        kb_status = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)
        if kb_status["knowledgeBase"]["status"] == "ACTIVE":
            break
        time.sleep(5)

    return kb_id


def create_data_source(kb_id):
    """Create S3 data source for the knowledge base."""
    existing = bedrock_agent.list_data_sources(knowledgeBaseId=kb_id)
    for ds in existing["dataSourceSummaries"]:
        if ds["name"] == "telecom-rag-s3-source":
            print(f"✅ Data source already exists: {ds['dataSourceId']}")
            return ds["dataSourceId"]

    resp = bedrock_agent.create_data_source(
        knowledgeBaseId=kb_id,
        name="telecom-rag-s3-source",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": f"arn:aws:s3:::{S3_BUCKET}",
                "inclusionPrefixes": [S3_PREFIX],
            },
        },
        vectorIngestionConfiguration={
            "chunkingConfiguration": {
                "chunkingStrategy": "FIXED_SIZE",
                "fixedSizeChunkingConfiguration": {
                    "maxTokens": 1000,
                    "overlapPercentage": 10,
                },
            }
        },
    )
    ds_id = resp["dataSource"]["dataSourceId"]
    print(f"✅ Created data source: {ds_id}")
    return ds_id


def start_ingestion(kb_id, ds_id):
    """Start ingestion job to index documents."""
    resp = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
    )
    job_id = resp["ingestionJob"]["ingestionJobId"]
    print(f"⏳ Started ingestion job: {job_id}")

    # Monitor progress
    while True:
        status = bedrock_agent.get_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            ingestionJobId=job_id,
        )
        job_status = status["ingestionJob"]["status"]
        stats = status["ingestionJob"].get("statistics", {})
        print(f"   Status: {job_status} | Scanned: {stats.get('numberOfDocumentsScanned', 0)} | Indexed: {stats.get('numberOfNewDocumentsIndexed', 0) + stats.get('numberOfModifiedDocumentsIndexed', 0)}")
        if job_status in ("COMPLETE", "FAILED"):
            break
        time.sleep(15)

    if job_status == "COMPLETE":
        print("✅ Ingestion complete!")
    else:
        print(f"❌ Ingestion failed: {status['ingestionJob'].get('failureReasons', 'Unknown')}")
    return job_status


def setup_all():
    """Run full AWS setup."""
    print("=" * 60)
    print("🚀 TELECOM RAG — AWS INFRASTRUCTURE SETUP")
    print("=" * 60)

    print("\n📦 Step 1: Creating S3 bucket...")
    create_s3_bucket()

    print("\n📤 Step 2: Uploading documents...")
    upload_documents()

    print("\n🔑 Step 3: Creating IAM role...")
    role_arn = create_kb_role()

    print("\n🔍 Step 4: Creating OpenSearch Serverless collection...")
    collection_arn, collection_id = create_opensearch_collection()

    print("\n📊 Step 5: Creating vector index...")
    try:
        create_opensearch_index(collection_id)
    except Exception as e:
        print(f"⚠️  Index creation note: {e}")
        print("   (Will be auto-created by Bedrock KB if needed)")

    print("\n🧠 Step 6: Creating Bedrock Knowledge Base...")
    kb_id = create_knowledge_base(role_arn, collection_arn)

    print("\n📁 Step 7: Creating data source...")
    ds_id = create_data_source(kb_id)

    print("\n🔄 Step 8: Starting document ingestion...")
    start_ingestion(kb_id, ds_id)

    # Save KB ID for the app
    config_path = os.path.join(os.path.dirname(__file__), ".kb_config")
    with open(config_path, "w") as f:
        json.dump({"kb_id": kb_id, "ds_id": ds_id}, f)
    print(f"\n✅ Configuration saved to {config_path}")
    print(f"   Knowledge Base ID: {kb_id}")
    print("=" * 60)
    return kb_id, ds_id


if __name__ == "__main__":
    setup_all()
