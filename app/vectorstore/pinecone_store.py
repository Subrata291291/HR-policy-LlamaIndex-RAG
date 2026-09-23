from pinecone import Pinecone, ServerlessSpec

from app.config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)


EMBEDDING_DIMENSION = 3072


def create_pinecone_client():
    return Pinecone(api_key=PINECONE_API_KEY)


def create_index_if_not_exists(pc):
    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )


def get_index(pc):
    return pc.Index(PINECONE_INDEX_NAME)


def get_policy_type(file_name):
    policy_types = {
        "Leave_Policy.pdf": "leave",
        "Work_From_Home_Policy.pdf": "work_from_home",
        "Employee_Benefits.pdf": "benefits",
        "Attendance_Policy.pdf": "attendance",
        "Travel_Policy.pdf": "travel",
    }

    return policy_types.get(file_name, "unknown")


def upsert_node(index, node, vector):
    file_name = node.metadata.get("file_name", "unknown")

    metadata = {
        "text": node.text,
        "file_name": file_name,
        "file_path": node.metadata.get("file_path", "unknown"),
        "policy_type": get_policy_type(file_name),
    }

    index.upsert(
        vectors=[
            {
                "id": node.node_id,
                "values": vector,
                "metadata": metadata,
            }
        ]
    )


def upsert_nodes(index, nodes, embed_model):
    vectors = []

    for node in nodes:
        vector = embed_model.get_text_embedding(node.text)

        file_name = node.metadata.get("file_name", "unknown")

        metadata = {
            "text": node.text,
            "file_name": file_name,
            "file_path": node.metadata.get("file_path", "unknown"),
            "policy_type": get_policy_type(file_name),
        }

        vectors.append(
            {
                "id": node.node_id,
                "values": vector,
                "metadata": metadata,
            }
        )

    index.upsert(vectors=vectors)

    return len(vectors)