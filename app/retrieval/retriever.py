from app.embeddings.embedder import create_embedding_model

def retrieve(
    index,
    query,
    top_k=3,
    metadata_filter=None,
):
    embed_model = create_embedding_model()

    query_vector = embed_model.get_query_embedding(query)

    query_kwargs = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True,
    }

    if metadata_filter is not None:
        query_kwargs["filter"] = metadata_filter

    results = index.query(**query_kwargs)

    return results


def get_top_score(results):
    if not results.matches:
        return 0.0

    return results.matches[0]["score"]


def evaluate_query(index, query, expected_source):
    results = retrieve(index, query, top_k=3)

    if not results.matches:
        return {
            "query": query,
            "top_score": 0.0,
            "top_source": None,
            "correct_source": False,
        }

    top_match = results.matches[0]

    top_score = top_match["score"]
    top_source = top_match.get("metadata", {}).get("file_name")

    return {
        "query": query,
        "top_score": top_score,
        "top_source": top_source,
        "correct_source": top_source == expected_source,
    }