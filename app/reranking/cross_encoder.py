from sentence_transformers import CrossEncoder
from huggingface_hub import snapshot_download


MODEL_REPO = "cross-encoder/ettin-reranker-68m-v1"

_reranker = None


def create_reranker():
    global _reranker

    if _reranker is None:
        model_path = snapshot_download(
            repo_id=MODEL_REPO,
        )

        _reranker = CrossEncoder(model_path)

    return _reranker


def rerank_results(query, results, top_k=3):
    reranker = create_reranker()

    candidates = []

    for match in results.matches:
        text = match.get("metadata", {}).get("text", "")

        if text:
            candidates.append(
                {
                    "text": text,
                    "source": match.get(
                        "metadata", {}
                    ).get("file_name", "unknown"),
                    "pinecone_score": match["score"],
                }
            )

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = []

    for candidate, score in zip(candidates, scores):
        ranked.append(
            {
                "reranker_score": float(score),
                "pinecone_score": candidate["pinecone_score"],
                "source": candidate["source"],
                "text": candidate["text"],
            }
        )

    ranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return ranked[:top_k]