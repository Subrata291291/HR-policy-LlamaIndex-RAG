from app.vectorstore.pinecone_store import (
    create_pinecone_client,
    get_index,
)

from app.retrieval.retriever import retrieve
from app.rag.answer import generate_answer
from app.config.settings import RETRIEVAL_THRESHOLD


NOT_FOUND_MESSAGE = (
    "I could not find this information in the provided HR policies."
)


def create_rag_pipeline():
    pc = create_pinecone_client()
    index = get_index(pc)
    return index


def ask(
    query,
    top_k_retrieval=5,
    debug=False,
):
    index = create_rag_pipeline()

    # Step 1: Pinecone Retrieval
    results = retrieve(
        index,
        query,
        top_k=top_k_retrieval,
    )

    if debug:
        print("\n" + "=" * 70)
        print("1. PINECONE RETRIEVAL")
        print("=" * 70)
        print(f"Query: {query}")
        print(f"Retrieved candidates: {len(results.matches)}")

        for i, match in enumerate(results.matches, start=1):
            metadata = match.get("metadata", {})

            print(f"\nCandidate {i}")
            print(f"Source         : {metadata.get('file_name')}")
            print(f"Pinecone score : {match['score']}")
            print(
                f"Text preview   : "
                f"{metadata.get('text', '')[:250]}..."
            )

    # Step 2: Check if anything was retrieved
    if not results.matches:
        if debug:
            print("\nNo retrieval results.")

        return {
            "answer": NOT_FOUND_MESSAGE,
            "source": None,
            "retrieval_score": None,
            "evidence": None,
        }

    # Step 3: Retrieval threshold
    top_result = results.matches[0]
    top_score = top_result["score"]

    if debug:
        print("\n" + "=" * 70)
        print("2. RETRIEVAL THRESHOLD CHECK")
        print("=" * 70)
        print(f"Top Pinecone score : {top_score}")
        print(f"Threshold          : {RETRIEVAL_THRESHOLD}")

        if top_score >= RETRIEVAL_THRESHOLD:
            print("Decision            : PASS")
        else:
            print("Decision            : REJECT")

    if top_score < RETRIEVAL_THRESHOLD:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "source": None,
            "retrieval_score": None,
            "evidence": None,
        }

    # Step 4: LLM generation + verification
    if debug:
        print("\n" + "=" * 70)
        print("3. LLM GENERATION + VERIFICATION")
        print("=" * 70)
        print("Threshold passed.")
        print("Sending retrieved context to LLM...")

    retrieved_results = []

    for match in results.matches:
        metadata = match.get("metadata", {})

        retrieved_results.append(
            {
                "text": metadata.get("text", ""),
                "source": metadata.get(
                    "file_name",
                    "unknown",
                ),
                "pinecone_score": match["score"],
            }
        )

    answer = generate_answer(
        query,
        retrieved_results,
    )

    # Step 5: Final result
    if debug:
        print("\n" + "=" * 70)
        print("4. FINAL RESULT")
        print("=" * 70)
        print(f"Source          : {answer['source']}")
        print(
            f"Retrieval score : "
            f"{top_score}"
        )
        print(f"Answer          : {answer['answer']}")

    return {
        "answer": answer["answer"],
        "source": answer["source"],
        "retrieval_score": top_score,
        "evidence": answer["evidence"],
    }