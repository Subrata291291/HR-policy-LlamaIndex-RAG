from app.vectorstore.pinecone_store import (
    create_pinecone_client,
    get_index,
)
import time
from app.retrieval.retriever import retrieve

from app.rag.answer import generate_answer

from app.rag.intent import (
    classify_intent,
    GREETING,
    OUT_OF_SCOPE,
)

from app.rag.memory import (
    get_history,
    add_message,
)

from app.rag.query_rewriter import rewrite_query

NOT_FOUND_MESSAGE = (
    "I could not find this information in the provided HR policies."
)

GREETING_MESSAGE = (
    "Hi! 👋 How can I help you with our HR policies today?"
)

OUT_OF_SCOPE_MESSAGE = (
    "I'm designed to answer questions about HR policies, "
    "employee benefits, leave, attendance, work from home, "
    "travel, and related company policies. "
    "I can't help with general questions outside this scope."
)


def create_rag_pipeline():
    pc = create_pinecone_client()
    index = get_index(pc)

    return index


def ask(
    query,
    session_id,
    top_k_retrieval=3,
    debug=False,
):
    start_time = time.perf_counter()
    history = get_history(session_id)
    intent_start = time.perf_counter()

    intent = classify_intent(query)

    intent_time = (
        time.perf_counter() - intent_start
    )

    if debug:
        print("\n" + "=" * 70)
        print("1. INTENT GUARD")
        print("=" * 70)
        print(f"Query  : {query}")
        print(f"Intent : {intent}")
        print(f"Session: {session_id}")
        print(f"History messages: {len(history)}")

    if intent == GREETING:
        answer_text = GREETING_MESSAGE

        add_message(
            session_id,
            "user",
            query,
        )

        add_message(
            session_id,
            "assistant",
            answer_text,
        )

        return {
            "answer": answer_text,
            "source": None,
            "retrieval_score": None,
            "evidence": None,
        }

    if intent == OUT_OF_SCOPE:
        return {
            "answer": OUT_OF_SCOPE_MESSAGE,
            "source": None,
            "retrieval_score": None,
            "evidence": None,
        }


    # ==========================================================
    # STEP 2: POLICY QUESTION
    # ==========================================================

    if debug:
        print("Decision : POLICY")
        print("Continuing to Pinecone retrieval.")

    index = create_rag_pipeline()

    rewrite_start = time.perf_counter()

    if history:
        search_query = rewrite_query(
            query,
            history,
        )
    else:
        search_query = query

    rewrite_time = (
        time.perf_counter() - rewrite_start
    )

    if debug:
        print(
            f"Search query: {search_query}"
        )


    retrieval_start = time.perf_counter()

    results = retrieve(
        index,
        search_query,
        top_k=top_k_retrieval,
    )

    retrieval_time = (
        time.perf_counter() - retrieval_start
    )

    retrieval_time = (
        time.perf_counter() - retrieval_start
    )

    # ==========================================================
    # STEP 3: PINECONE RETRIEVAL
    # ==========================================================

    if debug:
        print("\n" + "=" * 70)
        print("2. PINECONE RETRIEVAL")
        print("=" * 70)

        print(f"Query: {query}")
        print(
            f"Retrieved candidates: "
            f"{len(results.matches)}"
        )

        for i, match in enumerate(
            results.matches,
            start=1,
        ):
            metadata = match.get(
                "metadata",
                {},
            )

            print(f"\nCandidate {i}")

            print(
                f"Source         : "
                f"{metadata.get('file_name')}"
            )

            print(
                f"Pinecone score : "
                f"{match['score']}"
            )

            print(
                f"Text preview   : "
                f"{metadata.get('text', '')[:250]}..."
            )

    # ==========================================================
    # STEP 4: NO RETRIEVAL RESULTS
    # ==========================================================

    if not results.matches:

        if debug:
            print("\nNo retrieval results.")

        return {
            "answer": NOT_FOUND_MESSAGE,
            "source": None,
            "retrieval_score": None,
            "evidence": None,
        }

    # ==========================================================
    # STEP 5: PREPARE RETRIEVED CONTEXT
    # ==========================================================

    top_result = results.matches[0]

    top_score = top_result["score"]

    retrieved_results = []

    for match in results.matches:

        metadata = match.get(
            "metadata",
            {},
        )

        retrieved_results.append(
            {
                "text": metadata.get(
                    "text",
                    "",
                ),
                "source": metadata.get(
                    "file_name",
                    "unknown",
                ),
                "pinecone_score": match["score"],
            }
        )

    # ==========================================================
    # STEP 6: LLM GENERATION + POLICY VERIFICATION
    # ==========================================================

    if debug:
        print("\n" + "=" * 70)
        print("3. LLM GENERATION + VERIFICATION")
        print("=" * 70)

        print(
            "Sending HR policy context to LLM..."
        )

    generation_start = time.perf_counter()

    answer = generate_answer(
        query,
        retrieved_results,
    )

    generation_time = (
        time.perf_counter() - generation_start
    )

    # ==========================================================
    # STEP 7: FINAL RESULT
    # ==========================================================

    if debug:
        print("\n" + "=" * 70)
        print("4. FINAL RESULT")
        print("=" * 70)

        print(
            f"Source          : "
            f"{answer['source']}"
        )

        print(
            f"Retrieval score : "
            f"{top_score}"
        )

        print(
            f"Answer          : "
            f"{answer['answer']}"
        )

    add_message(
        session_id,
        "user",
        query,
    )

    add_message(
        session_id,
        "assistant",
        answer["answer"],
    )

    elapsed_time = (
        time.perf_counter() - start_time
    )

    if debug:
        print("\n" + "=" * 70)
        print("5. PERFORMANCE BREAKDOWN")
        print("=" * 70)

        print(
            f"Intent classification : "
            f"{intent_time:.2f} sec"
        )

        print(
            f"Query rewriting      : "
            f"{rewrite_time:.2f} sec"
        )

        print(
            f"Pinecone retrieval    : "
            f"{retrieval_time:.2f} sec"
        )

        print(
            f"Generation + verify   : "
            f"{generation_time:.2f} sec"
        )

        print(
            f"Total pipeline time   : "
            f"{elapsed_time:.2f} sec"
        )

    return {
        "answer": answer["answer"],
        "source": answer["source"],
        "retrieval_score": top_score,
        "evidence": answer["evidence"],
    }