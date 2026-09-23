from app.llm.providers import create_groq_llm


NOT_FOUND_MESSAGE = (
    "I could not find this information in the provided HR policies."
)


def build_context(retrieved_results):
    contexts = []

    for result in retrieved_results:
        text = result.get("text", "")

        if text:
            contexts.append(text)

    return "\n\n".join(contexts)


def generate_draft_answer(query, context, llm):
    prompt = f"""
You are an HR policy assistant.

Answer the question ONLY using facts explicitly stated in the
HR POLICY CONTEXT.

Rules:
- Do not use outside knowledge.
- Do not infer missing information.
- Do not add common HR practices.
- Every factual claim must be supported by the context.
- If the context does not contain enough information, say:
{NOT_FOUND_MESSAGE}

HR POLICY CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    response = llm.complete(
        prompt,
        temperature=0,
    )

    return str(response).strip()


def verify_answer(query, context, draft_answer, llm):
    prompt = f"""
You are a strict evidence verifier.

Your job is to determine whether the DRAFT ANSWER is fully supported
by the HR POLICY CONTEXT.

Rules:

1. Check EVERY factual claim in the draft answer.
2. A claim is supported only if the same information is explicitly
   present in the context.
3. Do not use outside knowledge.
4. Do not infer or assume missing information.
5. If even ONE factual claim is unsupported, return exactly:
UNSUPPORTED
6. If every factual claim is explicitly supported, return exactly:
SUPPORTED

HR POLICY CONTEXT:
{context}

USER QUESTION:
{query}

DRAFT ANSWER:
{draft_answer}

VERDICT:
"""

    response = llm.complete(
        prompt,
        temperature=0,
    )

    return str(response).strip().upper()


def generate_answer(query, retrieved_results):
    context = build_context(retrieved_results)

    if not context.strip():
        return {
            "answer": NOT_FOUND_MESSAGE,
            "source": None,
            "retrieval_score": None,
            "evidence": None,
        }

    llm = create_groq_llm()

    draft_answer = generate_draft_answer(
        query,
        context,
        llm,
    )

    if NOT_FOUND_MESSAGE in draft_answer:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "source": None,
            "retrieval_score": None,
            "evidence": None,
        }

    verdict = verify_answer(
        query,
        context,
        draft_answer,
        llm,
    )

    if verdict != "SUPPORTED":
        return {
            "answer": NOT_FOUND_MESSAGE,
            "source": None,
            "retrieval_score": None,
            "evidence": None,
        }

    top_result = retrieved_results[0]

    return {
        "answer": draft_answer,
        "source": top_result.get("source"),
        "retrieval_score": top_result.get(
            "pinecone_score"
        ),
        "evidence": top_result.get("text"),
    }