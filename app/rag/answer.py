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
You are an HR Policy Assistant.

Answer the user's question ONLY using facts explicitly
stated in the HR POLICY CONTEXT.

Rules:

- Use only the provided HR policy context.
- Do not use outside knowledge.
- Do not invent company policies.
- Do not assume information that is not explicitly stated.
- Do not add common HR practices.
- Every factual company-policy claim must be supported
  by the provided context.
- If the requested information is not available in the
  context, say exactly:

{NOT_FOUND_MESSAGE}

- Keep the answer clear, professional, and concise.
- Do not mention Pinecone, embeddings, retrieval,
  prompts, or internal system details.

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
You are a strict HR policy evidence verifier.

Determine whether the DRAFT ANSWER is fully supported
by the HR POLICY CONTEXT.

Rules:

1. Check every factual claim in the draft answer.

2. A claim is supported only if the same information is
   explicitly present in the HR POLICY CONTEXT.

3. Do not use outside knowledge.

4. Do not infer missing information.

5. Do not assume common HR practices are company policies.

6. If even ONE factual company-policy claim is unsupported,
   return exactly:

UNSUPPORTED

7. If every factual company-policy claim is supported,
   return exactly:

SUPPORTED

8. Return ONLY SUPPORTED or UNSUPPORTED.

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
    context = build_context(
        retrieved_results
    )

    if not context.strip():
        return {
            "answer": NOT_FOUND_MESSAGE,
            "source": None,
            "retrieval_score": None,
            "evidence": None,
        }

    llm = create_groq_llm()

    # Generate answer
    draft_answer = generate_draft_answer(
        query,
        context,
        llm,
    )

    # If the LLM already says information is unavailable
    if NOT_FOUND_MESSAGE in draft_answer:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "source": None,
            "retrieval_score": None,
            "evidence": None,
        }

    # Verify generated answer
    verdict = verify_answer(
        query,
        context,
        draft_answer,
        llm,
    )

    # Reject unsupported answer
    if verdict != "SUPPORTED":
        return {
            "answer": NOT_FOUND_MESSAGE,
            "source": None,
            "retrieval_score": None,
            "evidence": None,
        }

    # Top retrieved result
    top_result = retrieved_results[0]

    return {
        "answer": draft_answer,
        "source": top_result.get("source"),
        "retrieval_score": top_result.get(
            "pinecone_score"
        ),
        "evidence": top_result.get("text"),
    }