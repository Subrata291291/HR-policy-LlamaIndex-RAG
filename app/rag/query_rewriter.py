from app.llm.providers import create_groq_llm


def rewrite_query(query, history):
    if not history:
        return query

    conversation = []

    for message in history:
        role = message.get("role", "")
        content = message.get("content", "")

        if content:
            conversation.append(
                f"{role}: {content}"
            )

    history_text = "\n".join(conversation)

    llm = create_groq_llm()

    prompt = f"""
You rewrite follow-up questions for an HR Policy RAG system.

Your job is to convert the user's CURRENT QUESTION
into a standalone search query that can be understood
without the previous conversation.

Rules:

- Use the previous conversation only to resolve references
  such as "it", "they", "that", "how many", "what about it",
  or similar follow-up wording.
- Preserve the user's actual intent.
- Do not answer the question.
- Do not add information that is not present in the
  current question or conversation.
- Do not invent HR policies.
- Keep the rewritten query concise.
- If the current question is already clear and standalone,
  return it unchanged.
- Return ONLY the rewritten query.

PREVIOUS CONVERSATION:
{history_text}

CURRENT QUESTION:
{query}

REWRITTEN QUERY:
"""

    response = llm.complete(
        prompt,
        temperature=0,
    )

    rewritten_query = str(
        response
    ).strip()

    if not rewritten_query:
        return query

    return rewritten_query