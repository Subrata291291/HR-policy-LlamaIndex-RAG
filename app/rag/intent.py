from app.llm.providers import create_groq_llm


GREETING = "GREETING"
POLICY = "POLICY"
OUT_OF_SCOPE = "OUT_OF_SCOPE"


def classify_intent(query: str) -> str:
    llm = create_groq_llm()

    prompt = f"""
You are an intent classifier for an HR Policy Assistant.

The assistant has ONLY three possible intents.

==================================================
1. GREETING
==================================================

Use GREETING for simple conversational messages such as:

- hi
- hello
- hey
- good morning
- good afternoon
- how are you?
- thanks
- thank you
- bye

These are allowed only for basic conversation.

==================================================
2. POLICY
==================================================

Use POLICY when the user is asking about company HR policies,
rules, benefits, employee procedures, attendance, leave,
work from home, travel, working hours, allowances, or
other employment policies.

Examples:

- What is the annual leave entitlement?
- How many sick leave days are available?
- Can I work from home?
- What is the internet allowance?
- What are the working hours?
- What is the travel meal allowance?
- Can unused leave be carried forward?

==================================================
3. OUT_OF_SCOPE
==================================================

Use OUT_OF_SCOPE for everything else.

Examples:

- What is Python?
- What is React?
- What is AI?
- Tell me a joke.
- What is the weather?
- Who is the Prime Minister?
- Explain JavaScript.
- Write a poem.
- Solve this math problem.

IMPORTANT:

The assistant must NOT answer OUT_OF_SCOPE questions.

==================================================

USER MESSAGE
==================================================

{query}

==================================================

RETURN ONLY ONE OF THESE VALUES:

GREETING
POLICY
OUT_OF_SCOPE

Do not return anything else.
"""

    response = llm.complete(
        prompt,
        temperature=0,
    )

    intent = str(response).strip().upper()

    if intent not in {
        GREETING,
        POLICY,
        OUT_OF_SCOPE,
    }:
        return OUT_OF_SCOPE

    return intent