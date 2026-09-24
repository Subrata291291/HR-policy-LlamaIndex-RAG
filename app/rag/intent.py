from app.llm.providers import create_groq_llm


GREETING = "GREETING"
POLICY = "POLICY"
OUT_OF_SCOPE = "OUT_OF_SCOPE"


def classify_simple_intent(query: str):
    text = query.strip().lower()

    greetings = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "bye",
        "good morning",
        "good afternoon",
        "good evening",
    }

    if text in greetings:
        return GREETING

    return None


def is_obvious_policy_query(query: str):
    text = query.strip().lower()

    policy_phrases = {
        # Leave
        "annual leave",
        "paid leave",
        "sick leave",
        "casual leave",
        "leave policy",
        "leave days",
        "carry forward",
        "leave balance",

        # Work from home
        "work from home",
        "work from home policy",
        "wfh",
        "remote work",
        "remote working",

        # Benefits
        "health insurance",
        "retirement contribution",
        "retirement contributions",
        "learning reimbursement",
        "course reimbursement",
        "education reimbursement",
        "internet allowance",
        "employee benefits",
        "benefits policy",

        # Attendance
        "attendance policy",
        "attendance",
        "working hours",
        "work hours",
        "office hours",
        "late arrival",
        "late coming",
        "early departure",

        # Travel
        "travel policy",
        "business travel",
        "travel reimbursement",
        "travel expenses",
        "hotel limit",
        "hotel expenses",
        "meal allowance",
        "travel allowance",
        "international travel",

        # HR procedures
        "hr policy",
        "company policy",
        "employee policy",
        "manager approval",
        "hr portal",
    }

    for phrase in policy_phrases:
        if phrase in text:
            return True

    return False


def classify_intent(query: str) -> str:

    # 1. Handle obvious greetings locally
    simple_intent = classify_simple_intent(query)

    if simple_intent is not None:
        return simple_intent

    # 2. Handle obvious HR policy questions locally
    if is_obvious_policy_query(query):
        return POLICY

    # 3. Use LLM only for ambiguous questions
    llm = create_groq_llm()

    prompt = f"""
Classify the user's message into exactly ONE category.

GREETING:
hello, hi, hey, good morning, thanks, thank you, bye,
or simple casual conversation.

POLICY:
company HR policies, leave, benefits, attendance,
work from home, working hours, travel, allowances,
or employee procedures.

OUT_OF_SCOPE:
anything unrelated to company HR policies.

USER MESSAGE:
{query}

Return ONLY one of:
GREETING
POLICY
OUT_OF_SCOPE
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