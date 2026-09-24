MAX_HISTORY_TURNS = 6

conversation_store = {}


def get_history(session_id):
    return conversation_store.get(
        session_id,
        [],
    )


def add_message(
    session_id,
    role,
    content,
):
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    conversation_store[session_id].append(
        {
            "role": role,
            "content": content,
        }
    )

    conversation_store[session_id] = (
        conversation_store[session_id][-MAX_HISTORY_TURNS:]
    )


def clear_history(session_id):
    conversation_store.pop(
        session_id,
        None,
    )