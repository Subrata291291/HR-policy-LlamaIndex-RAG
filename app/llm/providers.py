from llama_index.llms.openai_like import OpenAILike

from app.config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
)


def create_groq_llm():
    return OpenAILike(
        model=GROQ_MODEL,
        api_base="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        context_window=131072,
        is_chat_model=True,
    )