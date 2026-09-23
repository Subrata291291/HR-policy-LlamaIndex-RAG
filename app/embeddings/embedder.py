from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from app.config.settings import GOOGLE_API_KEY


EMBEDDING_MODEL = "gemini-embedding-001"


def create_embedding_model():
    embed_model = GoogleGenAIEmbedding(
        model_name=EMBEDDING_MODEL,
        api_key=GOOGLE_API_KEY,
    )

    return embed_model