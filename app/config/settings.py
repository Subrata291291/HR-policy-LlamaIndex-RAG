import os
from pathlib import Path

from dotenv import load_dotenv


# ==============================
# Project Configuration
# ==============================

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# ==============================
# API Keys
# ==============================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# ==============================
# LLM Models
# ==============================

GROQ_MODEL = os.getenv("GROQ_MODEL")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


# ==============================
# Embedding
# ==============================

EMBEDDING_MODEL = "gemini-embedding-001"


# ==============================
# Pinecone
# ==============================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "hr-policy-rag"
)


# ==============================
# Application
# ==============================

APP_NAME = "HR Policy RAG"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


RETRIEVAL_THRESHOLD = float(
    os.getenv("RETRIEVAL_THRESHOLD", "0.60")
)