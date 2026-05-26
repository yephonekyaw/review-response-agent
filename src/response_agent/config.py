"""Environment-driven configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    provider: str            # "gemini" or "ollama"
    gemini_api_key: str | None
    gemini_chat_model: str
    gemini_embed_model: str
    gemini_min_interval: float
    gemini_max_retries: int
    ollama_host: str
    ollama_chat_model: str
    ollama_embed_model: str


def load_config() -> Config:
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower().strip()
    if provider not in {"gemini", "ollama"}:
        raise ValueError(f"LLM_PROVIDER must be 'gemini' or 'ollama', got {provider!r}")

    cfg = Config(
        provider=provider,
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        gemini_chat_model=os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
        gemini_embed_model=os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004"),
        gemini_min_interval=float(os.environ.get("GEMINI_MIN_INTERVAL", "6.5")),
        gemini_max_retries=int(os.environ.get("GEMINI_MAX_RETRIES", "5")),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        ollama_chat_model=os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1:8b"),
        ollama_embed_model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )

    if provider == "gemini" and not cfg.gemini_api_key:
        raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set.")
    return cfg
