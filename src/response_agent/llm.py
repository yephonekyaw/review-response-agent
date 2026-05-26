"""LLM provider abstraction: Gemini (default) or Ollama fallback.

Both providers expose the same surface:
    generate(prompt, *, json_mode=False, temperature=0.4) -> str
    embed(texts: list[str]) -> np.ndarray   # shape (n, dim)
"""
from __future__ import annotations

import re
import time
from typing import Callable, Protocol, TypeVar

import numpy as np

from .config import Config, load_config

T = TypeVar("T")


def _parse_retry_delay(err: Exception, default: float) -> float:
    """Pull `retryDelay` out of a Gemini 429 error (e.g. '40s'); fall back to default."""
    msg = str(err)
    m = re.search(r"retry in ([0-9.]+)s", msg) or re.search(r"'retryDelay': '([0-9.]+)s'", msg)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return default


class _GeminiThrottle:
    """Min-interval gate + 429 retry-with-backoff for the Gemini free tier."""

    def __init__(self, min_interval: float, max_retries: int):
        self._min_interval = max(0.0, min_interval)
        self._max_retries = max(0, max_retries)
        self._last_call = 0.0

    def _wait_min_interval(self) -> None:
        if self._min_interval <= 0:
            return
        gap = time.monotonic() - self._last_call
        if gap < self._min_interval:
            time.sleep(self._min_interval - gap)

    def call(self, fn: Callable[[], T]) -> T:
        attempt = 0
        while True:
            self._wait_min_interval()
            try:
                result = fn()
                self._last_call = time.monotonic()
                return result
            except Exception as err:  # noqa: BLE001 — google.genai.errors.ClientError
                if "429" not in str(err) or attempt >= self._max_retries:
                    raise
                delay = _parse_retry_delay(err, default=15.0) + 1.0
                attempt += 1
                print(f"      [rate-limit] sleeping {delay:.1f}s (retry {attempt}/{self._max_retries})")
                time.sleep(delay)
                self._last_call = time.monotonic()


class LLMProvider(Protocol):
    def generate(self, prompt: str, *, json_mode: bool = False, temperature: float = 0.4) -> str: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class GeminiProvider:
    def __init__(self, cfg: Config):
        from google import genai
        from google.genai import types
        self._types = types
        self._client = genai.Client(api_key=cfg.gemini_api_key)
        self._chat_model = cfg.gemini_chat_model
        self._embed_model = cfg.gemini_embed_model
        self._throttle = _GeminiThrottle(cfg.gemini_min_interval, cfg.gemini_max_retries)

    def generate(self, prompt: str, *, json_mode: bool = False, temperature: float = 0.4) -> str:
        cfg_kwargs = {"temperature": temperature}
        if json_mode:
            cfg_kwargs["response_mime_type"] = "application/json"
        resp = self._throttle.call(lambda: self._client.models.generate_content(
            model=self._chat_model,
            contents=prompt,
            config=self._types.GenerateContentConfig(**cfg_kwargs),
        ))
        return resp.text.strip()

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._throttle.call(
            lambda: self._client.models.embed_content(model=self._embed_model, contents=texts)
        )
        return np.array([e.values for e in resp.embeddings], dtype=np.float32)


class OllamaProvider:
    def __init__(self, cfg: Config):
        import ollama
        self._client = ollama.Client(host=cfg.ollama_host)
        self._chat_model = cfg.ollama_chat_model
        self._embed_model = cfg.ollama_embed_model

    def generate(self, prompt: str, *, json_mode: bool = False, temperature: float = 0.4) -> str:
        resp = self._client.chat(
            model=self._chat_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
            format="json" if json_mode else "",
        )
        return resp["message"]["content"].strip()

    def embed(self, texts: list[str]) -> np.ndarray:
        # Ollama's embed endpoint accepts a list of inputs.
        resp = self._client.embed(model=self._embed_model, input=texts)
        return np.array(resp["embeddings"], dtype=np.float32)


def get_provider(cfg: Config | None = None) -> LLMProvider:
    cfg = cfg or load_config()
    if cfg.provider == "gemini":
        return GeminiProvider(cfg)
    return OllamaProvider(cfg)
