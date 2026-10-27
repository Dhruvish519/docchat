"""Thin wrapper around the Google Gemini API (google-genai SDK).

Two kinds of model are used:
- an embedding model, which turns text into a vector of numbers (meaning -> geometry)
- a generative LLM, which writes the final answer from the retrieved passages
"""

import json
import time

import numpy as np
from google import genai
from google.genai import errors, types

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768        # gemini-embedding-001 supports 128-3072; 768 is plenty and small
EMBED_BATCH = 100      # max texts per embedding request


def _with_retry(fn, retries: int = 4):
    """Retry on rate limits (429) and temporary server errors (5xx)."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if attempt < retries and (code == 429 or (code and code >= 500)):
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            raise


class GeminiClient:
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    # ---------- embeddings ----------
    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            resp = _with_retry(lambda: self.client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type,
                                                output_dimensionality=EMBED_DIM),
            ))
            vectors.extend(e.values for e in resp.embeddings)
        return np.array(vectors, dtype="float32")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    # ---------- generation ----------
    def generate(self, prompt: str, temperature: float = 0.2, system: str | None = None) -> str:
        resp = _with_retry(lambda: self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature,
                                               system_instruction=system),
        ))
        return (resp.text or "").strip()

    def generate_stream(self, prompt: str, temperature: float = 0.2, system: str | None = None):
        """Yield the answer piece by piece (gives the 'typing' effect in the UI)."""
        stream = _with_retry(lambda: self.client.models.generate_content_stream(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature,
                                               system_instruction=system),
        ))
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    def generate_json(self, prompt: str) -> dict:
        resp = _with_retry(lambda: self.client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3,
                                               response_mime_type="application/json"),
        ))
        text = (resp.text or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            return json.loads(text[start:end + 1]) if start >= 0 else {}
