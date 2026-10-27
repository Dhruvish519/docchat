"""Step 3 of the RAG pipeline: put it all together.

    question
      -> (rewrite follow-ups into a standalone question using chat history)
      -> embed the question
      -> FAISS similarity search: top-k most relevant chunks
      -> build a prompt with numbered passages
      -> Gemini writes a grounded answer that cites [1], [2] ...
"""

from . import prompts
from .llm import EMBED_MODEL, EMBED_DIM, GeminiClient
from .pdf_utils import Chunk, pdf_to_chunks
from .vector_store import VectorStore, cache_key, load_cached, save_cached


def index_pdf(client: GeminiClient, pdf_bytes: bytes, file_name: str,
              chunk_size: int = 1000, overlap: int = 200):
    """Chunk + embed one PDF (or load it from the cache). Returns (chunks, vectors, pages, cached)."""
    key = cache_key(pdf_bytes, file_name, EMBED_MODEL, EMBED_DIM, chunk_size, overlap)
    cached = load_cached(key)
    if cached:
        return (*cached, True)
    chunks, pages = pdf_to_chunks(pdf_bytes, file_name, chunk_size, overlap)
    if not chunks:
        return [], None, pages, False
    vectors = client.embed_documents([c.text for c in chunks])
    save_cached(key, chunks, vectors, pages)
    return chunks, vectors, pages, False


def condense_question(client: GeminiClient, question: str, history: list[dict], turns: int = 3) -> str:
    """Turn 'what about its side effects?' into 'What are the side effects of drug X?'."""
    recent = [m for m in history if m["role"] in ("user", "assistant")][-turns * 2:]
    if not recent:
        return question
    hist = "\n".join(f"{m['role'].title()}: {m['content'][:600]}" for m in recent)
    rewritten = client.generate(prompts.REWRITE.format(history=hist, question=question), temperature=0)
    return rewritten.strip().strip('"') or question


def build_context(results: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.file_name}, page {c.page})\n{c.text}"
        for i, (c, _) in enumerate(results, start=1)
    )


def retrieve(client: GeminiClient, store: VectorStore, question: str, k: int):
    return store.search(client.embed_query(question), k=k)


def answer_stream(client: GeminiClient, question: str, results, temperature: float = 0.2):
    prompt = prompts.ANSWER_USER.format(context=build_context(results), question=question)
    return client.generate_stream(prompt, temperature=temperature, system=prompts.ANSWER_SYSTEM)


def summarize(client: GeminiClient, chunks: list[Chunk], file_name: str, max_chars: int = 15000) -> dict:
    """Summary + suggested questions, from a sample of chunks spread across the document."""
    if not chunks:
        return {}
    budget = max(1, max_chars // 1000)
    step = max(1, len(chunks) // budget)
    sample = chunks[::step][:budget]
    text = "\n---\n".join(f"(page {c.page}) {c.text}" for c in sample)
    data = client.generate_json(prompts.SUMMARY.format(file_name=file_name, text=text))
    return {"summary": str(data.get("summary", "")).strip(),
            "questions": [str(q) for q in data.get("questions", [])][:3]}
