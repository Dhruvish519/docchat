"""DocChat - chat with your PDFs using Retrieval-Augmented Generation (RAG) and Google Gemini.

Run with:  streamlit run app.py
"""

import os
from datetime import datetime

import numpy as np
import pymupdf
import streamlit as st
from google.genai import errors

from rag import pipeline
from rag.llm import GeminiClient
from rag.vector_store import VectorStore

MODELS = ["gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash"]

st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")


# ---------------------------------------------------------------- helpers
def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default) or os.environ.get(name, default)
    except Exception:          # no secrets.toml file
        return os.environ.get(name, default)


@st.cache_resource(show_spinner=False)
def get_client(api_key: str, model: str) -> GeminiClient:
    return GeminiClient(api_key, model)


def friendly_error(e: Exception) -> str:
    msg = str(e)
    code = getattr(e, "code", None)
    if "API_KEY_INVALID" in msg or "API key not valid" in msg:
        return "Your Gemini API key is not valid. Check it in the sidebar."
    if code == 404:
        return "This model isn't available for your key. Pick a different model in the sidebar."
    if code == 429:
        return "Free-tier rate limit reached. Wait a minute and try again, or pick a lighter model."
    return f"Gemini API error: {msg[:300]}"


@st.cache_data(show_spinner=False, max_entries=64)
def render_page(pdf_bytes: bytes, page: int, zoom: float = 1.6) -> bytes:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pix = doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return pix.tobytes("png")


def chat_as_markdown(messages) -> str:
    lines = [f"# DocChat conversation — {datetime.now():%d %b %Y %H:%M}\n"]
    for m in messages:
        who = "You" if m["role"] == "user" else "DocChat"
        lines.append(f"**{who}:** {m['content']}\n")
        for i, s in enumerate(m.get("sources", []), start=1):
            lines.append(f"> [{i}] {s['file_name']}, page {s['page']}")
        lines.append("")
    return "\n".join(lines)


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['file_name']} — page {s['page']}** "
                        f"<span style='color:gray'>· similarity {s['score']:.2f}</span>",
                        unsafe_allow_html=True)
            st.caption(s["text"][:500] + ("…" if len(s["text"]) > 500 else ""))


# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("docs", {})          # file_name -> {bytes, chunks, vectors, pages, summary}
ss.setdefault("messages", [])
ss.setdefault("pending_question", None)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📄 DocChat")
    st.caption("Chat with your PDFs · RAG + Gemini")

    server_key = secret("GEMINI_API_KEY")
    if server_key:
        # Never put the server's key into a widget: visitors could reveal it.
        st.success("🔑 Using the app's built-in API key", icon="✅")
        own_key = st.text_input("Use your own key instead (optional)", type="password")
        api_key = own_key or server_key
    else:
        api_key = st.text_input("Gemini API key", type="password",
                                help="Get a free key at https://aistudio.google.com/apikey")
    default_model = secret("LLM_MODEL", MODELS[0])
    model_options = MODELS if default_model in MODELS else [default_model] + MODELS
    model = st.selectbox("LLM model", model_options, index=model_options.index(default_model))

    with st.expander("⚙️ Retrieval settings"):
        top_k = st.slider("Passages to retrieve (top-k)", 2, 10, 5)
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.1)
        chunk_size = st.select_slider("Chunk size (characters)", [500, 750, 1000, 1500, 2000], 1000)
        use_memory = st.toggle("Understand follow-up questions", value=True,
                               help="Rewrites follow-ups like 'what about page 3?' into a full question using chat history.")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("🧹 Clear chat", width="stretch"):
        ss.messages = []
        st.rerun()
    c2.download_button("💾 Export", chat_as_markdown(ss.messages), "docchat_conversation.md",
                       width="stretch", disabled=not ss.messages)

if not api_key:
    st.title("📄 DocChat")
    st.info("👈 Enter your **Gemini API key** in the sidebar to start. "
            "You can get one for free at [Google AI Studio](https://aistudio.google.com/apikey).")
    st.stop()

client = get_client(api_key, model)

# ---------------------------------------------------------------- upload + indexing
st.title("📄 DocChat")
uploads = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

current_names = {f.name for f in uploads or []}
for name in list(ss.docs):                       # forget files the user removed
    if name not in current_names:
        del ss.docs[name]

for f in uploads or []:
    doc = ss.docs.get(f.name)
    if doc and doc["size"] == f.size and doc["chunk_size"] == chunk_size:
        continue
    data = f.getvalue()
    with st.status(f"Reading **{f.name}**…", expanded=False) as status:
        try:
            chunks, vectors, pages, cached = pipeline.index_pdf(
                client, data, f.name, chunk_size, overlap=chunk_size // 5)
        except errors.APIError as e:
            status.update(label=f"Could not index {f.name}", state="error")
            st.error(friendly_error(e))
            st.stop()
        if not chunks:
            status.update(label=f"{f.name}: no text found", state="error")
            st.warning(f"**{f.name}** has no selectable text (it may be a scanned image). Skipping it.")
            continue
        summary = (doc or {}).get("summary") if doc and doc["size"] == f.size else None
        if summary is None:
            status.update(label=f"Summarising **{f.name}**…")
            try:
                summary = pipeline.summarize(client, chunks, f.name)
            except Exception:
                summary = {}
        ss.docs[f.name] = {"bytes": data, "size": f.size, "chunk_size": chunk_size,
                           "chunks": chunks, "vectors": vectors, "pages": pages, "summary": summary}
        status.update(label=f"**{f.name}** ready · {pages} pages · {len(chunks)} chunks"
                            + (" (from cache)" if cached else ""), state="complete")

if not ss.docs:
    st.markdown("""
#### How it works
1. **Upload** PDFs — each page is split into overlapping text *chunks*.
2. Each chunk is turned into an **embedding** (a vector of numbers that captures meaning) with Gemini.
3. Embeddings go into a **FAISS** vector index.
4. When you ask a question, the most similar chunks are **retrieved** and given to the **Gemini LLM**,
   which writes an answer **grounded in your documents**, with page citations.
""")
    st.stop()

# ---------------------------------------------------------------- vector store
selected = st.multiselect("Search in", list(ss.docs), default=list(ss.docs))
if not selected:
    st.warning("Select at least one document.")
    st.stop()

store_key = tuple((n, ss.docs[n]["size"], ss.docs[n]["chunk_size"]) for n in sorted(selected))
if ss.get("store_key") != store_key:
    all_chunks = [c for n in sorted(selected) for c in ss.docs[n]["chunks"]]
    all_vectors = np.vstack([ss.docs[n]["vectors"] for n in sorted(selected)])
    ss.store = VectorStore(all_chunks, all_vectors)
    ss.store_key = store_key
store: VectorStore = ss.store

total_pages = sum(ss.docs[n]["pages"] for n in selected)
m1, m2, m3 = st.columns(3)
m1.metric("Documents", len(selected))
m2.metric("Pages", total_pages)
m3.metric("Chunks indexed", len(store.chunks))

tab_chat, tab_docs = st.tabs(["💬 Chat", "📑 Documents"])

# ---------------------------------------------------------------- documents tab
with tab_docs:
    name = st.selectbox("Document", selected)
    d = ss.docs[name]
    summ = d.get("summary") or {}
    if summ.get("summary"):
        st.subheader("Summary")
        st.write(summ["summary"])
    left, right = st.columns([1, 2])
    page = left.number_input("Page", 1, d["pages"], 1)
    left.caption(f"{d['pages']} pages · {len(d['chunks'])} chunks")
    right.image(render_page(d["bytes"], int(page)), width="stretch")

# ---------------------------------------------------------------- chat tab
with tab_chat:
    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                show_sources(m.get("sources"))

    if not ss.messages:
        suggestions = [q for n in selected for q in (ss.docs[n].get("summary") or {}).get("questions", [])][:4]
        if suggestions:
            st.caption("Try asking:")
            cols = st.columns(len(suggestions))
            for col, q in zip(cols, suggestions):
                if col.button(q, width="stretch"):
                    ss.pending_question = q
                    st.rerun()

question = st.chat_input("Ask a question about your documents…")
if ss.pending_question:
    question, ss.pending_question = ss.pending_question, None

if question:
    with tab_chat:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                search_q = question
                if use_memory and ss.messages:
                    with st.spinner("Understanding your follow-up…"):
                        search_q = pipeline.condense_question(client, question, ss.messages)
                    if search_q.lower() != question.lower():
                        st.caption(f"🔎 Searching for: *{search_q}*")
                with st.spinner("Searching your documents…"):
                    results = pipeline.retrieve(client, store, search_q, top_k)
                answer = st.write_stream(pipeline.answer_stream(client, search_q, results, temperature))
            except errors.APIError as e:
                st.error(friendly_error(e))
                st.stop()
            sources = [{"file_name": c.file_name, "page": c.page, "text": c.text, "score": s}
                       for c, s in results]
            show_sources(sources)
    ss.messages.append({"role": "user", "content": question})
    ss.messages.append({"role": "assistant", "content": answer, "sources": sources})
