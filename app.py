"""
RAG Technical Documentation Chatbot
Entry point for the Streamlit application.
"""
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from src.config import get_settings
from src.embeddings import EmbeddingModel
from src.generation import RAGChain
from src.ingestion import IngestionPipeline, TextChunker
from src.retrieval import DocumentRetriever
from src.vectorstore import ChromaVectorStore


# ── Bootstrap (runs once per session) ────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def build_vector_store(persist_dir: str, collection: str, model_name: str, device: str):
    embedding_model = EmbeddingModel(model_name=model_name, device=device)
    return ChromaVectorStore(
        embeddings=embedding_model.embeddings,
        persist_dir=Path(persist_dir),
        collection_name=collection,
    )


@st.cache_resource(show_spinner=False)
def build_chain(_vector_store, groq_api_key: str, llm_model: str, temperature: float, max_tokens: int, k: int, threshold: float):
    retriever = DocumentRetriever(_vector_store, k=k, score_threshold=threshold)
    return RAGChain(
        retriever=retriever,
        groq_api_key=groq_api_key,
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role": str, "content": str, "sources": list}
    if "ingested_docs" not in st.session_state:
        st.session_state.ingested_docs = []


# ── UI Helpers ────────────────────────────────────────────────────────────────

def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)} chunks)", expanded=False):
        for i, chunk in enumerate(sources, 1):
            page_info = f" — page {chunk.page}" if chunk.page else ""
            score_pct = int(chunk.score * 100)
            st.markdown(f"**[{i}] {chunk.source}{page_info}** · relevance {score_pct}%")
            st.caption(chunk.content[:300] + ("…" if len(chunk.content) > 300 else ""))
            if i < len(sources):
                st.divider()


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])


def to_lc_history(messages: list) -> list:
    lc_msgs = []
    for msg in messages[-10:]:  # keep last 10 turns
        if msg["role"] == "user":
            lc_msgs.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_msgs.append(AIMessage(content=msg["content"]))
    return lc_msgs


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(settings, vector_store) -> tuple[IngestionPipeline, bool]:
    with st.sidebar:
        st.title("📂 Documents")

        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        pipeline = IngestionPipeline(chunker=chunker, vector_store=vector_store)

        process_clicked = False
        if uploaded_files:
            process_clicked = st.button("⚡ Process Documents", use_container_width=True, type="primary")

        if process_clicked and uploaded_files:
            with st.status("Processing documents…", expanded=True) as status:
                for uf in uploaded_files:
                    suffix = Path(uf.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        tmp_path = Path(tmp.name)

                    st.write(f"Processing `{uf.name}`…")
                    n_chunks = pipeline.ingest(tmp_path)
                    tmp_path.unlink(missing_ok=True)

                    if n_chunks > 0:
                        st.session_state.ingested_docs.append(uf.name)
                        st.write(f"✅ `{uf.name}` → {n_chunks} chunks")
                    else:
                        st.write(f"⏭️ `{uf.name}` already indexed")

                status.update(label="Done!", state="complete")
            st.rerun()

        # Loaded documents list
        live_sources = vector_store.list_sources()
        if live_sources:
            st.markdown("**Indexed documents**")
            for src in live_sources:
                col1, col2 = st.columns([4, 1])
                col1.caption(f"📄 {src}")
                if col2.button("🗑", key=f"del_{src}", help=f"Remove {src}"):
                    vector_store.delete_source(src)
                    st.rerun()
        else:
            st.info("No documents loaded yet.")

        st.divider()

        # Settings
        with st.expander("⚙️ Settings"):
            st.slider("Chunks to retrieve (k)", 1, 10, settings.retrieval_k, key="retrieval_k")
            st.selectbox(
                "LLM Model",
                ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
                key="llm_model",
            )

        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return pipeline, process_clicked


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        settings = get_settings()
    except Exception:
        st.error("❌ GROQ_API_KEY not set. Create a `.env` file from `.env.example`.")
        st.stop()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="📚",
        layout="wide",
    )
    st.title("📚 " + settings.app_title)

    init_session_state()

    vector_store = build_vector_store(
        persist_dir=str(settings.chroma_persist_dir),
        collection=settings.chroma_collection_name,
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )

    render_sidebar(settings, vector_store)

    # Build chain — respects live settings from sidebar sliders
    k = st.session_state.get("retrieval_k", settings.retrieval_k)
    llm_model = st.session_state.get("llm_model", settings.llm_model)
    chain = build_chain(
        _vector_store=vector_store,
        groq_api_key=settings.groq_api_key,
        llm_model=llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        k=k,
        threshold=settings.retrieval_score_threshold,
    )

    if not vector_store.list_sources():
        st.info("👈 Upload and process documents in the sidebar to start chatting.")

    render_chat_history()

    if question := st.chat_input("Ask a question about your documents…"):
        st.session_state.messages.append({"role": "user", "content": question, "sources": []})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""
            last_sources = []

            with st.spinner("Thinking…"):
                for token, sources in chain.stream(
                    question=question,
                    chat_history=to_lc_history(st.session_state.messages[:-1]),
                ):
                    full_answer += token
                    placeholder.markdown(full_answer + "▌")
                    last_sources = sources

            placeholder.markdown(full_answer)
            render_sources(last_sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_answer, "sources": last_sources}
        )


if __name__ == "__main__":
    main()
