import sys
import os
import streamlit as st
from typing import List, Tuple

# -------------------------------------------------
# Path Fix (for local imports)
# -------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------
# Internal Imports
# -------------------------------------------------
from crawler.web_loder import WebsiteLoader
from crawler.text_cleaner import TextCleaner
from processing.chunker import TextChunker
from processing.embeddings import EmbeddingStore
from qa.qa_pipeline import QAPipeline

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Web Assistant | Himanshu Kabra",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Session State Initialization
# -------------------------------------------------
def init_session_state():
    defaults = {
        "chat_history": [],          # List[Tuple[str, str]]
        "website_indexed": False,
        "current_website": None,
        "is_indexing": False,
        "document_chunks": [],
        "qa_pipeline": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# -------------------------------------------------
# Sidebar - Settings Panel
# -------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    website_url = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        help="Enter the website URL you want to analyze"
    )

    index_clicked = st.button(
        "Index Website",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_indexing
    )

    st.divider()

    if st.session_state.is_indexing:
        st.info("Indexing website content...")
    elif st.session_state.website_indexed:
        st.success("Website indexed and ready")
        st.caption(f"Source: {st.session_state.current_website}")
        st.caption(f"Chunks: {len(st.session_state.document_chunks)}")
    else:
        st.warning("No website indexed")

# -------------------------------------------------
# Main Header
# -------------------------------------------------
st.markdown(
    """
    <h1 style="margin-bottom: 0.2rem;">AI-Powered Web Assistant | Himanshu Kabra</h1>
    <p style="color:#666; max-width: 700px;">
        Ask questions strictly grounded in the indexed website content.
        If information is not present, the assistant will explicitly say so.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# Indexing Logic (FULL PIPELINE)
# -------------------------------------------------
if index_clicked:
    if not website_url.strip():
        st.error("Please enter a valid website URL before indexing.")
    else:
        st.session_state.is_indexing = True
        st.session_state.website_indexed = False

        with st.spinner("Crawling, cleaning, embedding, and indexing website..."):
            try:
                # STEP 3: Load website
                loader = WebsiteLoader()
                page_title, raw_text = loader.fetch(website_url)

                # STEP 4: Clean text
                cleaner = TextCleaner()
                clean_text = cleaner.clean(raw_text)

                # STEP 5: Chunk text
                chunker = TextChunker(chunk_size=500, chunk_overlap=80)
                chunks = chunker.create_chunks(
                    text=clean_text,
                    source_url=website_url,
                    page_title=page_title
                )

                # STEP 6: Embeddings + FAISS
                store = EmbeddingStore()
                store.create_and_store(chunks)

                # STEP 7: Init QA pipeline
                st.session_state.qa_pipeline = QAPipeline()

                # Save state
                st.session_state.document_chunks = chunks
                st.session_state.current_website = website_url
                st.session_state.website_indexed = True
                st.session_state.chat_history = []

            except Exception as e:
                st.session_state.is_indexing = False
                st.error(f"Indexing failed: {str(e)}")
                st.stop()

        st.session_state.is_indexing = False
        st.success(f"Website indexed successfully ({len(chunks)} chunks).")
        st.rerun()

# -------------------------------------------------
# Chat Display Area
# -------------------------------------------------
chat_area = st.container()

with chat_area:
    if not st.session_state.chat_history:
        st.markdown(
            """
            <div style="color:#888; padding:1rem 0;">
            Index a website to begin asking questions.
            </div>
            """,
            unsafe_allow_html=True
        )

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

# -------------------------------------------------
# Chat Input + REAL QA
# -------------------------------------------------
query = st.chat_input(
    "Ask a question about the website...",
    disabled=not st.session_state.website_indexed
)

if query:
    st.session_state.chat_history.append(("user", query))

    with st.chat_message("assistant"):
        with st.spinner("Analyzing website content..."):
            try:
                qa = st.session_state.qa_pipeline
                answer = qa.answer(query)
            except Exception as e:
                answer = f"Error generating answer: {str(e)}"

            st.markdown(answer)
            st.session_state.chat_history.append(("assistant", answer))
