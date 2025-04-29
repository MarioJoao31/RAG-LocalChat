import streamlit as st
from langchain_community.vectorstores import Chroma
from src.query_handler import similarity_query
from src.embedder import get_embedding_function  # make sure this exists
import os

# Set up Streamlit
st.set_page_config(page_title="RAG Q&A", layout="centered")
st.title("📚 Document Query Assistant")

# Load vector store with caching (avoid reloading on every run)
@st.cache_resource
def load_vector_store():
    embedding_function = get_embedding_function()
    return Chroma(
        collection_name="main_collection",
        embedding_function=embedding_function,
        persist_directory="Embeddings/Chroma"
    )

vector_store = load_vector_store()

# UI for query input
query = st.text_input("💬 Ask your question:")

if query:
    results = similarity_query(vector_store, query)

    if results:
        st.subheader("🔍 Top Results")
        for i, r in enumerate(results, start=1):
            st.markdown(f"**Result {i}** — *{r.metadata['source']}*")
            st.write(r.page_content[:500] + "...")
    else:
        st.warning("No results found.")
