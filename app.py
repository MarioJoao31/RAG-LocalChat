import streamlit as st
from langchain_community.vectorstores import Chroma
from src.query_handler import similarity_query
from src.embedder import get_huggingface_embedder  # make sure this exists
from src.rag_pipeline import add_single_file_to_vectorstore
from src.loader import save_uploaded_file  # make sure this exists
import os

# Set up Streamlit
st.set_page_config(page_title="RAG Q&A", layout="centered")
st.title("📚 Document Query Assistant")

# Load vector store with caching (avoid reloading on every run)
@st.cache_resource
def load_vector_store():
    embedding_function = get_huggingface_embedder()
    return Chroma(
        collection_name="main_collection",
        embedding_function=embedding_function,
        persist_directory="Embeddings/Chroma"
    )

vector_store = load_vector_store()

# UI for query input
query = st.text_input("💬 Ask your question:")

if query:
    results = similarity_query(vector_store, query,1)

    if results:
        st.subheader("🔍 Top Results")
        for i, r in enumerate(results, start=1):
            st.markdown(f"**Result {i}** — *{r.metadata['source']}*")
            st.write(r.page_content[:500] + "...")
    else:
        st.warning("No results found.")

# File uploader
st.subheader("Add a New Document")
uploaded_file = st.file_uploader("Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])


if uploaded_file is not None:
    try:
        # Save uploaded file to disk
        save_uploaded_file(uploaded_file) 

        add_single_file_to_vectorstore(uploaded_file, vector_store)

        st.success(f"File '{uploaded_file.name}' added to the vector store.")

    except Exception as e:
        st.error(f"Failed to process file: {e}")
