import streamlit as st
from langchain_community.vectorstores import Chroma
from src.query_handler import similarity_query
from src.embedder import get_huggingface_embedder  # make sure this exists
from src.rag_pipeline import add_single_file_to_vectorstore, run_rag_pipeline
from src.loader import save_uploaded_file, load_documents  # make sure this exists


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
    results = similarity_query(vector_store, query, 3)

    if results:
        st.subheader("🔍 Top Results")
        for i, r in enumerate(results, start=1):
            st.markdown(f"**Result {i}** — Path:*{r.metadata['source']}*")
            st.markdown(r.page_content.strip()[:2000] + "..." if len(r.page_content) > 2000 else r.page_content.strip())

    else:
        st.warning("No results found.")

# File uploader
st.subheader("📄 Upload a Document")
st.caption("Supported formats: `.txt`, `.pdf`, `.docx`. The uploaded file will be embedded using a language model and added to the vector database for future semantic search.")

uploaded_file = st.file_uploader("Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])


if uploaded_file is not None:
    try:
        # Save uploaded file to disk
        save_uploaded_file(uploaded_file) 

        add_single_file_to_vectorstore(uploaded_file, vector_store)

        st.success(f"File '{uploaded_file.name}' added to the vector store.")

    except Exception as e:
        st.error(f"Failed to process file: {e}")

#TODO: use the LLM to analyze the top results and provide a summary or answer to the question.
#TODO: add an description to the title upload file, so i can explain what type of files are accepted. and what it does.

st.subheader("📂 Load Documents from Local Folder Path")
st.caption("Enter the absolute path to a local folder containing `.txt`, `.pdf`, or `.docx` files. "
           "All valid documents inside the folder and its subfolders will be processed and added to the vector database.")


folder_path = st.text_input(
    "📁 Enter local folder path to load documents recursively:",
    placeholder="e.g., C:/Users/YourName/Documents/ProjectDocs"
    )

if folder_path and st.button("Load Folder to Vector Store"):
    if os.path.isdir(folder_path):
        docs = load_documents(folder_path)  # your existing recursive loader
        
        run_rag_pipeline(docs)

        st.success(f"✅ Added documents from the folder path: {folder_path}.")
    else:
        st.error("❌ The specified path does not exist.")

        #TODO: add a database to save the vat conversation 
        #TODO create a plus sign to add the documents 