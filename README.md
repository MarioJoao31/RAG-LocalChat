# 📚 RAG Document Query Assistant

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline with a user-friendly Streamlit interface. It allows you to ingest, chunk, embed, store, and query documents (PDF, DOCX, TXT) using local or cloud-based embedding models and a vector store.

---

## 🚀 Features

- 🔍 **Document Loader** (PDF, DOCX, TXT)  
- 🧠 **Embedding Models** (Hugging Face, Google Gemini, Local)  
- 🧩 **Semantic Chunking** with `SemanticChunker`  
- 🗂️ **Vector Store** via ChromaDB  
- 💬 **Natural Language Query Interface** (Streamlit)  

---

## 🧑‍💻 How to Run

### 1. 🔧 Install Requirements

```bash
pip install -r requirements.txt
```

### .env

```bash
GOOGLE_API_KEY=your_google_genai_api_key  # if using Google embeddings
```

### 🗃️ Load and Process Docs

Make sure your input files are in the Docs/ folder.
Run your data prep pipeline (from rag_pipeline.py) once to initialize VectorDataBase/.

### 🖥️ Launch the App

```bash
streamlit run app.py
```

# 🧠 Embedding Options

You can switch between:
 - ✅ Local Hugging Face model (sentence-transformers/all-MiniLM-L6-v2)
 - ☁️ Google Generative AI (embedding-001)
 - 🔗 Easily extensible via embedder.py