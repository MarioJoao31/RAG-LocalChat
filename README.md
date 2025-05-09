# 📚 RAG Document Query Assistant

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline with a user-friendly Streamlit interface. It allows you to ingest, chunk, embed, store, and query documents (PDF, DOCX, TXT) using local or cloud-based embedding models and a vector store.

---

## 🚀 Features

- 🔍 **Document Loader** (PDF, DOCX, TXT)  
- 🧠 **Embedding Models** (Hugging Face, Google Gemini, Local)  
- 🧹 **Semantic Chunking** with `SemanticChunker`  
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

---

## 🧠 Embedding Options

You can switch between:
 - ✅ Local Hugging Face model (sentence-transformers/all-MiniLM-L6-v2)
 - ☁️ Google Generative AI (embedding-001)
 - 🔗 Easily extensible via embedder.py

---

## 🔗 Google Drive Upload Setup

To enable file upload directly from Google Drive:

### 1. Go to [Google Cloud Console](https://console.cloud.google.com/)

### 2. Create a Project or Select an Existing One

### 3. Enable the Google Drive API
- Go to **APIs & Services → Library**
- Search and enable **Google Drive API**

### 4. Create OAuth 2.0 Credentials
- Go to **APIs & Services → Credentials**
- Click **Create Credentials → OAuth client ID**
- Select **Desktop App**
- Click **Create**

### 5. Download `client_secrets.json`
- After creation, click the download icon
- Rename the file to `client_secrets.json`
- Place it in the project root directory

### 6. Run the Google Drive Integration Feature
Once the file is in place, your app can authenticate with Google and access your Drive documents.
