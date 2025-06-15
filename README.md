# 📚 RAG Document Query Assistant

This project implements a modular **Retrieval-Augmented Generation (RAG)** pipeline with a user-friendly **Streamlit** interface. It allows you to ingest, chunk, embed, store, and query documents (PDF, DOCX, TXT) using local or cloud-based embedding models and a vector store. The system is built using **LangChain** and **LangSmith** for pipeline orchestration, observability, and experimentation.

---

## 🚀 Features

- 🔍 **Document Loader** (PDF, DOCX, TXT)  
- 🧠 **Embedding Models** (Hugging Face, Google Gemini, Local)  
- 🧹 **Semantic Chunking** with `SemanticChunker`  
- 🗂️ **Vector Store** via ChromaDB  
- 💬 **Natural Language Query Interface** (Streamlit)  
- 🔄 **Multiple Answer Generation Models** (switch dynamically in UI)  
- 🔗 **Google Drive Folder Linking** for document ingestion  
- 🧪 **LangChain + LangSmith Integration** for modular pipelines and tracing  
- 🧠 **Feedback (Like/Dislike)** system for user-based reward signals and future retraining

---

## 🧑‍💻 How to Run

### 1. 🔧 Install Requirements

```bash
pip install -r requirements.txt
```

### .env

```bash
GOOGLE_API_KEY=your_google_genai_api_key  # if using Google embeddings
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_TRACING_V2=true
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

## 🤖 Answer Generation Models

You can switch between:
 - 💡 Multiple model selection (LLM choices) directly from the UI
 - Easily pluggable using llm_router.py and LangChain agents

--- 
## ❤️ Reward Feedback Feature

Users can provide feedback (👍 Like / 👎 Dislike) on each response.
This feedback is stored for future use in fine-tuning or retraining the LLM components, enabling continuous improvement based on user preferences.

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
