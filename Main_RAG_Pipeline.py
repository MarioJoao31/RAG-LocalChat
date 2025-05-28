
from src.loader import load_documents
from src.rag_pipeline import run_rag_pipeline

try:
    docs = load_documents("Data/Documents")
    print("Documents loaded successfully.")
except Exception as e:
    print(f"Error loading documents: {e}")
    docs = []

print(f"Loaded {len(docs)} documents.")
try:
    if not docs:
        raise ValueError("No documents found. Please check the directory.")
    
    #creating the vector store
    vector_store = run_rag_pipeline(docs=docs)
except ValueError as ve:
    print(ve)
    docs = []
except Exception as e:
    print(f"Error in RAG pipeline: {e}")
    docs = []

# Validate the vector store and print chunks
if hasattr(vector_store, 'chunks') and vector_store.chunks:
    print(f"Vector store contains {len(vector_store.chunks)} chunks.")
    vector_store.pretty_print_chunks()
else:
    print("No chunks found in vector store.")


