from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
import os

from src.loader import read_file, load_documents
from src.embedder import get_sentence_transformer_model, LocalEmbeddingFunction
from src.retriever import create_vector_store, add_documents_to_store, similarity_query

def run_rag_pipeline(docs=None):
    """
    Add the multiple documents to the vector store.
    If no documents are passed, it will load the documents from the default directory.

    """
    if docs is None:
        docs = load_documents()
    texts = [doc['text'] for doc in docs]
    metadatas = [{"source": doc['path']} for doc in docs]

    model = get_sentence_transformer_model()
    embedding_function = LocalEmbeddingFunction(model)

    text_splitter = SemanticChunker(
        embedding_function,
        breakpoint_threshold_type="percentile"
    )
    
    chunks = text_splitter.split_documents([
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(texts, metadatas)
    ])
    
    vector_store = create_vector_store(embedding_function)
    add_documents_to_store(vector_store, chunks)


    print(f"Vector store created with {len(chunks)} chunks.")

    return vector_store

def add_single_file_to_vectorstore(file, vector_store=None):

    # Validate file existence and name
    if file is None or not hasattr(file, "name") or not file.name:
        raise ValueError("Invalid file: no file or filename provided.")

    # Validate supported file type
    ext = os.path.splitext(file.name)[-1].lower()
    if ext not in [".txt", ".pdf", ".docx"]:
        raise ValueError(f"Unsupported file format: {ext}")

    # Read file content
    try:
        text = read_file(file)
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

    if not text.strip():
        raise ValueError("File is empty or unreadable.")

    metadata = {"source": file.name}
    

    # Get embedding function and splitter
    model = get_sentence_transformer_model()
    embedding_function = LocalEmbeddingFunction(model)
    text_splitter = SemanticChunker(
        embedding_function, 
        breakpoint_threshold_type="percentile"
    )

    # Convert and split into chunks
    chunks = text_splitter.split_documents(
        [Document(page_content=text, metadata=metadata)]
    )

    if not chunks:
        raise ValueError("No chunks were created from the document.")

    # If not passed, get default vector store
    if vector_store is None:
        vector_store = create_vector_store(embedding_function)

    # Add new chunks
    vector_store.add_documents(chunks)
    print(f"Successfully added {len(chunks)} chunks from {file.name} to the vector store.")

