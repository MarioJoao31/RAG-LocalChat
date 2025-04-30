from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from src.loader import read_file
from src.embedder import get_sentence_transformer_model, LocalEmbeddingFunction
from src.retriever import create_vector_store, add_documents_to_store, similarity_query

def run_rag_pipeline():
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
    # Read file content
    text = read_file(file)
    metadata = {"source": file.name}

    # Get embedding function and splitter
    model = get_sentence_transformer_model()
    embedding_function = LocalEmbeddingFunction(model)
    text_splitter = SemanticChunker(
        embedding_function, 
        breakpoint_threshold_type="percentile"
    )

    # Convert and split into chunks
    chunks = text_splitter.split_documents([Document(page_content=text, metadata=metadata)])

    # If not passed, get default vector store
    if vector_store is None:
        vector_store = create_vector_store(embedding_function)

    # Add new chunks
    vector_store.add_documents(chunks)
