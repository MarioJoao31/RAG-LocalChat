from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from src.loader import load_documents
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

 
