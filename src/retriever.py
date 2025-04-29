from langchain_chroma import Chroma

def create_vector_store(embedding_function, collection_name="main_collection"):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory="Embeddings/Chroma",
    )

def add_documents_to_store(vector_store, chunks):
    vector_store.add_documents(documents=chunks)
    print(f"Added {len(chunks)} documents to the vector store.")

def similarity_query(vector_store, query, k=3):
    results = vector_store.similarity_search(query, k=k)
    return results
