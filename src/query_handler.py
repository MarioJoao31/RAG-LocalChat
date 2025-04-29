from langchain_community.vectorstores import Chroma

def similarity_query(vector_store, query: str, k: int = 3):
    """
    Perform a similarity search on the vector store.
    
    Args:
        vector_store: A loaded Chroma vector store.
        query (str): The user's question.
        k (int): Number of results to return.
    
    Returns:
        List of Document objects.
    """
    results = vector_store.similarity_search(query, k=k)
    return results
