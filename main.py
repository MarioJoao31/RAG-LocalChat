from src.rag_pipeline import run_rag_pipeline
from src.query_handler import similarity_query

if __name__ == "__main__":
    vector_store = run_rag_pipeline()
    res = similarity_query(vector_store,"Maintenace on my Bike?", k=1)
    print(res)
