from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer

class LocalEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0]

def get_huggingface_embedder(model_path="Models/HuggingFace/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_path, model_kwargs={"local_files_only": True})

def get_google_embedder(model="Models/embedding-001"):
    return GoogleGenerativeAIEmbeddings(model=model)

def get_sentence_transformer_model(model_path="Models/HuggingFace/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_path)
