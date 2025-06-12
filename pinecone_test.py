from pinecone import Pinecone
from dotenv import load_dotenv
import os


# Initialize a Pinecone client with your API key

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))

# Create a dense index with integrated embedding
index_name = "n8ntest"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )