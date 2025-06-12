import openai
from dotenv import load_dotenv
import os
import traceback

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

def test_openai_api_key(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        return f"Error: {e}"

# Run the test
result = test_openai_api_key(api_key)
print(result)
