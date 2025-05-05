from src.rag_pipeline import run_rag_pipeline
from src.query_handler import similarity_query
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


from transformers import AutoTokenizer, AutoModelForCausalLM

template = """
You are a helpful assistant. Answer the question based on the context provided.
If the question is not answerable based on the context, say "I don't know".
Context: {context}
Question: {question}
Answer:
"""

context = "My name is mario and i am a bike mechanic. I have been working on bikes for 10 years. I have a lot of experience with track bikes and road bikes. I also have experience with mountain bikes and BMX bikes. I can help you with any questions you have about bike maintenance, repairs, or upgrades."
question = "what is my name and what do i do?"
formatted_prompt = template.format(context=context, question=question)

tokenizer = AutoTokenizer.from_pretrained("Models/HuggingFace/gpt2")
model = AutoModelForCausalLM.from_pretrained("Models/HuggingFace/gpt2")

# Tokenize and generate
inputs = tokenizer(formatted_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)

# Decode output
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Answer:\n", answer)

