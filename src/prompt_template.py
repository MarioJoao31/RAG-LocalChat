rag_prompt_template = """
You are a Chat bot. Answer the question based on the context provided.
If the question is not answerable based on the context, say "I don't know".
This context is the string output of the vector store, so take this in consideration.
Context: {context}
Question: {question}
Answer:
"""