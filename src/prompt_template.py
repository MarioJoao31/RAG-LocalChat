rag_prompt_template = """
You are a helpful assistant. Use the provided context and the recent conversation history to answer the user's question. If the answer is not contained within the context, respond with "I don't know the answer, please add more file to my Vector Storage."

Context:
{retrieved_context}

Conversation History:
{formatted_history}

Sources:
{source_list}

Current Question:
{current_question}

Answer (please cite the source file names when applicable):
"""

