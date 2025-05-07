
from src.db_handler import insert_question_answer
from transformers import AutoTokenizer, AutoModelForCausalLM


template = """
You are a helpful assistant. Answer the question based on the context provided.
If the question is not answerable based on the context, say "I don't know".
Context: {context}
Question: {question}
Answer:
"""

context = "My name is mario and i am a bike mechanic. I have been working on bikes for 10 years. I have a lot of experience with track bikes and road bikes. I also have experience with mountain bikes and BMX bikes. I can help you with any questions you have about bike maintenance, repairs, or upgrades."
question = "what type of questions can i help?"
formatted_prompt = template.format(context=context, question=question)



tokenizer = AutoTokenizer.from_pretrained("Models/HuggingFace/gpt2")
model = AutoModelForCausalLM.from_pretrained("Models/HuggingFace/gpt2")


# Tokenize and generate
inputs = tokenizer(formatted_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)

# Decode output
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract the first "Answer:"
if "Answer:" in decoded_output:
    answer = decoded_output.split("Answer:")[1].split("Question:")[0].strip()
else:
    answer = decoded_output.strip()


print(type(answer))

insert_question_answer(question, answer)

print("Full decoded output:\n", decoded_output)
print("Extracted answer:\n", answer)


