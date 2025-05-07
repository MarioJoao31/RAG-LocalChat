import streamlit as st
from langchain_community.vectorstores import Chroma
from src.query_handler import similarity_query
from src.embedder import get_huggingface_embedder
from src.rag_pipeline import add_single_file_to_vectorstore, run_rag_pipeline
from src.loader import save_uploaded_file, load_documents
from src.db_handler import insert_question_answer
from transformers import AutoTokenizer, AutoModelForCausalLM



import os

# Set up Streamlit
st.set_page_config(page_title="RAG Q&A", layout="centered")
st.title("📚 Document Query Assistant")

# Load vector store with caching
@st.cache_resource
def load_vector_store():
    embedding_function = get_huggingface_embedder()
    return Chroma(
        collection_name="main_collection",
        embedding_function=embedding_function,
        persist_directory="Embeddings/Chroma"
    )

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text
    text to the Model. Eventually we will replace this with ArcticTokenizer
    """
    return AutoTokenizer.from_pretrained("Models/HuggingFace/gpt2")

def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)



vector_store = load_vector_store()

# Initialize state variables
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def clear_chat_history():
    st.session_state.chat_history = [{"role": "assistant", "content": "Ask me anything."}]

template = """
You are a helpful assistant. Answer the question based on the context provided.
If the question is not answerable based on the context, say "I don't know".
This context is the string output o vectore Store, so take this in consideration.
Context: {context}
Question: {question}
Answer:
"""





# UI UI UI UI UI UI for query input + "+" button

with st.sidebar:
    st.title('💬 Chatbot')
    st.write('Create chatbots using various LLM models.')
    
    st.subheader("Models and parameters")
    model = st.selectbox("Select a model",("Models/HuggingFace/gpt2", "Models/HuggingFace/roberta-base-squad2"), key="model")
    if model == "Models/HuggingFace/gpt2":
        model = AutoModelForCausalLM.from_pretrained("Models/HuggingFace/gpt2")
    
    ################### Load documents from folder
    st.subheader("📂 Load Documents from Local Folder Path")
    st.caption("Enter the absolute path to a local folder containing `.txt`, `.pdf`, or `.docx` files. "
            "All valid documents inside the folder and its subfolders will be processed and added to the vector database.")

    folder_path = st.text_input(
        "📁 Enter local folder path to load documents recursively:",
        placeholder="e.g., C:/Users/YourName/Documents/ProjectDocs"
    )

    if folder_path and st.button("Load Folder to Vector Store"):
        if os.path.isdir(folder_path):
            docs = load_documents(folder_path)
            run_rag_pipeline(docs)
            st.success(f"✅ Added documents from the folder path: {folder_path}.")
        else:
            st.error("❌ The specified path does not exist.")
    
    

    if st.button("➕ Upload single files"):
            st.session_state.show_uploader = True
    
    st.button("🧹 Clear chat", on_click=clear_chat_history)






#col1, col2, col3 = st.columns([5, 1, 1])
#with col1:
#    query = st.chat_input("💬 Ask your question:")
#
#with col2:
#    if not st.session_state.show_uploader:
#        if st.button("➕", help="Add a document"):
#            st.session_state.show_uploader = True
#
#with col3:
#    st.button('Clear', on_click=clear_chat_history)
    


# File uploader (only shows when "+" is clicked)
if st.session_state.show_uploader:
    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["txt", "pdf", "docx"],
        label_visibility="collapsed",
        key="file_uploader",
    )

    if uploaded_file is not None:
        try:
            save_uploaded_file(uploaded_file) 
            add_single_file_to_vectorstore(uploaded_file, vector_store)
            st.success(f"✅ File '{uploaded_file.name}' added to the vector store.")
            # Reset uploader state after successful upload
            st.session_state.show_uploader = False
            st.session_state.uploaded_file = uploaded_file
        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")

# Store LLM-generated responses
if "messages" not in st.session_state.keys():
    st.session_state.chat_history = [{"role": "assistant", "content": "Ask me anything."}]



#with col1:
#    query = st.chat_input("💬 Ask your question:")
#
#with col2:
#    if not st.session_state.show_uploader:
#        if st.button("➕", help="Add a document"):
#            st.session_state.show_uploader = True
#
#with col3:
#    st.button('Clear', on_click=clear_chat_history)


# User-provided prompt
if query := st.chat_input():
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)




# Generate a new response if last message is not from assistant
if st.session_state.chat_history[-1]["role"] != "assistant":
    
    # results from the vector store 
    Vector_results = similarity_query(vector_store, query, 1)


    # Combine the content of the top results into a single string
    context = " ".join([doc.page_content for doc in Vector_results])

    #create the prompt for the LLM
    formatted_prompt = template.format(context=context, question=query)

    #get tokenizer and model
    tokenizer = get_tokenizer()
    # Tokenize and generate
    # Ensure the formatted prompt is not empty
    if not formatted_prompt.strip():
        st.error("❌ The prompt is empty. Please provide a valid query.")
    else:
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
        if inputs["input_ids"].size(1) == 0:
            st.error("❌ Tokenization failed. Please check the input format.")
        else:
            outputs = model.generate(**inputs, max_new_tokens=100)

    # Decode output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the first "Answer:" / gpt2 doesnt have stop sequence, so we need to extract the answer manually
    if "Answer:" in decoded_output:
        answer = decoded_output.split("Answer:")[1].split("Question:")[0].strip()
    else:
        answer = decoded_output.strip()

    #save the question and answer to the database
    try:
        insert_question_answer(query, answer)
    except Exception as e:
        print(f"❌ Failed to save question and answer to the database: {e}")

    #save the answer to the chat cache 
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    #print the answer to the chat window
    with st.chat_message("ChatBoty"):
        st.markdown(answer)


#TODO: put the template prompt in another file and load it here.
#TODO: 
