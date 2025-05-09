import streamlit as st
from langchain_community.vectorstores import Chroma
from src.query_handler import similarity_query
from src.embedder import get_huggingface_embedder
from src.rag_pipeline import add_single_file_to_vectorstore, run_rag_pipeline
from src.loader import save_uploaded_file, load_documents
from src.db_handler import insert_question_answer
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompt_template import rag_prompt_template
from src.gdrive_handler import download_all_from_folder
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
def get_tokenizer(model_name):
    """Get a tokenizer to make sure we're not sending too much text
    text to the Model. Eventually we will replace this with ArcticTokenizer
    """
    if not isinstance(model_name, str):
        raise ValueError("Model name must be a string.")

    return AutoTokenizer.from_pretrained(model_name)




vector_store = load_vector_store()

# Initialize state variables
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def clear_chat_history():
    #clear the chat history and reset the state variables
    st.session_state.chat_history = []
    st.session_state.chat_history = [{"role": "assistant", "content": "Ask me anything."}]



# UI UI UI UI UI UI for query input + "+" button

with st.sidebar:
    ###### Choose model and parameters
    st.title('💬 Models and parameters')
    st.write('Choose the model that you want to use.')
    
    model = st.selectbox("Select a model", ("gpt2", "openchat-3.5-0106","TinyLlama-1.1B-Chat-v1.0"), key="model")
    
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.7, step=0.01, help="Randomness of generated output")
    if temperature >= 1:
        st.warning('Values exceeding 1 produces more creative and random output as well as increased likelihood of hallucination.')
    if temperature < 0.1:
        st.warning('Values approaching 0 produces deterministic output. Recommended starting value is 0.7')
    
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01, help="Top p percentage of most likely tokens for output generation")



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

    folder_link = st.text_input("🔗 Enter Google Drive file ID to import:")
    if st.button("📥 Import from Drive") and folder_link:
        try:
            paths = download_all_from_folder(folder_link)

            if  paths.__len__() == 0:
                st.error("❌ No files found in the folder.")
                
            for path in paths:
                with open(path, "rb") as f:
                    add_single_file_to_vectorstore(f, vector_store)
                st.success(f"✅ File '{os.path.basename(path)}' added from Google Drive.")
        except Exception as e:
            print(f"❌ Failed to import file: {e}")
            st.error(f"❌ Failed to import file: {e}")



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
if "chat_history" not in st.session_state.keys():
    st.session_state.chat_history = [{"role": "assistant", "content": "Ask me anything."}]


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User-provided prompt
if query := st.chat_input():
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)


# Generate a new response if last message is not from assistant
if st.session_state.chat_history and (
    st.session_state.chat_history[-1]["role"] != "assistant" or
    st.session_state.chat_history[-1]["content"] != "Ask me anything."
):
    # results from the vector store 
    Vector_results = similarity_query(vector_store, query, 1)

    # Combine history (e.g., last 1–2 messages) into historical context
    chat_history_context = ""
    if len(st.session_state.chat_history) >= 2:
        for msg in st.session_state.chat_history[-4:-1]:  # include last 2-3 turns
            if msg["role"] == "user":
                chat_history_context += f"Previous Question: {msg['content']}\n"
                
            elif msg["role"] == "assistant":
                chat_history_context += f"Previous Answer: {msg['content']}\n"
                
    
    # Combine chat history context with vector store results
    context = chat_history_context + " " + " ".join([doc.page_content for doc in Vector_results])

   

    #create the prompt for the LLM
    formatted_prompt = rag_prompt_template.format(context=context, question=query)


    # Load the model based on user selection
    match model:
            case "gpt2":
                model = AutoModelForCausalLM.from_pretrained("Models/HuggingFace/gpt2")
                show_parameters = False
                print("Model loaded GPT2")
            case "openchat-3.5-0106":
                print("Model too big")
                #model = AutoModelForCausalLM.from_pretrained("Models/HuggingFace/openchat-3.5-0106")
            case "TinyLlama-1.1B-Chat-v1.0":
                show_parameters = True
                model = AutoModelForCausalLM.from_pretrained("Models/HuggingFace/TinyLlama-1.1B-Chat-v1.0")
                print("Model loaded TinyLlama")
            case _:
                st.error("Selected model is not supported.")


    
    #get tokenizer and model
    tokenizer = get_tokenizer(model_name=model.config._name_or_path)


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
        insert_question_answer(query, answer, model.config._name_or_path)
    except Exception as e:
        print(f"❌ Failed to save question and answer to the database: {e}")

    #save the answer to the chat cache 
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    #print the answer to the chat window
    with st.chat_message("assistant"):
        st.markdown(answer)



#TODO: add model type to the database table
