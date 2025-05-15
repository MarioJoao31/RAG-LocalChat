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
from streamlit_chat import message as chat_message  # Rename to avoid conflict

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


def generate_answer(model, formatted_prompt, temperature, top_p, top_k, repetition_penalty, no_repeat_ngram_size, max_new_tokens):
    """Generate an answer using the model and the provided prompt.
    Args:
        model: The language model to use for generation.
        formatted_prompt: The formatted prompt to provide to the model.
        temperature: Temperature parameter for generation.
        top_p: Top-p sampling parameter for generation.
        top_k: Top-k sampling parameter for generation.
        repetition_penalty: Repetition penalty parameter for generation.
        no_repeat_ngram_size: Size of n-grams to avoid repeating in the output.
        max_new_tokens: Maximum number of new tokens to generate.
    Returns:
        decoded_output: The generated answer from the model.
    """
    maximum_model_Tokens = 1024

    #get tokenizer and model
    tokenizer = get_tokenizer(model_name=model.config._name_or_path)

    token_count = log_prompt_token_info(formatted_prompt, tokenizer, maximum_model_Tokens)

    print(f"token_count: {token_count}")

    
    # Ensure the formatted prompt is not empty
    if not formatted_prompt.strip():
        st.error("❌ The prompt is empty. Please provide a valid query.")
    else:
        #TODO: need to review this part of max new tokens and max model tokens 
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True,max_length=maximum_model_Tokens)
        
        #validates if the input is empty 
        if inputs["input_ids"].size(1) == 0:
            st.error("❌ Tokenization failed. Please check the input format.")
        else:
            # Generate output
            outputs = model.generate(**inputs, 
                    temperature=temperature, 
                    top_p=top_p, 
                    do_sample=True, 
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    max_new_tokens=max_new_tokens)

    # Decode output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded_output

def log_prompt_token_info(prompt: str, tokenizer, max_model_tokens: int = 512):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    token_count = len(tokens)
    
    print(f"\n🔍 Prompt Token Info:")
    print(f"- Token count: {token_count} / {max_model_tokens}")
    
    if token_count > max_model_tokens:
        print("⚠️ WARNING: Prompt exceeds max token limit. The model input will be truncated.")
    elif token_count > 0.9 * max_model_tokens:
        print("⚠️ NOTICE: Prompt is nearing the token limit.")
    else:
        print("✅ Prompt is within acceptable range.")
    
    return token_count

vector_store = load_vector_store()

# Initialize state variables
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "past" not in st.session_state:
    st.session_state.past = []
if "generated" not in st.session_state:
    st.session_state.generated = []

def clear_chat_history():
    #clear the chat history and reset the state variables
    st.session_state.past = []
    st.session_state.generated = []


def main():

    with st.sidebar:
        ###### Choose model and parameters
        st.title('💬 Models and parameters')
        st.write('Choose the model that you want to use.')
        
        model = st.selectbox("Select a model", ("TinyLlama-1.1B-Chat-v1.0","gpt2"), key="model")
        
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.00, value=0.7, step=0.01, help="Randomness of generated output")
        if temperature >= 1:
            st.warning('Values exceeding 1 produces more creative and random output as well as increased likelihood of hallucination.')
        if temperature < 0.1:
            st.warning('Values approaching 0 produces deterministic output. Recommended starting value is 0.7')
        
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01, help="Top p percentage of most likely tokens for output generation")

        top_k = st.sidebar.slider(
        'top_k',
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        help="Top k most likely tokens to sample from"
        )

        repetition_penalty = st.sidebar.slider(
            'repetition_penalty',
            min_value=0.8,
            max_value=2.0,
            value=1.1,
            step=0.1,
            help="Penalty for repeated tokens (1.0 = no penalty)"
        )

        no_repeat_ngram_size = st.sidebar.slider(
            'no_repeat_ngram_size',
            min_value=0,
            max_value=10,
            value=3,
            step=1,
            help="Prevent repetition of n-grams of this size"
        )

        max_new_tokens = st.sidebar.slider(
            'max_new_tokens',
            min_value=1,
            max_value=512,
            value=256,
            step=1,
            help="Max tokens to generate beyond the input"
        )


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
        
        ############### Upload single files
        if st.button("➕ Upload single files"):
                st.session_state.show_uploader = True
        
        st.button("🧹 Clear chat", on_click=clear_chat_history)


        ################ Load documents from Google Drive
        st.subheader("🔗 Load Documents from Google Drive")
        folder_link = st.text_input(" Enter Google Drive folder link to import:")
        if st.button("📥 Import from Drive") and folder_link:
            with st.spinner("Importing files from google drive..."):
                try:
                    paths = download_all_from_folder(folder_link)

                    if  paths.__len__() == 0:
                        st.error("❌ No files found in the folder.")

                    for path in paths:
                        with open(path, "rb") as f:
                            add_single_file_to_vectorstore(f, vector_store)
                        st.success(f"✅ Files '{os.path.basename(path)}' added from Google Drive.")
                    st.success(f"✅ {paths.__len__()} files added from Google Drive.")
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


    # Render chat messages using streamlit-chat
    for i in range(len(st.session_state.past)):
        chat_message(st.session_state.past[i], is_user=True, key=f"{i}_user")
        chat_message(st.session_state.generated[i], key=f"{i}", allow_html=True)


    

    query = st.chat_input("Ask your question:")
    # User-provided prompt
    if query:
        #append the question to the chat history
        st.session_state.past.append(query)
        #write the message of the user
        chat_message(query, is_user=True, key="user")

        # results from the vector store 
        Vector_results = similarity_query(vector_store, query, 1)

        # Combine history (e.g., last 1–2 messages) into historical context
        #chat_history_context = ""
        #if len(st.session_state.chat_history) >= 2:
        #    for msg in st.session_state.chat_history[-4:-1]:  # include last 2-3 turns
        #        if msg["role"] == "user":
        #            chat_history_context += f"Previous Question: {msg['content']}\n"
        #            
        #        elif msg["role"] == "assistant":
        #            chat_history_context += f"Previous Answer: {msg['content']}\n"
                  
        
        # context is only the result of the vector store 
        context =  " ".join([doc.page_content for doc in Vector_results])

        
        # Extract source file paths
        source_paths = [doc.metadata.get("source", "unknown file") for doc in Vector_results]
        formatted_sources = "\n".join([f"- {path}" for path in source_paths])


        #create the prompt for the LLM
        formatted_prompt = rag_prompt_template.format(retrieved_context=context,
                                                    source_list=formatted_sources,
                                                    current_question=query
                                                    )


        # Load the model based on user selection
        match model:
                case "gpt2":
                    model = AutoModelForCausalLM.from_pretrained("Models/HuggingFace/gpt2")               
                case "TinyLlama-1.1B-Chat-v1.0":
                    model = AutoModelForCausalLM.from_pretrained("Models/HuggingFace/TinyLlama-1.1B-Chat-v1.0")
                case _:
                    st.error("Selected model is not supported.")
        

        decoded_output = generate_answer(model, formatted_prompt, temperature, top_p, top_k, repetition_penalty, no_repeat_ngram_size, max_new_tokens)

        
        # Extract the first "Answer:" / gpt2 doesnt have stop sequence, so we need to extract the answer manually
        if "Answer:" in decoded_output:
            answer = decoded_output.split("Answer:")[1].split("Question:")[0].strip()
        else:
            answer = decoded_output.strip()

        #
        st.session_state.generated.append(answer + "\n\nSources:\n" + formatted_sources)


        #save the question and answer to the database
        try:
            insert_question_answer(query, answer, model.config._name_or_path)
        except Exception as e:
            print(f"❌ Failed to save question and answer to the database: {e}")

        #print the answer to the chat window
        chat_message(answer + "\n\nSources:\n" + formatted_sources, is_user=False, allow_html=True)
            


   
    #TODO: put a max token limiter or find a way to resume the context
if __name__ == "__main__":
    main()