import streamlit as st
from langchain_community.vectorstores import Chroma
from src.query_handler import similarity_query
from src.embedder import get_huggingface_embedder
from src.rag_pipeline import add_single_file_to_vectorstore, run_rag_pipeline
from src.loader import save_uploaded_file, load_documents
from src.db_handler import insert_question_answer, insert_feedback
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompt_template import rag_prompt_template
from src.gdrive_handler import download_all_from_folder
from streamlit_chat import message as chat_message  # Rename to avoid conflict
from langsmith import traceable
from src.agents.file_writer_agent import file_writer_agent  # Assuming this is your agent logic

import os
import time
import wx

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

@traceable
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
    maximum_model_Tokens = 2048

    #get tokenizer and model
    tokenizer = get_tokenizer(model_name=model.config._name_or_path)


    
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

def truncate_to_token_budget(text, tokenizer, max_tokens):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def clear_chat_history():
    #clear the chat history and reset the state variables
    st.session_state.past = []
    st.session_state.generated = []

def render_chat_messages(prefix=""):
    for i in range(len(st.session_state.past)):
        chat_message(st.session_state.past[i], is_user=True, key=f"{prefix}{i}_user")

        # Extract bot data
        bot_msg = st.session_state.generated[i]
        if isinstance(bot_msg, dict) and "answer" in bot_msg:
            bot_answer = bot_msg["answer"]
            bot_sources = bot_msg.get("sources", [])
        else:
            bot_answer = str(bot_msg)
            bot_sources = []

        chat_message(bot_answer + "\n\nSources:\n" + "\n".join(bot_sources), key=f"{prefix}{i}_ai", allow_html=True)

        # Track feedback click state in session state
        if f"{prefix}{i}_feedback" not in st.session_state:
            st.session_state[f"{prefix}{i}_feedback"] = None  # None / "up" / "down"

        col1, col2, col3, col4, col5 = st.columns([.25, .20, .20, 1, 1])
        with col2:
            thumbs_up = st.button("👍", key=f"{prefix}{i}_thumbs_up")
        with col3:
            thumbs_down = st.button("👎", key=f"{prefix}{i}_thumbs_down")

        if thumbs_up:
            insert_feedback(st.session_state.generated[i].get("message_id"), "positive")
            st.session_state[f"{prefix}{i}_feedback"] = "up"
        elif thumbs_down:
            insert_feedback(st.session_state.generated[i].get("message_id"), "negative")
            st.session_state[f"{prefix}{i}_feedback"] = "down"

        # Show colored feedback below buttons
        feedback = st.session_state[f"{prefix}{i}_feedback"]
        with col4:
            if feedback == "up":
                placeholder = st.empty()
                placeholder.markdown("<span style='color: green; font-weight: bold;'>You liked this response 👍</span>", unsafe_allow_html=True)
                time.sleep(3)
                placeholder.empty()
            elif feedback == "down":
                placeholder = st.empty()
                placeholder.markdown("<span style='color: red; font-weight: bold;'>You disliked this response 👎</span>", unsafe_allow_html=True)
                time.sleep(3)
                placeholder.empty()

def should_use_file_writer_agent(query: str) -> bool:
    file_intent_keywords = [
        "save to file", "write a file", "generate document", 
        "create file", "write to disk", "output to txt", 
        "generate markdown", "write code to file"
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in file_intent_keywords)

#load vector store when loading the page 
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



def main():

    with st.sidebar:
        ###### Choose model and parameters
        st.title('💬 Models and parameters')
        st.write('Choose the model that you want to use.')
        
        model = st.selectbox("Select a model", ("TinyLlama-1.1B-Chat-v1.0","gpt2"), key="model")
        
        # Collapsible section for parameters
        with st.expander("🔧 Generation Parameters", expanded=False):
            temperature = st.slider(
                'Temperature', min_value=0.01, max_value=1.00, value=0.7, step=0.01,
                help="Randomness of generated output"
            )
            if temperature >= 1:
                st.warning('Values exceeding 1 produce more creative/random output and increased hallucinations.')
            if temperature < 0.1:
                st.warning('Low values produce more deterministic output. Recommended default: 0.7')

            top_p = st.slider(
                'Top-p (nucleus sampling)', min_value=0.01, max_value=1.0, value=0.9, step=0.01,
                help="Top-p percentage of most likely tokens for output generation"
            )

            top_k = st.slider(
                'Top-k', min_value=0, max_value=100, value=50, step=1,
                help="Top-k most likely tokens to sample from"
            )

            repetition_penalty = st.slider(
                'Repetition Penalty', min_value=0.8, max_value=2.0, value=1.1, step=0.1,
                help="Penalty for repeated tokens (1.0 = no penalty)"
            )

            no_repeat_ngram_size = st.slider(
                'No-Repeat N-Gram Size', min_value=0, max_value=10, value=3, step=1,
                help="Prevent repetition of n-grams of this size"
            )

            max_new_tokens = st.slider(
                'Max New Tokens', min_value=1, max_value=512, value=256, step=1,
                help="Max tokens to generate beyond the input"
            )


        ################### Load documents from folder
        st.subheader("📂 Load Documents from Local Folder Path or single files")
       
         ######## Folders button
        if st.button("➕ Upload Folders"):
            app = wx.App(False)
            dialog = wx.DirDialog(None, "Select a folder:", style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
            
            if dialog.ShowModal() == wx.ID_OK:
                folder_path = dialog.GetPath()
                
                #verify if its path 
                if os.path.isdir(folder_path):
                    docs = load_documents(folder_path)
                    run_rag_pipeline(docs)
                    st.success(f"✅ Added documents from the folder path: {folder_path}.")
                else:
                    st.error("❌ The specified path does not exist.")
                
            dialog.Destroy()
            
        ############### Upload single files
        if st.button("➕ Upload single files"):
                st.session_state.show_uploader = True
        
        # File uploader (only shows when "+" is clicked)
        if st.session_state.show_uploader:
            uploaded_file = st.file_uploader(
                "Upload your document",
                type=["txt", "pdf", "docx"],
                label_visibility="collapsed",
                key="file_uploader",
            )
            # Check if a file is uploaded
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

    # Render chat messages using streamlit-chat
    # Render existing chat history
    render_chat_messages("initial_")
    
    
    query = st.chat_input("Ask your question:")
    # User-provided prompt
    if query:
        with st.spinner("Generating Response..."):
            
            
                    
               
            
            #append the question to the chat history
            st.session_state.past.append(query)

            # Load the model based on user selection
            match model:
                    case "gpt2":
                        model = AutoModelForCausalLM.from_pretrained("Models/HuggingFace/gpt2")               
                    case "TinyLlama-1.1B-Chat-v1.0":
                        model = AutoModelForCausalLM.from_pretrained("Models/HuggingFace/TinyLlama-1.1B-Chat-v1.0")
                    case _:
                        st.error("Selected model is not supported.")

            # results from the vector store 
            Vector_results = similarity_query(vector_store, query, 3)

            # context is only the result of the vector store 
            context =  " ".join([doc.page_content for doc in Vector_results])

            
            # Extract source file paths
            source_paths = [doc.metadata.get("source", "unknown file") for doc in Vector_results]
            formatted_sources = "\n".join([f"- {path}" for path in source_paths])

            #gets the last question and answer and the
            if st.session_state.past and st.session_state.generated:
                # Remove sources from the previous answer if present
                prev_answer = st.session_state.generated[-1]
                prev_question = st.session_state.past[-1]
                if isinstance(prev_answer, dict) and 'answer' in prev_answer:
                    answer_text = prev_answer['answer']
                    # Remove sources if appended at the end (split by "\n Sources:")
                    answer_text = answer_text.split("\n Sources:")[0].strip()
                else:
                    answer_text = str(prev_answer).split("\n Sources:")[0].strip()
                last_message = f"Question: {prev_question}\nAnswer: {answer_text}"
            else:
                last_message = ""

            # Truncate context to fit within token budget
            tokenizer = get_tokenizer(model_name=model.config._name_or_path)

            print("Token counts:")
            retrieved_context_len = truncate_to_token_budget(context, tokenizer, 1000)
            formatted_history_len = truncate_to_token_budget(last_message, tokenizer, 500)
            query_len = truncate_to_token_budget(query, tokenizer, 300)


            print("Context:", len(tokenizer.encode(retrieved_context_len)))
            print("History:", len(tokenizer.encode(formatted_history_len)))
            print("Query:", len(tokenizer.encode(query_len)))


            #create the prompt for the LLM
            formatted_prompt = rag_prompt_template.format(retrieved_context=retrieved_context_len,
                                                        formatted_history=formatted_history_len,
                                                        current_question=query
                                                        )


        
    
            decoded_output = generate_answer(model, formatted_prompt, temperature, top_p, top_k, repetition_penalty, no_repeat_ngram_size, max_new_tokens)

            
            # Extract the first "Answer:" / gpt2 doesnt have stop sequence, so we need to extract the answer manually
            if "Answer:" in decoded_output:
                answer = decoded_output.rsplit("Answer:", 1)[-1].split("Question:")[0].strip()
            else:
                answer = decoded_output.strip()

            # logic to check if the question should be handled by the file writer agent
            if should_use_file_writer_agent(query):
                with st.spinner("Using file writer agent..."):
                    # Write the answer to a local file
                        output_file = "local_model_answer.txt"
                        with open(output_file, "w", encoding="utf-8") as file:
                            file.write(answer)
            
            #save the question and answer to the database and return 
            try:
                message_id = insert_question_answer(query, answer, model.config._name_or_path)
            except Exception as e:
                print(f"❌ Failed to save question and answer to the database: {e}")

            
            st.session_state.generated.append({
                "answer": answer ,
                "sources":  source_paths,
                "message_id": message_id
            })
            
            # Render existing chat history
            render_chat_messages("post_")
            
            
            #reset source list
            source_paths = []

            #rerender the page
            st.rerun()
        

if __name__ == "__main__":
    main()