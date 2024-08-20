import streamlit as st
import os
from dotenv import load_dotenv
#from llm_pdf_general_chat import get_vectorstore, graph
from llm_pdf_chat import get_vectorstore, graph
import pprint

# Load environment variables
load_dotenv()

# Streamlit configuration
#st.set_page_config(page_title="Adaptive RAG Application",layout="centered",page_icon=":page_with_curl:")

# Function to handle the uploaded file and get the vectorstore retriever
def handle_file_upload(uploaded_file):
    if uploaded_file is not None and uploaded_file.type == "application/pdf":
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return get_vectorstore(file_path)
    else:
        st.error("Please upload a valid PDF file.")
        return None

# Main chat interface
st.title("Adaptive Retrieval-Augmented Generation (RAG) Application")
st.write("Chat with the system to retrieve information from uploaded PDFs and web search.")

if "history" not in st.session_state:
    st.session_state["history"] = []

if "retriever" not in st.session_state:
    st.session_state["retriever"] = None

# Display chat history
for message in st.session_state["history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Layout for text input and file upload

user_input = st.chat_input("Ask your question...")
with st.sidebar:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

# Handle file upload
if uploaded_file:
    st.session_state["retriever"] = handle_file_upload(uploaded_file)

# User input handling
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["history"].append({"role": "user", "content": user_input})

# Retrieve and generate response
    if st.session_state["retriever"]:
        app = graph(st.session_state["retriever"])

        # Simulate a state object to pass through the graph
        state = {"question": user_input}
        for output in app.stream(state):
            for key, value in output.items():
                # Node
                pprint.pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint.pprint("\n---\n")

        response = value["generation"]
    else:
        response = "Please upload a PDF document to enable document retrieval."

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state["history"].append({"role": "assistant", "content": response})