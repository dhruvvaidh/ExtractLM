import streamlit as st
#st.set_page_config(page_title="Chat with CSVs", page_icon=":bar_chart:")
from setup_csv import get_data
from dotenv import load_dotenv
from llm import decompose_prompt, qa_pairs, compile_answers, csv_to_db
import sqlite3
import os

# Load environment variables
load_dotenv()

def get_vectorstore(filepath):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    dbname = filename+".db"
    conn = sqlite3.connect(dbname)
    db = csv_to_db(conn=conn, dbname=dbname, filepath=filepath)
    return db, conn

# Function to handle user input
def handle_userinput(user_question, db, chat_history, data_dictionary):
    sub_questions = decompose_prompt(user_question, data_dictionary)
    sub_qa_pairs = qa_pairs(sub_questions, db)
    final_output = compile_answers(sub_qa_pairs, user_question)
    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "bot", "content": final_output})
    return chat_history

# Main function for Streamlit app
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Chat with CSVs")
st.write("Ask questions about your csv and get detailed answers.")

data_dictionary, filepath = get_data()
if filepath:
    db, conn = get_vectorstore(filepath)

# Display chat history using Streamlit chat elements
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

# User input and submit button
user_question = st.chat_input("Ask a question about the csv")

if user_question:
    st.session_state.chat_history = handle_userinput(user_question, db, st.session_state.chat_history, data_dictionary)
    st.rerun()

conn.close()