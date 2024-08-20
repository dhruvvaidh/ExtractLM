import streamlit as st
#st.set_page_config(page_title="Setup your csv file",page_icon=":gear:")
from llm import csv_to_db
import sqlite3
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def get_vectorstore(filepath):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    dbname = filename+".db"
    conn = sqlite3.connect(dbname)
    db = csv_to_db(conn=conn, dbname=dbname, filepath=filepath)
    return db, conn

def handle_file_upload(uploaded_file):
    if uploaded_file is not None and uploaded_file.name.endswith('.csv'):
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return get_vectorstore(file_path)
    else:
        st.error("Please upload a valid CSV file.")
        return None, None
    

st.title("Setup CSV")
st.write("This page is to configure your csv file and give the AI Assistant more information about your data")

# Initialize session state variables if not already initialized
if "column_descriptions" not in st.session_state:
    st.session_state.column_descriptions = {}

if "filepath" not in st.session_state:
    st.session_state.filepath = None

uploaded_file = st.file_uploader("Upload your CSV")
if uploaded_file is not None and uploaded_file.name.endswith('.csv'):
        st.session_state.filepath = os.path.join("/tmp", uploaded_file.name)
        with open(st.session_state.filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
if uploaded_file:
    df = pd.read_csv(st.session_state.filepath)
    #db,conn = handle_file_upload(files)
    st.write("Uploaded CSV: ")
    st.dataframe(df.head())

    st.write("Write brief information about all the columns in the dataset")

    # Create a form
    with st.form("Data Dictionary"):
        st.write("Please enter descriptions for the following columns:")

        # Loop through each column in the DataFrame and create a text input for its description
        for column in df.columns:
            description = st.text_input(f"Description for {column}", key=column)
            st.session_state.column_descriptions[column] = description

        # Submit button for the form
        submit_button = st.form_submit_button(label='Submit')

    # Display the dictionary if the form is submitted
    if submit_button:
        st.write("Data Dictionary:")
        st.json(st.session_state.column_descriptions)

def get_data(column_descriptions = st.session_state.column_descriptions):
    """Returns the database, database connector and the data dictionary"""
    key_value_string = ', '.join([f"{key}:{value}" for key, value in column_descriptions.items()])

    return key_value_string, st.session_state.filepath#,db,conn