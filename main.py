import streamlit as st

st.set_page_config(page_title="ExtractLM: Intelligent Data Chat", page_icon=":robot:")
pdf_chat_page = st.Page("1_pdf_general_chat.py", title="PDF and General Chat")#, icon=":page_facing_up:")
setup_csv_page = st.Page("setup_csv.py", title="Setup CSV")# ,icon=":gear:")
csv_chat_page = st.Page("3_csv_chat.py", title="Chat with CSV")# ,icon=":bar_chart:")

pg = st.navigation([pdf_chat_page, setup_csv_page,csv_chat_page])
# Title and Introduction
st.title("ExtractLM: Intelligent Data Chat" + ":robot:")
st.subheader("Chat with your Data through CSVs and PDFs")

st.markdown(
    """
Welcome to **ExtractLM**, an intelligent Retrieval-Augmented Generation (RAG) application designed to make data interaction seamless and intuitive. With ExtractLM, you can chat with your data stored in CSV files and PDF documents, asking questions and receiving detailed, context-aware answers.

---

### How to Use ExtractLM:

#### 1. Chat with PDFs :page_with_curl:

- **Upload your PDFs**: Start by uploading your PDF documents using the upload option next to the chat input box.
- **Ask your Questions**: Simply type in your query. ExtractLM will search through the uploaded documents to provide relevant answers.
- **Automatic Web Search**: If the uploaded documents do not contain the required information, ExtractLM will intelligently rewrite your query and conduct a web search to find the best possible answers.

#### 2. Chat with CSVs :bar_chart:

- **Setup CSV Data**: First, navigate to the **"Setup CSV"** page to upload your desired CSV file.
- **Preview Data**: After uploading, you'll be able to preview your data to ensure everything is correct.
- **Define Data Dictionary**: Enter essential information about each column in your dataset using the provided form. This step helps ExtractLM understand your data better.
- **Start Chatting**: Once the setup is complete, proceed to the **"CSV Chat"** page where you can start interacting with your CSV file. Ask questions and receive data-driven insights directly from your dataset.

---

### Get Started Now:
Choose an option from the sidebar to begin chatting with your data!

- **Setup CSV**: Begin the setup process for your CSV data.
- **CSV Chat**: Interact with your CSV data in a conversational manner.
- **PDF Chat**: Start querying your PDF documents.

---

Happy Data Chatting! :speech_balloon:
    """
)

# Additional space for a clean layout
st.write("\n\n")

pg.run()