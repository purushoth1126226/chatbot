from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import logging
from datetime import datetime

# Configure logging
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, "app.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Use st.cache to avoid re-computing the embeddings if PDF content doesn't change
@st.cache(allow_output_mutation=True)
def get_embeddings_and_chain(pdf_content):
    try:
        text_chunks = CharacterTextSplitter(separator="\n", chunk_size=1000,
                                            chunk_overlap=200, length_function=len).split_text(pdf_content)
        
        if not text_chunks:
            raise ValueError("Text chunks are empty. Check your PDF content extraction.")

        embeddings = OpenAIEmbeddings()
        if not embeddings:
            raise ValueError("Embeddings are not generated correctly.")
        
        docsearch = FAISS.from_texts(text_chunks, embeddings) 
        if not docsearch:
            raise ValueError("Faiss index is not created correctly.")
        
        llm = OpenAI() 
        chain = load_qa_chain(llm, chain_type="stuff")
        
        logging.info("Embeddings and chain successfully generated.")
        return embeddings, docsearch, chain

    except ValueError as e:
        logging.error(f"Error in get_embeddings_and_chain: {e}")
        st.error(f"Error: {e}")
        return

def log_user_interaction(query, response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"User Query: {query} - Bot Response: {response} (Timestamp: {timestamp})"
    logging.info(log_message)

def main():
    load_dotenv()
    st.set_page_config(page_title="My Chatbot App")
    
    st.title("Welcome to My Chatbot App 🤖")
    
    # List PDF files in the "pdf_files" directory
    pdf_files = [file for file in os.listdir("pdf_files") if file.lower().endswith(".pdf")]

    if not pdf_files:
        st.error("No PDF files found in the 'pdf_files' directory.")
        return

    # Dropdown to choose a PDF file
    selected_pdf_file = st.sidebar.selectbox("Select PDF File", pdf_files)
    
    # Full path to the selected PDF file
    pdf_file_path = os.path.join("pdf_files", selected_pdf_file)
    
    # extract the text
    pdf_text = extract_text_from_pdf(pdf_file_path)
    
    try:
        # Fetch embeddings and chain
        embeddings, docsearch, chain = get_embeddings_and_chain(pdf_text)
    except ValueError as e:
        st.error(f"Error: {e}")
        return
    
    # show user input
    query = st.text_input("Type your question:")
    if query:
        docs = docsearch.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        
        # Log the user interaction
        log_user_interaction(query, response)

        # Display the response in the app below the search box
        st.write(response)

if __name__ == '__main__':
    main()
