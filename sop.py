import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# Constants
PDF_PATH = "en_SC.pdf"
VECTOR_DB_DIR = "./vector_db"
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "your-openai-api-key-here")

# Function to initialize the vector database with FAISS
def initialize_vector_db(pdf_path):
    if not os.path.exists(VECTOR_DB_DIR):
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Create embeddings and store in FAISS
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_db = FAISS.from_documents(texts, embeddings)
        vector_db.save_local(VECTOR_DB_DIR)
        st.write("Vector database created successfully!")
    else:
        st.write("Loading existing vector database...")
    
    # Load the vector database
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_db = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    return vector_db

# Function to get response from the LLM
def get_response(query, vector_db):
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    result = qa_chain({"query": query})
    response = result["result"]
    sources = result["source_documents"]
    
    # Format response with quotes from the book
    quoted_response = f"{response}\n\n**References from the Book Steps to Christ by Ellen G. White:**\n"
    for i, doc in enumerate(sources, 1):
        quoted_response += f"{i}. \"{doc.page_content[:200]}...\" (Page {doc.metadata.get('page', 'unknown')})\n"
    
    return quoted_response

# Streamlit UI
def main():
    st.title("The Book: Steps to Christ")
    st.write("Ask a question, and receive an answer referencing the Book.")

    # Initialize vector database
    if "vector_db" not in st.session_state:
        with st.spinner("Initializing vector database... This may take a moment."):
            st.session_state.vector_db = initialize_vector_db(PDF_PATH)

    # Query input
    query = st.text_input("Enter your question:", placeholder="e.g., What does Ellen G. White say about prayer?")
    
    # Process query and display response
    if st.button("Submit") and query:
        with st.spinner("Searching the book and generating response..."):
            response = get_response(query, st.session_state.vector_db)
            st.markdown(response)

if __name__ == "__main__":
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
        st.error("Please provide a valid OpenAI API key in the code or via Streamlit secrets.")
    else:
        main()
