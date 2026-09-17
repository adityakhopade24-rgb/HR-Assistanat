import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
load_dotenv()

PDF_PATH = "C:/Users/Acer/HR Assistant/HR-Management-by-Pravin-Durai.pdf"
CHROMA_PATH = "./chroma_db"

def ingest():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"{PDF_PATH} not found.")
    
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("Splitting document...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Total Chunks : {len(chunks)}")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating Chroma DB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_PATH
    )
    print("Chroma Database Created Successfully")

if __name__ == "__main__":
    ingest()