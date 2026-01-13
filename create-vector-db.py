import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def create_vector_db(pdf_path: str, vector_db_path: str):
    """Create a vector database from a PDF file."""
    print("📄 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✓ Loaded {len(docs)} pages")

    print("✂️ Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    print("🧠 Generating embeddings...")
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )
    
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    print("💾 Saving vector database...")
    vector_db.save_local(vector_db_path)
    print(f"✓ Vector DB saved to {vector_db_path}")
    
    return vector_db

if __name__ == "__main__":
    PDF_FILE = "./recipes-book.pdf"
    VECTOR_DB_DIR = "./vector_store"
    
    create_vector_db(PDF_FILE, VECTOR_DB_DIR)