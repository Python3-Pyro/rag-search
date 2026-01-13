import streamlit as st
import os
from dotenv import load_dotenv
import time
from typing import List

from openai import OpenAI
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langsmith import Client

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Config
UPLOADED_FILE_DIR = "./uploaded_pdfs"
VECTOR_STORE_NAME_PREFIX = "pdf_rag_"

# Create directories if they don't exist
os.makedirs(UPLOADED_FILE_DIR, exist_ok=True)

# ============== PAGE CONFIG ==============
st.set_page_config(page_title="PDF RAG Agent", layout="wide")
st.title("📄 PDF RAG Agent")
st.markdown("Upload a PDF and ask questions about its content!")

# ============== SIDEBAR ==============
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Vector DB name input
    custom_db_name = st.text_input(
        "Vector Store Name",
        placeholder="Enter a name for your vector store",
        help="Give your vector store a memorable name"
    )
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file:
        st.success(f"✓ File loaded: {uploaded_file.name}")
        
        # Create vector store button
        if st.button("🚀 Create Vector Store on OpenAI"):
            with st.spinner("Creating OpenAI vector store..."):
                try:
                    # Use custom name if provided, otherwise generate from filename
                    if custom_db_name.strip():
                        store_name = custom_db_name.strip()
                    else:
                        store_name = VECTOR_STORE_NAME_PREFIX + uploaded_file.name.replace(".pdf", "").replace(" ", "_")
                    
                    # Check if vector store already exists
                    st.info("🔎 Checking for existing vector store...")
                    existing_store_id = None
                    stores = openai_client.vector_stores.list()
                    
                    for s in stores.data:
                        if s.name == store_name:
                            existing_store_id = s.id
                            break
                    
                    if existing_store_id:
                        st.success(f"♻️ Vector store found: {existing_store_id}")
                        st.session_state.vector_store_id = existing_store_id
                        st.session_state.vector_db_ready = True
                    else:
                        # Create new vector store
                        st.info("🆕 Creating new vector store...")
                        new_store = openai_client.vector_stores.create(name=store_name)
                        store_id = new_store.id
                        st.success(f"✅ Vector store created: {store_id}")
                        
                        # Save uploaded file
                        pdf_path = os.path.join(UPLOADED_FILE_DIR, uploaded_file.name)
                        with open(pdf_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Upload to OpenAI vector store with chunking strategy
                        st.info("📤 Uploading PDF to OpenAI...")
                        chunking_strategy = {
                            "static": {
                                "max_chunk_size_tokens": 400,
                                "chunk_overlap_tokens": 120,
                            },
                            "type": "static"
                        }
                        
                        openai_client.vector_stores.files.upload(
                            vector_store_id=store_id,
                            file=open(pdf_path, "rb"),
                            chunking_strategy=chunking_strategy
                        )
                        
                        st.success("✅ PDF uploaded to OpenAI vector store!")
                        st.info("⏳ Waiting for vector store to index (10 seconds)...")
                        time.sleep(10)
                        
                        st.session_state.vector_store_id = store_id
                        st.session_state.vector_db_ready = True
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============== AGENT SETUP ==============
@st.cache_resource
def initialize_agent(store_id: str):
    """Initialize the RAG agent with OpenAI vector store"""
    
    # Define RAG tool
    @tool
    def rag_search(query: str) -> List[str]:
        """Search OpenAI vector store for relevant PDF chunks"""
        results = openai_client.vector_stores.search(
            vector_store_id=store_id,
            query=query,
            max_num_results=3,
            ranking_options={"score_threshold": 0.75, "ranker": "auto"}
        )
        
        if not results.data:
            return ["No relevant information found."]
        
        out = []
        for item in results.data:
            for block in item.content or []:
                if block.type == "text" and block.text:
                    out.append(block.text)
        
        return out if out else ["No relevant text extracted."]
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    
    # Create agent
    agent = create_agent(
        llm,
        tools=[rag_search],
        system_prompt="You are a helpful assistant that answers questions based on the provided PDF document. Use the rag_search tool to find relevant information."
    )
    
    return agent, rag_search

# ============== MAIN CHAT INTERFACE ==============
if st.session_state.get("vector_db_ready", False):
    st.success("✅ Vector store is ready. You can ask questions now!")
    
    # Initialize agent
    try:
        agent, rag_search = initialize_agent(st.session_state.vector_store_id)
        
        # Chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask a question about the PDF...")
        
        if user_input:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = agent.invoke({"messages": [("user", user_input)]})
                        response = result["messages"][-1].content
                        
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        st.info("Make sure you've uploaded a PDF and created the vector store first.")

else:
    st.info("👈 Upload a PDF and create a vector store using the sidebar to get started!")
