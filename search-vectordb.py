import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

VECTOR_DB_DIR = "./vector_store"

# ---- Load local vector store ----
print("📂 Loading local vector database...")
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

vector_db = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
print("✓ Vector database loaded successfully!")

# ---- Run similarity search ----
print("\n🔍 Searching vector database...")
query = "Recipe for bread pakoda"
results = vector_db.similarity_search(query, k=3)

print(f"\n🔎 Search results for '{query}':\n")
for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(f"Content: {doc.page_content[:300]}...")
    if doc.metadata:
        print(f"Metadata: {doc.metadata}")
    print()

print("🎉 Search completed!")