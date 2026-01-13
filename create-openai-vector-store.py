from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Step 1: Check if vector store exists on OpenAI ----
VECTOR_STORE_NAME = "recipes-book-vector-store"

print("\n🔎 Checking if vector store already exists on OpenAI...")

existing_store_id = None
stores = client.vector_stores.list()

for s in stores.data:
    if s.name == VECTOR_STORE_NAME:
        existing_store_id = s.id
        break

if existing_store_id:
    print(f"♻️ Vector store FOUND. Reusing existing store (ID: {existing_store_id})")
else:
    print("🆕 Vector store NOT found. Creating a new one...")
    new_store = client.vector_stores.create(name=VECTOR_STORE_NAME)
    existing_store_id = new_store.id
    print(f"✅ New vector store created (ID: {existing_store_id})")
    
    chunking_strategy = {
    "static": {
        "max_chunk_size_tokens": 800,
        "chunk_overlap_tokens": 100,
        },
    "type": "static"
    }

    print("\n📤 Uploading PDF to vector store with static chunking strategy...")
    
    client.vector_stores.files.upload(
        vector_store_id=existing_store_id,
        file=open("recipes-book.pdf", "rb"),
        chunking_strategy=chunking_strategy
    )

    print("📄 File successfully added to vector store (File object ID:", existing_store_id, ")")
    
# # Test similarity search
print("\n🔍 Testing semantic similarity search...")
if not existing_store_id:
    print("\n⏳ Waiting 10 seconds for vector store indexing to update...")
    time.sleep(10)
results = client.vector_stores.search(
    vector_store_id=existing_store_id,
    query="recipe for bread pakoda",
    max_num_results=3,  # instead of k
    ranking_options={
                "score_threshold": 0.75,
                "ranker": "auto"
                }
)

print("\n🔎 Search Results:\n", results)
print("\n🎉 Semantic search completed!")