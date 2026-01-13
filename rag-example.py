import os
from dotenv import load_dotenv

from typing import List
from openai import OpenAI
from langchain_openai import ChatOpenAI


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


openai_client = OpenAI(api_key=OPENAI_API_KEY)

VECTOR_STORE_NAME = "recipes-book-vector-store"

# ---- Lookup vector store ID ----
store_id = None
stores = openai_client.vector_stores.list()
for s in stores.data:
    if s.name == VECTOR_STORE_NAME:
        store_id = s.id
        break

if not store_id:
    raise ValueError("❌ Vector store not found!")

print(f"♻ Using vector store ID: {store_id}")

# ---- Define RAG search tool ----
def rag_search(query: str) -> List[dict]:
    """Search OpenAI vector store for relevant PDF chunks and return text content with sources"""
    response = openai_client.vector_stores.search(
        vector_store_id=store_id,
        query=query,
        max_num_results=3,
        ranking_options={"score_threshold": 0.75, "ranker": "auto"}
    )

    # Correct unwrapping and parsing
    if not response.data:
        return [{"content": "No relevant information found.", "filename": ""}]

    extracted = []
    for item in response.data:
        for block in item.content or []:
            if block.type == "text" and block.text:
                extracted.append({
                    "content": block.text,
                    "filename": item.filename
                })

    return extracted if extracted else [{"content": "No relevant text extracted from results.", "filename": ""}]

# ---- Initialize LLM ----
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0
)

# ---- Run RAG logic manually ----
if __name__ == "__main__":
    query = "show me the recipe for bread pakora"
    print(f"\n🔍 Searching vector store for: {query}...")

    # 1. Call your tool directly to get context
    context_results = rag_search(query)
    
    # 2. Format context with citations
    context_text = ""
    sources = set()
    for result in context_results:
        context_text += f"{result['content']}\n\n"
        if result['filename']:
            sources.add(result['filename'])
    
    sources_str = ", ".join(sources) if sources else "Unknown"

    # 3. Build a prompt that instructs the LLM to cite sources
    prompt = f"""
    You are a helpful assistant. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    
    IMPORTANT: Include the source filename(s) as citations in your answer. When referencing information from the context, mention which document it came from.
    
    Context:
    {context_text}
    
    Question: {query}
    
    Answer:
    """

    print("\n🤖 Generating response...\n")
    
    # 4. Get response from LLM
    # Use .invoke() for the ChatOpenAI model
    response = llm.invoke(prompt)

    # 5. Print the result with sources
    print("📌 Final Answer:\n", response.content)
    print(f"\n📚 Sources: {sources_str}")