from typing import TypedDict, List, Literal, Annotated
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI
from tavily import TavilyClient

# Load config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Vector store lookup (already embedded PDF)
VECTOR_STORE_NAME = "recipes-book-vector-store"
store_id = None
for s in openai_client.vector_stores.list().data:
    if s.name == VECTOR_STORE_NAME:
        store_id = s.id
        break
if not store_id:
    raise ValueError("❌ Vector store not found!")

print(f"♻ Using vector store ID: {store_id}")

# Tool definitions
@tool
def rag_search(query: str) -> str:
    """Search OpenAI vector store for relevant PDF chunks from internal recipe documents"""
    print(f"🔍 RAG Search called with query: {query}")
    res = openai_client.vector_stores.search(
        vector_store_id=store_id,
        query=query,
        max_num_results=3,
        ranking_options={"score_threshold": 0.5, "ranker": "auto"}
    )
    if not res.data:
        return "No relevant information found in internal documents."
    out = []
    for item in res.data:
        for block in item.content or []:
            if block.type == "text" and block.text:
                out.append(block.text)
    result = "\n\n".join(out) if out else "No relevant text extracted from internal documents."
    print(f"✅ RAG Search returned {len(out)} chunks")
    return result

@tool
def web_search(query: str) -> str:
    """Perform a web search using Tavily to find current information from the internet"""
    print(f"🌐 Web Search called with query: {query}")
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(query, max_results=3)
        
        # Format the results nicely
        results = []
        for result in response.get('results', []):
            results.append(f"Title: {result.get('title', 'N/A')}\nContent: {result.get('content', 'N/A')}\nURL: {result.get('url', 'N/A')}")
        
        final_result = "\n\n---\n\n".join(results) if results else "No web search results found."
        print(f"✅ Web Search returned {len(results)} results")
        return final_result
    except Exception as e:
        print(f"❌ Web search error: {str(e)}")
        return f"Web search error: {str(e)}"

# Define state with message reducer
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

# System prompt
SYSTEM_PROMPT = """You are a helpful assistant that MUST use both tools to answer questions:

1. FIRST use rag_search to find information from internal documents
2. THEN use web_search to find information from the internet
3. After using BOTH tools, provide a comprehensive answer that clearly separates the findings

Format your final answer like this:

**Based on your internal data (RAG):**
[Summarize what you found from rag_search here]

**Based on web search results:**
[Summarize what you found from web_search here]

Always use BOTH tools before giving your final answer."""

# Bind tools to LLM
llm_with_tools = llm.bind_tools([rag_search, web_search])

# Agent node with structured output
def agent_node(state: AgentState):
    print("\n🤖 Agent thinking...\n")
    messages = state["messages"]
    
    # Only add system message if this is the first call (no system message exists)
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Tool execution node
tools = [rag_search, web_search]
tool_node = ToolNode(tools)

# Router function to decide next step
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"🔧 Tool calls detected: {[tc['name'] for tc in last_message.tool_calls]}")
        return "tools"
    # Otherwise, end
    print("✅ No more tool calls, ending workflow")
    return "end"

# Build graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
graph.add_edge("tools", "agent")

app = graph.compile()

# Run
if __name__ == "__main__":
    initial_state = {
        "messages": [HumanMessage(content="show me the recipe for Vegetable Spring Rolls")]
    }
    
    print("\n" + "="*60)
    print("🔍 Starting agent workflow...")
    print("="*60)
    
    final = app.invoke(initial_state)
    
    print("\n" + "="*60)
    print("📌 Final answer:")
    print("="*60)
    print(final["messages"][-1].content)
    
    print("\n" + "="*60)
    print("📋 Full Message History:")
    print("="*60)
    for i, msg in enumerate(final["messages"]):
        print(f"\n[{i}] {type(msg).__name__}: {msg.content[:100] if hasattr(msg, 'content') and msg.content else '(tool call)'}")