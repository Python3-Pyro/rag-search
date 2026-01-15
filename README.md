# RAG Search Agent

A powerful AI agent that combines **Retrieval-Augmented Generation (RAG)** with **web search** capabilities using LangGraph, OpenAI, and Tavily API.

## 🌟 Features

- **Hybrid Search**: Uses both internal RAG search and web search to provide comprehensive answers
- **LangGraph Integration**: Built with LangGraph for robust agentic workflows
- **OpenAI Vector Store**: Leverages OpenAI's vector store for semantic search on embedded documents
- **Web Search**: Tavily API integration for real-time internet information
- **Streamlit UI**: Interactive web interface for easy interaction
- **Multi-tool Support**: Intelligent tool selection and execution

## 📋 Prerequisites

- Python 3.9+
- OpenAI API Key
- Tavily API Key
- Virtual environment (recommended)

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Python3-Pyro/rag-search.git
   cd rag-search
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install langgraph langchain langchain-openai openai tavily-python streamlit python-dotenv
   ```

4. **Set up environment variables**:
   ```bash
   cp sample.env .env
   ```
   
   Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## 📁 Project Structure

In order of execution for our hands-on demo

| File | Description |
|------|-------------|
| `text-embedding.py` | Text embedding utilities |
| `create-vector-db.py` | Script to create vector database from documents |
| `search-vectordb.py` | Utility to search the vector database |
| `create-openai-vector-store.py` | Script to create and manage OpenAI vector stores |
| `rag-example.py` | Example RAG implementation |
| `lg-agent.py` | Main agentic workflow using LangGraph - combines RAG and web search |
| `streamlit_app.py` | Interactive web UI for the RAG search agent |
| `recipes-book.pdf` | Sample PDF document for RAG |
| `.env` | Environment variables (API keys) |

## 💻 Usage

### Option 1: Run the LangGraph Agent (Command Line)

```bash
python lg-agent.py
```

The agent will:
1. Process your query
2. Search internal documents using RAG
3. Perform a web search
4. Return a comprehensive answer combining both sources

Example output:
```
**Based on your internal data (RAG):**
[Findings from internal documents]

**Based on web search results:**
[Findings from web search]
```

### Option 2: Run the Streamlit App (Web UI)

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Option 3: Create a Vector Store

```bash
python create-openai-vector-store.py
```

This will embed your documents and create a vector store in OpenAI.

## 🔧 Configuration

### Vector Store Setup
The agent expects a vector store named `recipes-book-vector-store`. You can:
- Create a new vector store using `create-openai-vector-store.py`
- Modify the `VECTOR_STORE_NAME` in `lg-agent.py` to match your store

### Tool Customization
To add more tools or modify search behavior, edit the tool definitions in `lg-agent.py`:
- `rag_search()` - Search internal documents
- `web_search()` - Search the internet

## 🤖 How It Works

1. **Agent Initialization**: The agent receives a user query
2. **Tool Selection**: LLM decides which tools to use (or both)
3. **Parallel Execution**: Tools execute and gather information
4. **Response Generation**: Agent synthesizes findings into a comprehensive answer

### System Prompt
The agent is instructed to:
- FIRST use RAG to search internal documents
- THEN use web search for external information
- FINALLY provide a comprehensive answer separating both sources

## 📊 Example Queries

```python
# Example in lg-agent.py
initial_state = {
    "messages": [HumanMessage(content="show me the recipe for Vegetable Spring Rolls")]
}
```

## 🔐 Security

- **Never commit `.env`** - API keys are sensitive (`.gitignore` protects this)
- Use environment variables for all credentials
- Keep API keys secret and rotate them regularly

## 📦 Dependencies

- `langgraph` - Agent orchestration framework
- `langchain` & `langchain-openai` - LLM integrations
- `openai` - OpenAI API client
- `tavily-python` - Web search API
- `streamlit` - Web UI framework
- `python-dotenv` - Environment variable management

## 🐛 Troubleshooting

### Vector store not found
Ensure the vector store exists in your OpenAI account and matches the name in `lg-agent.py`.

### API key errors
Verify your `.env` file has correct API keys and the file is in the project root.

### Tool execution fails
Check internet connection and API quota/rate limits.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available under the MIT License.

## 📞 Support

For issues or questions:
1. Check the [GitHub Issues](https://github.com/Python3-Pyro/rag-search/issues)
2. Review the code comments for implementation details
3. Consult the [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---

**Happy searching! 🔍**
