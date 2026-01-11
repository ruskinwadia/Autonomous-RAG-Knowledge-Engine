# Autonomous RAG Knowledge Engine

A document-based AI assistant built with **FastAPI**, **Streamlit**, **LangGraph**, and **ChromaDB**. Upload a PDF and chat with your document using advanced RAG (Retrieval-Augmented Generation).

## 🎬 Demo (YouTube)

[![Demo Video](https://img.youtube.com/vi/wM9__MPE7PQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=wM9__MPE7PQ)

---

## 🚀 Features

### Core Requirements ✅
| Feature | Implementation |
|---------|----------------|
| **Document Ingestion** | PDF parsing with `pdfplumber`, intelligent chunking |
| **Vector Retrieval** | Gemini embeddings + ChromaDB |
| **Source Citations** | Every answer includes `[DocumentName, Page X]` |
| **Guardrails** | Rejects off-topic questions gracefully |
| **Grounded Responses** | Only answers from document content |

### Bonus Features ✅
| Feature | Implementation |
|---------|----------------|
| **Streaming Responses** | Token-by-token streaming via SSE |
| **Hybrid Search** | BM25 keyword + vector similarity with Reciprocal Rank Fusion |
| **Model Selection** | Choose between ⚡ Fast (8B) or 🧠 Smart (70B) models |
| **"Glass Box" UI** | Real-time thinking process visibility |
| **Starter Questions** | Auto-generated based on document content |
| **Follow-up Suggestions** | 3 contextual questions after each answer |

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Frontend
        UI["Streamlit UI<br>(Port 8501)"]
    end
    
    subgraph Backend
        API["FastAPI<br>(Port 8000)"]
        Agent["LangGraph<br>Agent + Tools"]
    end
    
    subgraph Storage
        DB["ChromaDB<br>(Port 8001)"]
    end
    
    UI -->|"User Query"| API
    API --> Agent
    Agent -->|"Hybrid Search"| DB
    DB -->|"Relevant Chunks"| Agent
    Agent -->|"Streaming Response"| API
    API -->|"SSE Tokens"| UI

```

### Key Components

| Component | Technology | Details |
|-----------|------------|---------|
| **LLM** | Groq | ⚡ `llama-3.1-8b-instant` (fast) or 🧠 `llama-3.3-70b-versatile` (accurate) |
| **Embeddings** | Google Gemini | `gemini-embedding-001` with task-specific modes (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY) |
| **Vector Store** | ChromaDB | Dockerized persistent storage |
| **Keyword Search** | BM25 (rank_bm25) | Combined with vector via Reciprocal Rank Fusion |
| **Agent Framework** | LangGraph | StateGraph with tool calling and conditional routing |
| **Backend** | FastAPI | Streaming SSE responses, async endpoints |
| **Frontend** | Streamlit | Custom CSS, dark theme, real-time updates |

### Implemented Features

| Category | Feature | Implementation |
|----------|---------|----------------|
| **Ingestion** | PDF Parsing | `pdfplumber` with text + table extraction |
| **Ingestion** | Structured Tables | Column:value format (e.g., `Dry Storage: No \| Apron: Yes`) |
| **Ingestion** | Header/Footer Removal | Frequency analysis removes lines appearing on >80% of pages |
| **Ingestion** | Smart Chunking | 1000 chars with 100 overlap, page metadata preserved |
| **Retrieval** | Hybrid Search | BM25 keyword + Gemini vector embeddings |
| **Retrieval** | Reciprocal Rank Fusion | Merges results from both search methods |
| **Retrieval** | Rate Limit Handling | Tenacity retry with exponential backoff |
| **Agent** | Tool Calling | LangGraph with `retrieve_documents` tool |
| **Agent** | Guardrails | Rejects off-topic questions, enforces document grounding |
| **Agent** | Citations | Every answer includes `[DocumentName, Page X]` |
| **Agent** | Follow-up Questions | 3 contextual suggestions after each answer |
| **UI** | Model Selection | Choose between Fast (8B) and Smart (70B) |
| **UI** | Starter Questions | Auto-generated from document content |
| **UI** | Streaming | Token-by-token display via SSE |
| **UI** | Thinking Process | Real-time visibility into agent reasoning |

---

## 🛠️ Setup

### Prerequisites
- Docker & Docker Compose
- API Keys: `GROQ_API_KEY`, `GOOGLE_API_KEY`

### Quick Start (Docker)

1. **Configure API Keys**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Run**
   ```bash
   docker compose up --build
   ```

3. **Access**
   - Frontend: [http://localhost:8501](http://localhost:8501)
   - Backend API: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Upload and process a PDF |
| `/ask` | POST | Send a question (streaming response) |
| `/document-info` | GET | Get current document info + starter questions |
| `/clear-document` | DELETE | Delete current document from vector store |

---

## 🔍 Design Decisions

### Why Hybrid Search?
During testing, we discovered that pure vector search missed variations like "multi agent" vs "multi-agent". We implemented BM25 + vector fusion for robust retrieval.

### Why Groq?
Gemini's free tier has aggressive rate limits. Groq offers a generous free tier with excellent latency and full tool-calling support.

### Why Single-Document Mode?
For simplicity and clarity, each upload replaces the previous document. Multi-document support with source filtering is a planned enhancement.

---

## ⚠️ Known Limitations

### Structured Table Data
Vector embeddings are fundamentally designed for semantic similarity, not exact string matching. When a PDF contains tables with **very similar entries** (e.g., model numbers like "RD EDGEPRO 210E TAI" vs "RD EDGEPRO 210D TAF"), accuracy varies by model size.

**Recommendation:** Use **🧠 Smart (70B)** model for table queries - it handles similar entries much better than the 8B model.

**Works well for:**
- Prose and paragraph text
- Distinct categorical data
- General document Q&A
- Table queries (with 70B model)

**May have accuracy issues with:**
- Tables with nearly-identical row identifiers (on 8B model)
- Spec sheets with incremental model numbers (on 8B model)

**Production solution:** Store tabular data in a structured database (SQLite/PostgreSQL) and use exact-match queries for table lookups.

---

## 🧪 Testing

```bash
# Run backend tests
pytest tests/

# Manual API test (after docker is running)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello", "chat_history": []}'
```

---

## 📝 Future Improvements

### Major Enhancement: LiteLLM Provider Abstraction
Add a **LiteLLM** container to the Docker stack to unify all LLM providers behind a single OpenAI-compatible API. This would:
- Allow switching between 100+ LLM providers (OpenAI, Anthropic, Gemini, Groq, Ollama, etc.)
- Keep the codebase using a single `openai` client
- Enable model selection dropdown in the UI
- Simplify API key management

### Other Improvements
- [ ] OCR for image-based PDF pages (using `doctr` or `pytesseract`)
- [ ] Multi-document support with source filtering
- [ ] Math/calculation tool for complex table operations
- [ ] Observability integration (LangSmith/Langfuse)
- [ ] Chat history export (JSON/Markdown)
