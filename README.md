# 🤖 PDF AI Assistant

A document-based AI assistant built with **FastAPI**, **Streamlit**, **LangGraph**, and **ChromaDB**. Upload a PDF and chat with your document using advanced RAG (Retrieval-Augmented Generation).

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
| **"Glass Box" UI** | Real-time thinking process visibility |
| **Starter Questions** | Auto-generated based on document content |
| **Follow-up Suggestions** | 3 contextual questions after each answer |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│  FastAPI Backend │────▶│    ChromaDB     │
│   (Port 8501)   │     │   (Port 8000)    │     │   (Port 8001)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                        │
        │                        ▼
        │               ┌─────────────────┐
        │               │   LangGraph     │
        │               │  Agent + Tools  │
        │               └─────────────────┘
        │                        │
        ▼                        ▼
   User Chat ◀────────── Streaming Response
```

### Key Components

| Component | Technology |
|-----------|------------|
| **LLM** | Groq (`llama-3.1-8b-instant`) |
| **Embeddings** | Google Gemini (`gemini-embedding-001`) |
| **Vector Store** | ChromaDB (Docker) |
| **Agent Framework** | LangGraph with tool calling |
| **Backend** | FastAPI with streaming SSE |
| **Frontend** | Streamlit with custom CSS |

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

---

## 🔍 Design Decisions

### Why Hybrid Search?
During testing, we discovered that pure vector search missed variations like "multi agent" vs "multi-agent". We implemented BM25 + vector fusion for robust retrieval.

### Why Groq?
Gemini's free tier has aggressive rate limits. Groq offers a generous free tier with excellent latency and full tool-calling support.

### Why Single-Document Mode?
For simplicity and clarity, each upload replaces the previous document. Multi-document support with source filtering is a planned enhancement.

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

- [ ] OCR for image-based PDF pages
- [ ] Multi-document support with source filtering
- [ ] Math/calculation tool for tables
- [ ] Observability (LangSmith/Langfuse)
- [ ] Model selection UI

---

## 📄 License

MIT
