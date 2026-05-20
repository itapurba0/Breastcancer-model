# Provenance-Backed Medical Chatbot

## Overview

This is a **Retrieval-Augmented Generation (RAG)** based chatbot system designed specifically for a breast cancer classification application. The chatbot is grounded in trusted medical documents, provides source citations, and is designed to be safe, auditable, and transparent.

### Key Features

✅ **Provenance-First Design**: All answers are grounded in retrieved documents with citations  
✅ **Source Citations**: Every answer shows which documents were used (with relevance scores)  
✅ **Safety Mechanisms**: Detects out-of-scope queries and declines to answer outside domain  
✅ **Fallback Modes**: Gracefully handles model failures with extractive synthesis  
✅ **Lazy Loading**: Models initialize on-demand to minimize startup time  
✅ **Efficient Embeddings**: Uses sentence-transformers (all-MiniLM-L6-v2) for fast semantic search  
✅ **Lightweight LLM**: Uses Google's flan-t5-small for on-device inference  

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query (Chat API)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │  Ingestion   │ │  Embeddings  │ │   Vector DB  │
  │  (Markdown   │ │  (Sentence-  │ │   (Qdrant)   │
  │   files)     │ │  Transformers)│ │              │
  └──────────────┘ └──────────────┘ └──────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                    ┌────▼────┐
                    │ Retriever│
                    │(Top-k    │
                    │similarity│
                    │search)   │
                    └────┬────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │   Safety     │ │    LLM       │ │   Synthesis  │
  │   Check      │ │  (flan-t5)   │ │   (Citations)│
  │              │ │              │ │              │
  └──────────────┘ └──────────────┘ └──────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │   Answer with Citations      │
         │   {                          │
         │     "answer": "...",         │
         │     "sources": [...],        │
         │     "status": "success"      │
         │   }                          │
         └──────────────────────────────┘
```

---

## Components

### 1. **ingestion.py** - Document Loading & Chunking
- Loads markdown documents from project root
- Chunks text with configurable overlap (default: 400 words, 100-word overlap)
- Extracts metadata (filename, title, chunk index)
- Supports: `PROJECT_DOCUMENTATION_CODEBASE.md`, `RESEARCH_PAPER_DRAFT.md`, `DOCUMENTATION.md`

### 2. **embeddings.py** - Semantic Representation
- Initializes sentence-transformers model (lazy-loaded)
- Computes embeddings for texts
- Single-text and batch embedding support
- Dimension: 384 (for all-MiniLM-L6-v2)

### 3. **retriever.py** - Vector Store & Search
- Manages Qdrant vector database (in-memory or persistent)
- Stores embeddings and metadata
- Performs similarity search (cosine distance)
- Returns top-k results with relevance scores

### 4. **llm_wrapper.py** - Language Model Interface
- Wraps Google's flan-t5-small model
- Enforces retrieval-based prompting (no knowledge outside context)
- Detects out-of-scope queries (score < 0.3)
- Fallback extractive synthesis if LLM fails

### 5. **synthesis.py** - Orchestration & Citations
- Coordinates full RAG pipeline
- Handles query processing, retrieval, generation
- Formats answers with citations
- Manages system initialization and status

---

## API Endpoints



from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://d0130761-1f31-4765-942e-8b76419da019.eu-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NjQ2NzFiNDQtMDNjMS00ZGYzLWE1ZDgtOTlhYjNlMWUxNWRlIn0.sX2xm-x-ut0wMFsjJGT4mF9Ilwq-Ozp6JNUp7QXlkkE",
)

print(qdrant_client.get_collections())





### `POST /chat`
**Description**: Send a message to the chatbot

**Request**:
```json
{
  "message": "What is the breast cancer classifier?"
}
```

**Response** (200 OK):
```json
{
  "answer": "Based on our medical documents, the breast cancer classifier is...",
  "sources": [
    {
      "id": 1,
      "source": "PROJECT_DOCUMENTATION_CODEBASE.md",
      "title": "Project Documentation",
      "chunk_index": 5,
      "score": 0.92,
      "text_preview": "The classifier uses transfer learning with..."
    },
    {
      "id": 2,
      "source": "RESEARCH_PAPER_DRAFT.md",
      "title": "Research Paper",
      "chunk_index": 12,
      "score": 0.87,
      "text_preview": "We employed a CNN architecture based on..."
    }
  ],
  "status": "success",
  "reasoning": "Retrieved 2 relevant documents"
}
```

### `GET /chat/status`
**Description**: Check chatbot system status

**Response** (200 OK):
```json
{
  "status": "ok",
  "chatbot_initialized": true,
  "system_info": {
    "rag_initialized": true,
    "embedding_model": "all-MiniLM-L6-v2 (lazy-loaded)",
    "vector_store": "Qdrant (in-memory or persistent)",
    "llm_model": "google/flan-t5-small (lazy-loaded)",
    "description": "Provenance-backed medical chatbot with source citations"
  }
}
```

---

## Setup & Usage

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

Key new packages:
- `langchain` - LLM orchestration framework
- `sentence-transformers` - Embedding model
- `qdrant-client` - Vector database client
- `torch` - PyTorch (dependency for transformers)

### 2. Initialize the Chatbot

The RAG system initializes automatically when the FastAPI app starts:

```bash
cd backend
uvicorn api:app --reload
```

Logs will show:
```
Initializing RAG chatbot system...
✓ RAG chatbot initialized successfully
```

### 3. Test the Chatbot

**Option A: Direct Python test**
```bash
cd backend/tests
python3 test_integration_rag.py
```

**Option B: API test with curl**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the breast cancer classifier?"}'
```

**Option C: Via Frontend**
1. Start frontend: `cd Frontend && npm run dev`
2. Open http://localhost:3000
3. Go to "Chatbot" tab
4. Type a question

### 4. Configuration

**Document Ingestion**:
Edit `backend/chatbot/ingestion.py` to add/remove source documents:
```python
target_files = [
    "PROJECT_DOCUMENTATION_CODEBASE.md",
    "RESEARCH_PAPER_DRAFT.md",
    "DOCUMENTATION.md",
    # Add more .md files here
]
```

**Chunk Size**:
In `api.py` lifespan, modify:
```python
RAG_INITIALIZED = rag_synthesis.init_rag(docs_dir, chunk_size=400, overlap=100)
```

**Embedding Model**:
In `backend/chatbot/embeddings.py`:
```python
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2", ...):
```

**LLM Model**:
In `backend/chatbot/llm_wrapper.py`:
```python
def generate_with_context(..., model_name: str = "google/flan-t5-small", ...):
```

---

## Query Examples

### Works Well (In-Scope)
✅ "What are the three classification classes?"  
✅ "How does the model use transfer learning?"  
✅ "What preprocessing steps are applied?"  
✅ "Explain Grad-CAM explainability"  
✅ "What are the triage recommendations?"  

### Out-of-Scope
❌ "What is the capital of France?" → "I don't have information about that"  
❌ "How do I cook pasta?" → "I don't have information about that"  
❌ "Tell me a joke" → "I don't have information about that"  

---

## Error Handling

### Graceful Degradation

**Model Not Installed**:
- Embeddings: Falls back to no retrieval (empty context)
- LLM: Falls back to extractive synthesis (returns raw chunks)

**Query Out of Scope**:
- Returns: "I don't have sufficient information..."
- Status: `out_of_scope`
- Sources: Empty

**Network/Parsing Error**:
- Returns: "An error occurred while processing..."
- Status: `error`
- Frontend shows error message

---

## Frontend Integration

### ChatInterface.tsx

The frontend chatbot UI automatically:
1. **Sends queries** to `POST /chat` endpoint
2. **Displays answers** with markdown rendering
3. **Shows sources** in a collapsible citations panel
4. **Handles errors** gracefully with user-friendly messages

**Sources Panel Features**:
- Click "X sources cited" to expand/collapse
- Shows source filename, chunk index, and relevance score
- Shows text preview for context
- Color-coded match score (% confidence)

---

## Testing

### Unit Tests
```bash
cd backend/tests
python3 -m py_compile test_chat.py  # Syntax check
```

### Integration Test
```bash
cd backend/tests
python3 test_integration_rag.py
```

Output shows:
- Documents loaded
- Embeddings computed
- RAG system initialized
- Sample queries processed
- System status

### Manual Testing
1. Start backend + frontend
2. Open chatbot interface
3. Ask questions from "works well" list
4. Check sources panel
5. Verify citations are accurate

---

## Performance Notes

- **Cold Start**: ~5-15 seconds (models lazy-load on first query)
- **Query Latency**: ~1-3 seconds (retrieval + generation)
- **Memory Usage**: ~1-2 GB (depends on model loading)
- **Throughput**: Sequential (not batched in this version)

To optimize:
- Pre-warm models in API lifespan
- Add request batching
- Use GPU if available
- Cache embeddings

---

## Safety & Privacy

✅ **No External APIs**: All processing is local (no API calls)  
✅ **No Model Training**: Only inference, no weight updates  
✅ **No Data Persistence**: Queries not logged by default  
✅ **Medical Content Only**: Out-of-scope queries rejected  
✅ **Citation Transparency**: All sources shown  

---

## Troubleshooting

### Issue: "Chatbot system not initialized"
**Solution**: Check that documents exist in project root
```bash
ls *.md  # Should show PROJECT_DOCUMENTATION_CODEBASE.md, etc.
```

### Issue: "No relevant documents retrieved"
**Solution**: Query may be too far from document content; rephrase

### Issue: "Slow response time"
**Solution**: LLM is initializing; second query will be faster

### Issue: "Source links don't work"
**Solution**: This is expected; sources show document names and previews (not clickable)

---

## Future Enhancements

- [ ] Streaming responses for long answers
- [ ] User feedback collection ("Was this helpful?")
- [ ] Query rewriting for better retrieval
- [ ] Batch query processing
- [ ] Persistent query logging (with consent)
- [ ] Multi-turn conversation context
- [ ] Custom document upload UI
- [ ] Answer confidence scoring
- [ ] Hybrid retrieval (BM25 + semantic)
- [ ] Fine-tuned retriever model

---

## References

- **RAG**: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020)
- **Embeddings**: Reimers & Gurevych, "Sentence-BERT" (EMNLP 2019)
- **LLM**: Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (JMLR 2020)
- **Vector DB**: Qdrant documentation (https://qdrant.tech/)

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review test output: `python3 tests/test_integration_rag.py`
3. Check backend logs: Watch terminal where `uvicorn api:app` is running
4. Check browser console: F12 → Console tab

---

**Last Updated**: May 2026  
**Unique Feature**: Provenance-backed conversational AI with medical source citations
