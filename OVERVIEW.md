# Breast Cancer Companion — Overview & Improvement Roadmap

## What this project does

A dual-purpose medical AI application:

1. **Image classifier** — TensorFlow/EfficientNetB0 model classifying breast ultrasound images as benign/malignant/normal, with Grad-CAM heatmap visualization and a clinical report generator
2. **RAG medical chatbot** — Qdrant vector DB + FastEmbed (ONNX) embeddings + OpenRouter LLM, answering questions from indexed medical PDFs

**Stack:** FastAPI (Python, port 8000) / Vite + React 18 + TypeScript + Tailwind + shadcn/ui (port 3000)

---

## Critical bugs

### 1. Embedding model mismatch between ingest and retrieval

- `backend/chatbot/ingest.py` uses **SentenceTransformer** (`sentence-transformers` / PyTorch) with `BAAI/bge-small-en-v1.5` to create embeddings during PDF indexing
- `backend/chatbot/engine.py` uses **FastEmbed** (`fastembed` / ONNX) with `BAAI/bge-small-en-v1.5` to create query embeddings at retrieval time
- These are **different libraries with different internal preprocessing** — they produce incompatible vectors, making RAG retrieval effectively random

**Fix:** Use the same embedding library and model in both files. If switching to FastEmbed (as engine.py already does), update `ingest.py` to match.

### 2. Dead code in chatbot engine

`backend/chatbot/engine.py` contains ~200 lines of commented-out code across three abandoned implementations mixed with the active one. This makes the file hard to read and maintain.

**Fix:** Remove all commented-out code blocks.

### 3. Empty file

`backend/chatbot/prompts.py` is an empty file with no content.

---

## Security

- `backend/chatbot/.env` with Qdrant and OpenRouter API keys exists on disk but is only gitignored — no rotation policy, no audit trail
- No input sanitization or rate limiting on `/predict` or `/chat` endpoints
- CORS is correctly restricted to localhost origins in `api.py`

---

## Testing gaps

- **CI runs no backend tests**: `.github/workflows/ci.yml` installs dependencies but never runs `pytest`
- Only **one test file** exists (`backend/tests/test_predict.py`), which tests only the `/predict` endpoint via monkeypatch
- **No tests** for:
  - Chat/RAG endpoint (`/chat`)
  - Classification model inference
  - Grad-CAM generation
  - Frontend (no Vitest, no Playwright, no component tests)

---

## Dependency issues

- `requirements.txt` lists both `opencv-python` and `opencv-python-headless` — installing both can conflict on headless systems
- `psycopg2-binary` is listed but never imported or used anywhere
- Both `sentence-transformers` (PyTorch) and `fastembed` (ONNX) are installed when only one is needed
- No `requirements-dev.txt` or dependency pinning beyond what pip resolves

---

## Code quality

- `tsconfig.json` has `strictNullChecks: false` and `noImplicitAny: false` — defeats TypeScript's primary value
- Multiple backend scripts suppress pyright with `# pyright: ignore` rather than fixing types
- `components.json` references `tailwind.config.ts` but the actual file is `tailwind.config.cjs`
- No consistent import ordering or linting rules beyond ESLint defaults

---

## Infrastructure

- **No Dockerfile or docker-compose.yml** — every new contributor must manually install Python, Node, TensorFlow (CPU-only works but is slow)
- **No pre-commit hooks** for lint, typecheck, or format
- **Documentation sprawl**: 6+ markdown/content files plus `.doc` files with overlapping information — no single canonical source:
  - `CHATBOT_IMPLEMENTATION_SUMMARY.md`
  - `CHATBOT_README.md`
  - `Data.md`
  - `DOCUMENTATION.md`
  - `PROJECT_DOCUMENTATION_CODEBASE.md`
  - `RESEARCH_PAPER_DRAFT.md`
  - Various `.doc` and `.ppt` files in root

---

## Nice-to-haves

- Multiple model artifact variants (`model_v2.keras`, `model_v3.keras`, `model_best.keras`, `model_finetuned.keras`, `model_v1.keras`) with no clear naming convention — prune to one canonical model
- `run_predict_concurrent.py` default URL points to port 8001 but the API runs on port 8000
- Inconsistent `.gitignore` — root gitignores `backend/.env` and `backend/chatbot/.env` but `backend/.gitignore` has its own separate list
