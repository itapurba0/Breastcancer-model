# Final Project Documentation — Breast Cancer Prediction & Medical Chatbot (Codebase Summary)

Preface

This document mirrors the structure and headings of your final project report (PDF) and `DOCUMENTATION.md`, but focuses specifically on what exists in this repository: files, scripts, endpoints, models, and how to run them.

Where the PDF goes into full academic detail, this summary keeps only the key points and maps each section to the actual codebase.

---

1. Introduction

1.1 Background

The project is a breast cancer prediction and medical chatbot system. In this repo, the implemented part is mainly the image classifier and its web interface; the chatbot is scaffolded and planned.

The classifier is a Keras/TensorFlow model served via a FastAPI backend (`backend/api.py`) and consumed by a React frontend (`Frontend/`).

1.2 Motivation

- Provide a reproducible pipeline for breast image classification.
- Make it easy to run locally for demos and experiments.
- Prepare the structure for a provenance-first Medical ChatBot (RAG) without requiring it to be fully implemented yet.

1.3 Scope and objectives

Implemented in this codebase:
- Multi-class classifier (`benign`, `malignant`, `normal`) with a REST API (`/predict`).
- Frontend UI for uploading images and viewing predictions + confidence.
- Training, evaluation and data-preparation scripts under `backend/classification_model/`.

Planned but only partially present (scaffolded):
- RAG-based Medical ChatBot exposed via `/chat`, with wrappers in `backend/chat_interface.py`.

---

2. Literature review (concise)

The full PDF contains the detailed literature review. For code, the key takeaways influencing implementation are:
- Use transfer learning backbones (e.g., EfficientNet/VGG) fine-tuned on the target dataset.
- Maintain strict preprocessing consistency between training and inference.
- Treat explainability (e.g., Grad-CAM) and provenance as first-class design concerns.

These ideas appear in the training scripts and in how `model_utils.py` defines a single canonical preprocessing function.

---

3. System requirements (developer-oriented)

3.1 Software

- Python 3.10+ (backend, training).
- FastAPI + Uvicorn (`backend/api.py`).
- TensorFlow/Keras (`backend/classification_model/*.keras` models).
- `pytest` for backend tests (`backend/tests/test_predict.py`).
- Node.js + npm for frontend (`Frontend/`).

3.2 Hardware

- Development: CPU-only is enough to run the backend and small tests.
- Training: GPU recommended for faster experiments.

3.3 Security & compliance

The codebase assumes a trusted development environment. Before production:
- Add HTTPS termination, authentication and rate limiting around `/predict` and `/chat`.
- Treat uploaded images as sensitive data; avoid logging raw images.

---

4. High-level architecture and data flow

4.1 Components and interactions

- Frontend (React, under `Frontend/src/`): upload image, show prediction and confidence.
- Backend (FastAPI, `backend/api.py`): exposes `/predict` now, `/chat` later.
- Model artifacts (`backend/classification_model/*.keras`): loaded via `backend/model_utils.py`.

4.2 Predict request flow

1. Frontend posts a multipart form with `file` to `POST /predict`.
2. `api.py` reads the bytes and calls `preprocess_image_bytes` in `model_utils.py`.
3. `model_utils.predict_with_model` runs the Keras model and returns probabilities.
4. The API returns `predicted`, `predicted_idx`, `confidence`, and `probabilities`.

4.3 Chatbot request flow (planned)

`POST /chat` will receive JSON `{ "message": "...", "history": [...] }`, call RAG logic in `backend/chatbot/` (to be implemented), and return `{ "response": "...", "sources": [...] }`.

`backend/chat_interface.py` already defines helper functions that the API can call once the RAG pipeline exists.

---

5. Model discovery & initialization (`backend/model_utils.py`)

- Maintains a list of candidate filenames (e.g., `model_best.keras`, `model_v2.keras`, `breast_classification_model.keras`).
- `init_model()` finds the first existing file in `backend/classification_model/`, loads it, and reads `class_indices.json` to build an `idx_to_name` mapping.
- The FastAPI app (in `api.py`) stores the loaded model and mapping on `app.state` during startup.

If you add a new model filename, either:
- Save it under an existing candidate name, or
- Add it to the `MODEL_CANDIDATES` list.

---

6. Preprocessing & inference contract (CRITICAL)

`preprocess_image_bytes` in `model_utils.py` is the single source of truth for inference preprocessing:
- Loads raw bytes into `PIL.Image`.
- Converts to RGB.
- Resizes to `(224, 224)`.
- Converts to `float32` array and expands to batch dimension.

Training scripts should match this contract (same resize, color mode, and normalization) to avoid silent distribution shifts.

---

7. Training & evaluation guidance (developer recipe)

Training scripts:
- `backend/classification_model/train.py` — baseline pipeline using `image_dataset_from_directory`.
- `backend/classification_model/train_v2.py` / `train2.py` — alternative pipelines with stronger augmentation/oversampling and two-stage fine-tuning.

Suggested flow:
1. Prepare data with `prepare_data.py` (Section 8).
2. Run `train.py` or `train_v2.py` in a virtual environment.
3. Save the best model and its `class_indices.json`.
4. Run `evaluate.py` on `final_data/test` to obtain final metrics.

The scripts already include callbacks like `ModelCheckpoint`, `ReduceLROnPlateau`, and `EarlyStopping` or can easily be extended to do so.

---

8. Hospital recommendation module (operational mapping)

The PDF describes a policy layer that maps model outputs to triage recommendations.

In this repo, there is no separate module yet; any such logic would sit:
- Either in the backend (e.g., inside `/predict` after getting `predicted` and `confidence`).
- Or in the frontend, mapping `predicted` + `confidence` into a short, non-diagnostic message.

You can add a simple mapping dict in `api.py` or in the frontend prediction card component.

---

9. Prediction API — endpoints & examples

- `GET /` — health/status check.
- `POST /predict` — multipart form with `file` (image).

Example response (actual code behaviour):

```json
{
  "predicted": "malignant",
  "predicted_idx": 1,
  "confidence": 0.92,
  "probabilities": {
    "benign": 0.05,
    "malignant": 0.92,
    "normal": 0.03
  }
}
```

If no local model is loaded and `MODEL_PROXY_URL` is configured, the backend can forward the request and return the proxy response.

---

10. Frontend behavior, UX & accessibility

The frontend is in `Frontend/` (Vite + React + TypeScript).

Key files:
- `Frontend/src/pages/Classification.tsx` — classification page.
- `Frontend/src/components/classification/ImageUploader.tsx` — core upload component.

Behaviour implemented in code:
- Accepts only image files.
- Uses `URL.createObjectURL` to preview the image before sending.
- Sends the file to `${BACKEND_URL}/predict` and displays the returned `predicted` and `confidence` values.
- Applies different colours/icons depending on predicted class.

Accessibility basics (already or easily supported):
- Use `<label>` elements and clear button text.
- Ensure keyboard navigation works for the upload and submit actions.

10.4 Chat page (UI & UX)

The Chat page is the frontend surface for the Medical ChatBot and lives in:
- `Frontend/src/pages/Chatbot.tsx`
- `Frontend/src/components/chatbot/ChatInterface.tsx`

Layout and components:
- Message list: scrollable area showing alternating user and assistant messages. Each assistant message includes generated text and a "Sources" toggle.
- Input area: multi-line text box with a send button and keyboard shortcuts (Enter to send, Shift+Enter for newline). The input is disabled while waiting for a response to avoid duplicate submits.
- Sources panel (collapsible): shows the top-k retrieved documents used to produce each answer. Each source entry displays a short title, source id, retrieval score, and an expandable snippet with a link to the full document if available.
- Model/mode indicator: small UI element showing which LLM/embedding models are in use (e.g., `flan-t5-small | all-MiniLM-L6-v2`) and a `model_version` badge if returned by the backend.
- Feedback controls: thumbs-up / thumbs-down per assistant message to collect feedback for quality monitoring.

Interactive behaviour:
- Streaming: if the backend supports streaming (SSE or chunked HTTP), the UI renders tokens incrementally and provides a cancel button to abort an in-flight request.
- Citation expansion: clicking a citation token (e.g., `[doc2]`) highlights the corresponding source in the sources panel and scrolls it into view.
- Copy & export: allow users to copy message text or export the current conversation as a small JSON transcript.
- Clear / new conversation: a button clears local chat state and optionally starts a new session id with the backend.

Accessibility:
- Label the input and send button for screen readers.
- Use ARIA live regions to announce new assistant messages.
- Ensure focus styles and tab order cover the message list, input, send button, sources toggle, and feedback controls.

Error handling & graceful fallbacks:
- Network or backend error: show a concise error banner with a retry action.
- If the chain or LLM fails, display an explanation and (if available) a fallback extractive summary with citations.

Frontend ↔ Backend integration notes:
- Endpoint: `POST ${BACKEND_URL}/chat` with JSON body `{ message, history? }`.
- Response: `{ response, sources, model_version? }`.
- Timeouts: use a reasonable client-side timeout (e.g., 30–60s) and clear loading indicators once a response or error is received.

---

11. Chatbot & RAG detailed design

Overview: the project ships (or is designed to ship) a FastAPI-based chat router that implements a provenance-first Medical ChatBot. The router exposes `POST /chat` and uses a vector store for document embeddings, a HuggingFace embeddings model to produce vectors, and a small LLM wrapped by LangChain (or an equivalent abstraction) for text-to-text synthesis. The design prioritizes traceability: answers are either produced directly by a RetrievalQA chain using retrieved context or clearly synthesized from retrieved excerpts with inline citations.

11.1 Goals

- Provide an assistant that can answer questions about model design, dataset provenance, training choices, and 
- Always include citations when answers are derived from retrieved documentation.
- Avoid hallucinated clinical advice by constraining the model to retrieved context and using conservative fallbacks.

11.2 Components

- Ingestion: normalize documents (project docs, configuration files, training manifests), split into chunks (e.g., 200–400 tokens with overlap), and compute embeddings using a stable embedding model.
- Vector store: Qdrant (recommended for production) or FAISS (for local development) storing embeddings and metadata (ids, titles, source paths).
- Embeddings: a HuggingFace embeddings model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) to convert text into dense vectors.
- LLM: a HuggingFace or hosted LLM (e.g., `google/flan-t5-small`) wrapped in a pipeline and exposed via LangChain or a minimal custom wrapper.
- Retrieval chain: a RetrievalQA-style chain that takes a user question, retrieves relevant documents from the vector store, and generates an answer using only the retrieved context.

11.3 Core wiring in the backend

- `get_qa_chain()`: initializes and caches a RetrievalQA chain wired to:
  - a `HuggingFaceEmbeddings` (or equivalent) instance for vectorization,
  - a `QdrantClient` (or FAISS-based retriever) for similarity search,
  - a `HuggingFacePipeline` or hosted LLM wrapper,
  - a prompt template instructing the LLM to "answer ONLY using the provided context; if the answer is not in the context, say you do not know".
- `QdrantRetrieverAdapter`: thin adapter that converts Qdrant search results into `Document` objects, normalizing payload keys to `doc.page_content` and `doc.metadata['source']`.
- Chat endpoint in `api.py`: orchestrates calls to `get_qa_chain()`, normalizes outputs, and applies safety checks (instruction-echo detection, length limits, etc.).

11.4 Answer synthesis and layered fallbacks

To reduce hallucination and provide traceability, the runtime follows a layered approach:

1. Primary flow: send the user query to the RetrievalQA chain and return its output if it appears grounded and usable.
2. Instruction-echo detection: scan the chain output for repeated system prompt fragments or obvious instruction text; treat such outputs as failures.
3. Synthesis from retrieved docs: if the primary flow fails, build a compact context from the top-k documents and call the LLM directly with a synthesis prompt that requests:
   - a one-line answer, followed by
   - 2–4 bullet points, each citing sources as `[doc1]`, `[doc2]`, etc.
4. Extractive summarization fallback: if LLM synthesis fails (e.g., timeout, OOM), compute a deterministic extractive summary from retrieved snippets and return that with citations only (no free-form generation).

11.5 API contract & configuration

- Endpoint: `POST /chat`.
- Request body: `{ "message": "...", "history": [...] }` (history optional).
- Response body: `{ "response": "...", "sources": [ { "id": "doc1", "score": 0.87 }, ... ], "model_version"?: "..." }`.

Key environment variables (planned):
- `QDRANT_URL` / `QDRANT_HOST` — Qdrant endpoint.
- `QDRANT_API_KEY` — API key for Qdrant (if required).
- `QDRANT_COLLECTION_NAME` — name of the collection, e.g., `Medical-ChatBot`.
- `HF_EMBEDDING_MODEL` — embedding model id.
- `HF_LLM_MODEL` — LLM model id.

11.6 Defensive behaviours

- Defensive normalization: handle missing fields in Qdrant payloads and log anomalies.
- Timeout and retry: bound Qdrant and LLM calls with short timeouts, retrying transient failures once with backoff.
- Instruction-echo detection: treat any response that looks like an instruction or a copy of the prompt as unusable and route to the fallback path.

11.7 Testing the chatbot

Unit tests (e.g., `test_fallback_chatbot.py`, to be added) can:
- Monkeypatch `get_qa_chain` to return fake chains that echo prompts or raise exceptions.
- Assert that the application correctly uses the synthesis and extractive fallback paths.

This design description applies to the planned implementation; the current repo already contains `backend/chat_interface.py` to wrap `POST /chat` once the RAG pieces are added.

---

13. Testing strategy and CI

Tests:
- `backend/tests/test_predict.py` — smoke test for `/predict` using a generated in-memory image.

To run tests locally:

```bash
pytest -q backend/tests/test_predict.py
```

For CI (GitHub Actions), the minimal pipeline should:
- Install backend deps.
- Run `pytest`.
- Optionally build the frontend with `npm run build`.

---

14. Performance, benchmarks & optimizations

Currently, there is no dedicated benchmark script, but you can:
- Use `run_predict_concurrent.py` to send many requests to `/predict` and observe latency.
- Add simple timing around `predict_with_model` in `model_utils.py` for p50/p95 measurements.

Potential optimizations:
- Use a smaller backbone or quantized model for faster CPU inference.
- Restrict image size at upload to reduce processing time.

---

15. Deployment, monitoring & operational concerns

Suggested deployment pattern:
- Package backend as a Docker image with the model file under `backend/classification_model/`.
- Serve frontend as a static site (e.g., from Nginx, Vercel, or a simple file server).

Monitoring ideas:
- Add basic logging around `/predict` (status codes, latency, model version).
- In production, expose Prometheus metrics for request counts and latencies.

---

16. Security, privacy 

Before any real-world use:
- Enforce HTTPS and authentication.
- Do not retain uploaded images longer than necessary; document retention policy.
- Redact sensitive data from logs.


17. Conclusion: 
 
This project delivers a pragmatic, auditable pipeline for breast‑cancer image classification 
and a provenance‑first medical chatbot. The architecture balances developer ergonomics 
(local model loading, explicit preprocessing, and a simple FastAPI entrypoint) with 
operational flexibility (support for a remote inference proxy and pluggable vector stores for 
RAG). Emphasis on a single canonical preprocessing function, manifest-driven model 
artifacts, and an async lifespan model loader reduces silent failures and supports 
reproducible inference. The chatbot design prioritizes traceability—each answer is either 
produced from retrieved context or synthesized from clearly cited excerpts—and implements 
layered fallbacks that preserve utility when generative synthesis is unavailable. Combined, 
these elements provide a solid foundation for research iteration, clinical demonstration, and 
cautious production rollout. 
Recommendations: 
 
1. Preserve the preprocessing contract:- 
 
● Never change the canonical preprocess_image_bytes behavior without 
recording a retraining plan and updating the training_manifest.yaml. Any 
normalization or resize change must be accompanied by retraining and 
validation on an independent holdout. 
2. Harden model management: 
● Version model artifacts and manifests in an artifact store (S3/GCS) and add a 
CI validation that runs a canonical inference against newly promoted models. 
Use semantic model versioning and keep class_indices.json aligned with 
training artifacts. 
3. Operationalize the chatbot carefully 
● Deploy Qdrant (or managed vector DB) with access rules, logging, and 
backup. For production scale, consider hosted embeddings/LLM providers to 
reduce latency and operational burden, but validate data sharing and privacy 
implications first. Maintain the layered fallback logic and include audit logging 
of retrieved doc IDs for every response. 
4. Extend CI and observability 
● Add linting/type checks, frontend build tests, and a model‑validation job to CI. 
Expose Prometheus metrics for request counts, inference latency, and model 
readiness; wire alerts for degraded performance. 
5. Security and privacy 
● Treat all uploads as PHI by default. Enforce TLS, authentication (API 
keys/OAuth2), rate limiting, retention policies, and encryption at rest. Redact 
PHI from logs and require opt‑in for any persistent storage of images or 
transcripts. 
6. UX and explainability 
16  
● Keep the frontend UX simple and transparent: show model version, 
confidence, Grad‑CAM overlays with caveats, and provenance panels in chat. 
Continue usability testing with clinician stakeholders before any clinical pilot. 
7. Roadmap items 
● Add active learning tooling for labeling failure cases, model ensembling or 
multi-scale inputs for robustness, and a small human‑in‑the‑loop review 
workflow for high‑risk predictions. 
8. These steps will preserve reproducibility, improve reliability, and position the project 
for incremental, safe production adoption while maintaining a strong emphasis on 
provenance and user trust. 


18 References:
    Commands: 
● python3 -m venv .venv 
● source .venv/bin/activate 
● python -m pip install --upgrade pip 
● if [ -f backend/requirements.txt ]; then pip install -r backend/requirements.txt; fi 
● pip install uvicorn fastapi pytest httpx 
● test: pytest -q backend/tests/test_predict.py 
● cd backend 
● uvicorn api:app --reload --host 0.0.0.0 --port 8001 # (press Ctrl+C to stop) 
● cd ../frontend 
● npm install 
● npm run dev 
 
Snippets: 
Full training_manifest.yaml examples and detailed API testing scenarios are available in the 
project repository appendices. 
Key References: 
1. Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via 
Gradient-based Localization." ICCV, 2017. 
2. Esteva, A., et al. "Dermatologist-level classification of skin cancer with deep neural 
networks." Nature, 2017. 
3. Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP 
Tasks." NeurIPS, 2020. 
4. Kaggle ,  Breast Cancer Image Segmentation 
---



