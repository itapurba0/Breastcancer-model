# Breast Cancer Prediction & Medical Chatbot — Documentation (Restructured)

> Purpose: single, navigable reference for developers and AI agents. Contains architecture, dev flows, model details, chatbot design, security, and operational playbooks. Images/diagrams are marked with placeholders.

Table of contents
1. Executive summary
2. Project scope & motivations
3. High-level architecture (place diagram)
4. Backend
   - API overview
   - Startup / lifespan behaviour
   - /predict flow (local model vs proxy)
   - /chat flow (RAG + LLM)
   - Key files and conventions
5. Prediction model
   - Data & preprocessing (explicit constraints)
   - Model artifacts & naming
   - Training & reproducibility notes (where to find scripts)
   - Inference details and expected payloads
6. Chatbot system
   - RAG architecture (place diagram)
   - Ingestion & vector store choices
   - LangChain / LLM integration patterns
   - Prompting, provenance, and safety measures
7. Frontend (Vite React)
   - App pages: Prediction page, Chatbot page
   - Component responsibilities (prediction.tsx, previewCard.tsx, Chat UI)
   - Dev config (env, endpoints, CORS)
   - UI behavior rules (post-submit, preview lifecycle)
8. Developer workflows
   - Local dev commands (backend & frontend)
   - Tests (pytest smoke test), and running them
   - Debugging checklist and common failure modes
9. Deployment & CI recommendations
   - Env vars to set (MODEL_PROXY_URL, TF envs)
   - Suggested GitHub Actions smoke tests
   - Docker / GPU notes
10. Security, compliance & data privacy
    - PHI handling, logging, and retention guidance
    - Hallucination mitigation and user disclaimers
11. Appendix
    - Quick curl examples
    - Response JSON samples
    - File pointer cheat sheet
    - Placeholder list of diagrams to add

Placeholders for diagrams/images
- [INSERT DIAGRAM: architecture-overview.png] — high-level components and data flows
- [INSERT DIAGRAM: model-training-pipeline.png] — data, preprocessing, training, evaluation
- [INSERT DIAGRAM: rag-chat-architecture.png] — ingestion → vector store → retriever → LLM → response

Short notes on style & constraints (copy into agent rules)
- Maintain preprocessing exactly: image resize to (224,224); do not divide by 255 unless retraining.
- Avoid importing TensorFlow at module top-level; use model_utils.init_model() for controlled load.
- Model filenames must match MODEL_CANDIDATES or be added to that list.
- Frontend dev endpoint: http://localhost:8001/predict — change via env if deploying.
- Revoke preview object URLs in frontend to avoid leaks.

Next steps
- Confirm this structure, then I will generate the full, detailed documentation (≈8000 words) section-by-section into this DOCUMENTATION.md, inserting example JSON and code references and marking image slots.
- If you prefer the documentation file under `.github/` or a different filename, say which and I will place it there.
