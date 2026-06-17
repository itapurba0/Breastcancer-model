# Breast Cancer Companion — Agent Guide

## Project overview

Breast cancer image classifier (3 classes: benign/malignant/normal) + RAG medical chatbot. Monorepo with two top-level packages.

## Directory structure

```
backend/                          # FastAPI (Python)
├── api.py                        # Entrypoint: uvicorn on :8000
├── model_utils.py                # TF model loading, Grad-CAM, preprocessing
├── classification_model/         # Training, evaluation, .keras files
│   ├── train.py                  # EfficientNetB0 transfer learning
│   ├── classify.py               # CLI: python classify.py <image>
│   ├── evaluate.py               # CLI: eval on new_data/val/
│   └── prepare_data.py           # Split raw data → train/val/test
├── chatbot/
│   ├── engine.py                 # RAG: async generator, FastEmbed + Qdrant + OpenRouter
│   ├── ingest.py                 # Index PDFs from chatbot/data/ → Qdrant
│   └── scraper.py                # Scrape WHO fact sheet → chatbot/data/
├── tests/
│   └── test_predict.py           # pytest (monkeypatches proxy_predict)
└── run_predict_concurrent.py     # Load-test /predict endpoint
Frontend/                         # Vite + React + TS + shadcn/ui
├── src/
│   ├── App.tsx                   # Router + AnimatePresence page transitions
│   ├── pages/
│   │   ├── Index.tsx             # Landing: hero + 3 feature cards + disclaimer
│   │   ├── Classification.tsx    # Page header + how-it-works + ImageUploader
│   │   ├── Chatbot.tsx           # ChatInterface → POST /chat (streaming)
│   │   ├── Auth.tsx              # Two-column login/register
│   │   └── NotFound.tsx          # Glass-panel 404
│   └── components/
│       ├── layout/
│       │   ├── Header.tsx        # Skip-to-content, aria-labels, sage active state
│       │   └── HeroCanvas.tsx    # Three.js particle background (sage palette)
│       ├── classification/
│       │   └── ImageUploader.tsx # Upload → analyze → color-coded results + report
│       └── chatbot/
│           └── ChatInterface.tsx # Premium clinical chat with streaming
├── vite.config.ts                # Dev :3000, proxies /chat → :8000, /predict → :8000
└── components.json               # shadcn/ui config
```

## Commands

### Backend
```sh
# Run API server
uvicorn api:app --host 0.0.0.0 --port 8000
# or
python api.py

# Tests
pytest tests/ -v

# Train model
cd classification_model && python train.py

# Evaluate model
cd classification_model && python evaluate.py

# Classify single image
cd classification_model && python classify.py /path/to/image.png

# Prepare data splits
cd classification_model && python prepare_data.py

# Ingest PDFs into Qdrant
cd chatbot && python ingest.py

# Scrape WHO data
cd chatbot && python scraper.py

# Load test predict endpoint
python run_predict_concurrent.py --file /path/to/img.png --count 100
```

### Frontend
```sh
npm run dev        # Vite dev server on :3000
npm run build      # Production build
npm run lint       # ESLint
npm run preview    # Vite preview
```

### CI (GitHub Actions)
- Backend: installs `backend/requirements.txt` only (no test step currently)
- Frontend: `npm ci` → `npm run lint` → `npm run build`

## Key architecture notes

- **Frontend → Backend proxy**: Vite proxies `/predict` and `/chat` to `localhost:8000`; no CORS issues in dev.
- **Chat is streaming**: `ChatInterface.tsx` reads a `ReadableStream`, backend uses `async generator` with `StreamingResponse`.
- **Model loading**: `model_utils.py` suppresses TF stderr, disables CUDA (`CUDA_VISIBLE_DEVICES=""`). Falls back to `MODEL_PROXY_URL` env var if TF missing or no model file found.
- **Model candidates** (tried in order): `model_v3.keras`, `breast_classification_model.keras`, `model_best.keras`, `model_finetuned.keras`, `model_v2.keras`.
- **Class indices**: `{"benign": 0, "malignant": 1, "normal": 2}` (in `class_indices.json`).
- **Grad-CAM**: Handles both nested (transfer learning) and flat model architectures.
- **RAG stack**: FastEmbed (`BAAI/bge-small-en-v1.5`) for embeddings, Qdrant for vector store, OpenRouter (`openai/gpt-oss-120b:free`) for LLM. Tokenizer parallelism disabled.
- **Page transitions**: `App.tsx` uses `AnimatePresence` with `useLocation()` for blur-fade page transitions.
- **Dark mode locked**: Tailwind `darkMode: false` + CSS `color-scheme: light` — no system dark mode override.

## UI design system

- **Palette**: deep sage `#2D6A4F` primary, warm amber `#E8B86D` secondary, mint wash `#D8F3DC` accent, warm white `#FEFDFB` background
- **Fonts**: Plus Jakarta Sans (headings), Inter (body), Inconsolata (data values only)
- **Classification result colors**: green `#22c55e` = normal, blue `#3b82f6` = benign, red `#ef4444` = malignant
- **Component library**: shadcn/ui (9 components kept: button, toaster, sonner, tooltip, input, label, skeleton, badge, toast)
- **Print CSS**: clinical report layout at 9pt, sage accents, report-specific utilities (`.report-accent-bar`, `.report-triage`, `.report-signature`, `.report-disclaimer`)
- **Classification page**: page header + 3-step how-it-works strip + ImageUploader with circular SVG confidence gauge, color-coded triage, Grad-CAM side-by-side
- **Chatbot page**: gradient background, clean message bubbles, 2x2 suggested questions, sources panel

## Required setup

1. **Backend env**: Create `backend/chatbot/.env` with:
   ```
   QDRANT_URL=<url>
   QDRANT_API_KEY=<key>
   OPENROUTER_API_KEY=<key>
   ```
2. **Python deps**: `pip install -r backend/requirements.txt` (TensorFlow installed but may run CPU-only)
3. **Chatbot data**: Place PDFs in `backend/chatbot/data/`, run `python ingest.py`
4. **Node**: `npm install` in `Frontend/`

## ESLint / TypeScript quirks

- `no-unused-vars` is **off** for both TS and ESLint
- `strictNullChecks: false`, `noImplicitAny: false` in `tsconfig.json`
- `@/*` path alias maps to `./src/*`
- Backend classify/evaluate scripts suppress pyright with `# pyright: ignore` comments
- 4 pre-existing lint warnings (react-refresh/only-export-components in shadcn/ui files) — not our issue

## Testing quirks

- Backend test `test_predict.py` monkeypatches `model_utils.proxy_predict` to return a fake result
- No integration test requires a real model or Qdrant instance
- No snapshot tests
