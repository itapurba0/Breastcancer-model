# Breast Cancer Companion — Presentation

> **Format**: Slide title | Bullet content | Speaker notes
>
> Use this markdown to create your PPT slides. Each `---` is a new slide. Content before `NOTES:` is slide content. Content after `NOTES:` is speaker script.

---

## Slide 1 — Title Slide

**Breast Cancer Companion**
A Deep Learning and RAG-Based System for Classification and Medical Inquiry

**Your Name**
Department of Information Technology
Bachelor of Technology — Final Year Project

---

## Slide 2 — Agenda

- Introduction & Motivation
- Literature Review
- System Architecture
- Model Training & Dataset
- Image Classification Pipeline
- Explainability (Grad-CAM)
- RAG Chatbot Design
- Frontend & UX
- Results & Performance
- Conclusion & Future Work

NOTES: Good morning/afternoon everyone. I'll be presenting my final year project — Breast Cancer Companion. This is a hybrid system combining deep learning for image classification with a retrieval-augmented chatbot for medical inquiry. I'll walk through the motivation, architecture, training methodology, and results over the next 10-12 minutes.

---

## Slide 3 — Background & Motivation

**The Problem**
- Breast cancer: 2.3M new cases, 685K deaths annually worldwide
- Early detection → 99% survival rate
- Radiologist shortage in developing countries

**Why This Matters**
- Automated triage can prioritize high-risk cases
- Clinicians need explainable AI, not black boxes
- Patients need reliable, accessible medical information

NOTES: Breast cancer is the most common cancer among women globally. Early detection dramatically improves outcomes, but many regions lack adequate radiologists. Existing AI systems are often black boxes — they give predictions without explanation. Our system addresses both problems: it classifies images with visual explanations (Grad-CAM), and provides a chatbot that cites verified medical sources rather than guessing.

---

## Slide 4 — Objectives

**What We Built**
1. **Image Classifier** — EfficientNetB0 trained on breast ultrasound images
   - 3 classes: benign, malignant, normal
   - Grad-CAM heatmaps for explainability

2. **RAG Medical Chatbot** — Provenance-aware Q&A system
   - Answers grounded in indexed medical PDFs
   - Every response cites source documents

3. **Clinical Decision Support**
   - Triage recommendations based on confidence
   - Location-aware hospital referrals

4. **Full-Stack Web Application**
   - FastAPI backend, React frontend

NOTES: The project has four main components. First, an image classifier using EfficientNetB0 that distinguishes between benign, malignant, and normal tissue. Second, a chatbot that uses Retrieval-Augmented Generation — meaning it retrieves relevant medical documents before answering, so every response is verifiable. Third, clinical support features like triage levels and hospital recommendations. And fourth, a complete web application tying it all together.

---

## Slide 5 — Technology Stack

| Layer | Technology |
|-------|-----------|
| **ML Framework** | TensorFlow / Keras (EfficientNetB0) |
| **Backend** | FastAPI (Python) |
| **Frontend** | React 18 + TypeScript + Vite |
| **UI Library** | shadcn/ui + Tailwind CSS |
| **Vector DB** | Qdrant |
| **Embeddings** | FastEmbed (BAAI/bge-small-en-v1.5) |
| **LLM** | OpenRouter (gpt-oss-120b:free) |
| **Auth DB** | MongoDB Atlas |
| **CI/CD** | GitHub Actions |

NOTES: Here's the technology stack. The ML model is built with TensorFlow and Keras using EfficientNetB0. The backend is FastAPI, a high-performance Python web framework. The frontend uses React with TypeScript and shadcn/ui components. For the chatbot, we use Qdrant as the vector database, FastEmbed for generating embeddings, and OpenRouter to access the LLM. MongoDB handles user authentication and chat history.

---

## Slide 6 — System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                   │
├─────────────┬──────────────────────┬────────────────────┤
│  FRONTEND   │      BACKEND         │   EXTERNAL          │
│  (React)    │     (FastAPI)        │   SERVICES          │
│             │                      │                     │
│  /predict ──▶  model_utils.py ────▶  TF/Keras Model     │
│  /chat    ──▶  engine.py      ────▶  Qdrant + OpenRouter│
│  /auth    ──▶  auth/routes.py ────▶  MongoDB            │
│  /facilities▶  facilities.py  ────▶  Curated Dataset    │
└─────────────┴──────────────────────┴────────────────────┘
```

**Data Flow (Classification):**
Upload Image → Preprocess (224×224 RGB) → EfficientNetB0 Inference → Grad-CAM → Triage → Results

**Data Flow (Chat):**
User Query → FastEmbed Search → Qdrant Retrieval → LLM Generation → Streamed Response + Citations

NOTES: Let me walk through the architecture. The frontend communicates with the FastAPI backend through four main endpoints. For predictions, the image is preprocessed to 224x224 RGB format, fed through EfficientNetB0, and the output is combined with Grad-CAM heatmaps and triage information. For the chatbot, user queries are converted to embeddings using FastEmbed, relevant documents are retrieved from Qdrant, and the LLM generates a response constrained to those documents. All responses are streamed in real-time.

---

## Slide 7 — Dataset & Preprocessing

**Dataset**: Breast Ultrasound Images (BUSI) with Ground Truth
- 3 classes: benign, malignant, normal
- Images from Kaggle (Al-Dhabyani et al., 2020)

**Preprocessing Pipeline**
- Mask filtering: exclude `_mask.png` files (data leakage prevention)
- Resize: 224×224 using OpenCV bilinear interpolation
- Color: BGR → RGB conversion
- Split: 80% train / 20% test (stratified)

**Class Balancing**
- Original dataset has class imbalance
- Each class augmented to exactly 1,000 samples
- Albumentations: HorizontalFlip, VerticalFlip, RandomRotate90

NOTES: We used the Breast Ultrasound Images dataset from Kaggle. A critical preprocessing step is filtering out mask files — these are segmentation annotations that would leak ground truth information if included in training. We resized all images to 224x224 using bilinear interpolation. Since the dataset was imbalanced (fewer malignant samples), we used Albumentations to generate augmented versions until each class had exactly 1,000 training samples. This prevents the model from developing bias toward the majority class.

---

## Slide 8 — Model Architecture

```
┌─────────────────────────────────────┐
│         EfficientNetB0              │
│  (Pre-trained on ImageNet)          │
│  All layers frozen during training  │
├─────────────────────────────────────┤
│       GlobalAveragePooling2D        │
│       (7×7×1280 → 1280 vector)      │
├─────────────────────────────────────┤
│         Dense(128, ReLU)            │
├─────────────────────────────────────┤
│         Dense(64, ReLU)             │
├─────────────────────────────────────┤
│      Dense(3, Softmax)              │
│   ┌──────┼──────┐                    │
│  Benign Malignant Normal            │
└─────────────────────────────────────┘
```

**Training Parameters:**
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Cross-entropy
- Batch Size: 32
- Epochs: 50
- Validation Split: 20%

NOTES: This slide shows the model architecture. We used EfficientNetB0 pre-trained on ImageNet as the backbone, with all layers frozen — this means we only trained the classification head. The features pass through global average pooling, then two dense layers with 128 and 64 units respectively, and finally a 3-unit softmax layer for the three classes. Training used Adam optimizer with a 0.001 learning rate, categorical cross-entropy loss, and 50 epochs with a 20% validation split.

---

## Slide 9 — Classification Results

[INSERT YOUR CONFUSION MATRIX IMAGE]

[INSERT YOUR ACCURACY/LOSS CURVES IMAGE]

| Metric | Value |
|--------|-------|
| **Accuracy** | [Your value]% |
| **Precision (Malignant)** | [Your value] |
| **Recall (Malignant)** | [Your value] |
| **F1-Score (Malignant)** | [Your value] |
| **Inference Time (CPU)** | ~850ms |

[INSERT YOUR MODEL COMPARISON TABLE — EfficientNetB0 vs Custom CNN vs ResNet50]

NOTES: Here are the results. [Talk through your actual metrics]. The model achieves strong performance on malignant classification, which is the most critical class — we want to minimize false negatives. The confusion matrix shows [describe misclassifications — likely confusion between benign and malignant]. Inference time on CPU is approximately 850ms per image, which is acceptable for an interactive web application.

---

## Slide 10 — Explainability: Grad-CAM

**What is Grad-CAM?**
- Gradient-weighted Class Activation Mapping
- Highlights regions that influenced the model's decision
- Builds clinician trust by showing "where the model is looking"

**Technical Implementation**
1. Forward pass through the model → capture last conv layer outputs
2. Backpropagate the class score gradient
3. Weight feature maps by gradient importance
4. Upsample heatmap to original image size
5. Overlay with 50% opacity on original image

**Handles both nested (transfer learning) and flat model architectures**

[INSERT SIDE-BY-SIDE: Original Image | Grad-CAM Overlay]

NOTES: One of the most important features of this system is explainability. Grad-CAM generates heatmaps showing which regions of the image most influenced the model's prediction. The implementation handles both nested transfer learning models and flat architectures — this is important because EfficientNetB0 as a backbone creates a nested model structure. The heatmap is overlaid on the original image at 50% opacity, allowing clinicians to verify the model is focusing on clinically relevant tissue rather than image artifacts or background.

---

## Slide 11 — RAG Chatbot Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│   FastEmbed     │  Embed user query (BAAI/bge-small-en-v1.5)
│   (ONNX)        │
└────────┬────────┘
         │ query vector
         ▼
┌─────────────────┐
│   Qdrant        │  Semantic search in medical knowledge base
│   Vector DB     │  Returns top-k relevant chunks with scores
└────────┬────────┘
         │ retrieved context
         ▼
┌─────────────────┐
│   System Prompt │  "Answer ONLY using the provided context"
│   Construction  │  + scope guardrails (breast cancer only)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   OpenRouter    │  GPT-based LLM generates cited answer
│   LLM           │  Temperature: 0.3 (low randomness)
└────────┬────────┘
         │ streamed response
         ▼
┌─────────────────┐
│   Streaming     │  Tokens delivered via async generator
│   Response      │  Frontend renders incrementally
└─────────────────┘
```

NOTES: The chatbot uses Retrieval-Augmented Generation. When a user asks a question, the system first converts it to a vector embedding using FastEmbed's lightweight ONNX model. This vector is used to search Qdrant for the most semantically similar document chunks. Only these retrieved chunks are provided as context to the LLM, along with a strict system prompt that says "answer ONLY using this context." The LLM runs on OpenRouter with temperature 0.3 for low randomness. The response is streamed back to the frontend token by token.

---

## Slide 12 — Provenance-Aware Design

**Why RAG instead of standard LLM?**

| Aspect | Standard LLM | Our RAG System |
|--------|-------------|----------------|
| **Knowledge Source** | Internal model weights | Retrieved medical documents |
| **Hallucination Risk** | High | Low |
| **Citations** | None | Source + chunk + score |
| **Verifiability** | Impossible | Complete traceability |
| **Scope Control** | Difficult | Easy (prompt-level) |

**Prompt Guardrails:**
- "You are a breast cancer companion ONLY"
- Rejects non-breast-cancer questions
- 8th-grade reading level
- "Do NOT sound like a doctor"

NOTES: This slide explains why we chose RAG over a standard LLM. The key advantage is provenance — every answer can be traced back to specific source documents. If you click "View sources" in the chat interface, you can see exactly which medical document each claim came from, with relevance scores. This is critical for medical applications where hallucination is unacceptable. The system also has strict scope guardrails — it will politely decline to answer questions about other cancers or unrelated medical topics.

---

## Slide 13 — Frontend Highlights

**Design System**
- Color palette: Sage (#2D6A4F), Amber (#E8B86D), Mint (#D8F3DC)
- Fonts: Plus Jakarta Sans (headings), Inter (body), Inconsolata (data)
- Dark mode locked to light theme

**Key Features**
- Drag-and-drop image upload with live preview
- Animated analysis progress (4-step)
- SVG confidence gauge with color-coded results
- Side-by-side original + Grad-CAM comparison
- Clinical triage card with tier, recommendation, rationale
- Print-ready clinical report generation
- Streaming chatbot with collapsible sources panel
- Responsive: mobile-first, two-column on desktop

**Accessibility**
- Skip-to-content navigation
- ARIA labels on all interactive elements
- prefers-reduced-motion support
- keyboard-accessible with visible focus rings

NOTES: The frontend is designed for clinical environments — clean, trustworthy, and accessible. The color palette uses sage green which evokes calm and professionalism. Results are color-coded: green for normal, blue for benign, red for malignant. The chatbot streams responses in real-time with animated typing indicators, and users can click "View sources" to see which documents were used. We also implemented accessibility features like skip-to-content navigation, ARIA labels, and reduced-motion support for users with motion sensitivity.

---

## Slide 14 — Clinical Report & Hospital Referral

**Clinical Report Generator**
1. Export button → patient demographics form
2. Generate print-ready report with:
   - Color-coded header matching prediction class
   - Patient info, scan, heatmap side-by-side
   - AI triage assessment
   - Clinical disclaimer (bold: "NOT a medical diagnosis")
3. Print / Save as PDF

**Hospital Recommendation**
- Curated dataset of 20 Indian cancer centers
- Scored by specialty match + geographic distance
- Haversine formula for distance calculation
- Falls back to Google Places API if available

| Triage Level | Condition | Action |
|-------------|-----------|--------|
| High Concern | Malignant ≥ 90% | Urgent oncology referral (24h) |
| Moderate Concern | Malignant 60-89% | Confirmatory tests |
| Routine Follow-up | Benign | Standard monitoring |
| Routine Screening | Normal | Continue screening |
| Further Eval | < 60% confidence | Clinical review needed |

NOTES: The system includes two unique features. First, a clinical report generator that produces a professional, print-ready diagnostic report with the patient's details, original scan, Grad-CAM heatmap, triage assessment, and a prominent disclaimer. This can be printed or saved as PDF for medical records. Second, a hospital recommendation module that suggests appropriate medical facilities based on the classification result. It uses a curated dataset of 20 Indian hospitals with matching by specialty and distance using the Haversine formula. For malignant cases with high confidence, it recommends urgent oncology referral.

---

## Slide 15 — User Authentication & Chat History

**Authentication Flow**
1. User signs up with email + password
2. Password hashed with bcrypt
3. JWT token issued (30-day expiry)
4. Token stored in frontend context
5. Protected routes guard against unauthorized access

**Chat Persistence**
- Conversations saved to MongoDB
- Up to 50 messages stored per user
- Loaded on page return (session continuity)
- Save triggered after each exchange

NOTES: The system includes full user authentication. Passwords are hashed using bcrypt, and JWTs with 30-day expiry are used for session management. Chat conversations are persisted to MongoDB, so users can leave and come back to their conversation history. The frontend automatically loads the last 50 messages when a logged-in user visits the chatbot page.

---

## Slide 16 — Deployment & CI

**Local Development**
```bash
# Backend
uvicorn api:app --host 0.0.0.0 --port 8000

# Frontend (Vite proxies API requests to :8000)
npm run dev
```

**CI Pipeline (GitHub Actions)**
- Push/PR to main/master triggers:
  1. Python dependency installation
  2. Node.js setup + npm ci
  3. ESLint check
  4. Production build

**Production Considerations**
- Docker containerization for backend
- Static frontend served via Vercel/Nginx
- HTTPS enforcement
- Rate limiting on /predict and /chat

NOTES: For deployment, the backend runs as a Uvicorn server and the frontend uses Vite with API proxying during development. For production, we recommend containerizing the backend with Docker and serving the static frontend build through Vercel or Nginx. The CI pipeline automatically runs linting and builds on every push. Key production concerns include HTTPS, rate limiting, and API authentication.

---

## Slide 17 — Limitations

**Dataset Limitations**
- Model performance bounded by dataset size and diversity
- Limited to breast ultrasound images (not mammograms or MRI)
- Potential demographic bias if training data lacks diversity

**Technical Limitations**
- No fine-tuning stage (all backbone layers frozen)
- No callbacks (EarlyStopping, ModelCheckpoint) during training
- ~850ms inference on CPU (not real-time)

**Chatbot Limitations**
- Knowledge limited to indexed documents only
- No real-time medical literature updates
- Not a substitute for professional clinical judgment

NOTES: Like any project, there are limitations. The model was trained on a relatively small dataset of ultrasound images, so generalization to different equipment or populations needs validation. We kept all backbone layers frozen — fine-tuning could potentially improve accuracy. The chatbot's knowledge is limited to whatever documents have been indexed; it doesn't have access to real-time medical literature. Most importantly, this is a decision support tool, not a diagnostic device.

---

## Slide 18 — Future Work

**Immediate Improvements**
- Add EarlyStopping and ModelCheckpoint callbacks
- Two-stage fine-tuning (unfreeze top backbone layers)
- Align OpenCV and PIL interpolation in preprocessing

**Medium-Term**
- Federated learning for privacy-preserving multi-institution training
- Multi-modal analysis (integrate patient age, genetics, biomarkers)
- On-premise Docker deployment for hospital IT

**Long-Term**
- Active learning loop: flag low-confidence cases for expert review + retraining
- Support for mammogram and MRI modalities
- Clinical pilot study with partner hospitals

NOTES: There are several clear paths for improvement. In the short term, we can add standard training callbacks and implement two-stage fine-tuning. Medium-term goals include federated learning so multiple hospitals can contribute to model improvement without sharing patient data, and multi-modal analysis combining imaging with patient history. Long term, we'd like to conduct clinical pilot studies and expand to other imaging modalities.

---

## Slide 19 — Conclusion

**Summary**
- Built a full-stack breast cancer decision support system
- EfficientNetB0 classifier with Grad-CAM explainability
- Provenance-aware RAG chatbot with source citations
- Clinical triage, hospital referral, and report generation

**Key Contributions**
1. Reproductible ML pipeline with preprocessing contract
2. Explainable AI through Grad-CAM visualization
3. Safe medical chatbot through retrieval-augmented generation
4. Accessible, production-ready web application

**Final Note**
This system is a research prototype — it assists clinicians but does not replace them.

NOTES: To summarize, we've built an integrated breast cancer companion system that handles both image classification and medical inquiry in a safe, explainable way. The key innovations are the preprocessing contract ensuring reproducibility, Grad-CAM for explainability, and the provenance-aware RAG chatbot that prevents hallucination by grounding answers in source documents. The complete system is deployed as a web application accessible from any modern browser.

---

## Slide 20 — Thank You

**Breast Cancer Companion**
Questions?

Contact: [Your Email]
GitHub: [Repository URL]

**References**
[1] Tan & Le, EfficientNet, ICML 2019
[2] Selvaraju et al., Grad-CAM, ICCV 2017
[3] Lewis et al., RAG, NeurIPS 2020
[4] Al-Dhabyani et al., Breast Ultrasound Dataset, Data in Brief 2020

---

*Total: 20 slides | Estimated presentation time: 10-15 minutes*
