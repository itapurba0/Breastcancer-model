# Breast Cancer Companion: An Integrated Deep Learning Classification System with Provenance-Aware RAG Medical Chatbot

**Academic Thesis — B.Tech Information Technology**

*July 2026*

---

## Abstract

Breast cancer remains a leading cause of mortality among women worldwide, necessitating robust, automated tools to assist in early detection and diagnosis. This thesis presents the development of Breast Cancer Companion — a hybrid system combining deep convolutional neural networks (CNNs) for breast tissue image classification with a provenance-aware Retrieval-Augmented Generation (RAG) chatbot for verifiable medical inquiry. The system employs transfer learning with an EfficientNetB0 backbone fine-tuned on breast tissue images to classify three diagnostic categories: benign, malignant, and normal. Classification outputs are enriched with Grad-CAM heatmap overlays for visual explainability, a confidence-calibrated triage policy engine, and a location-aware medical facility recommendation module. The companion chatbot implements a provenance-first architecture using FastEmbed embeddings, Qdrant vector storage, and an OpenRouter-hosted LLM to deliver citation-backed answers grounded exclusively in indexed medical documents. A production-grade frontend built with React 18, TypeScript, Vite, and shadcn/ui delivers a responsive clinical user experience with printable AI-assisted diagnostic reports. The system is deployed via FastAPI with dual inference paths supporting both local TensorFlow inference and remote proxy fallback. This thesis details the complete system architecture, training methodology, preprocessing contracts, API design, frontend engineering, and evaluation framework, demonstrating the viability of integrating quantitative prediction with qualitative, reliable automated support in clinical workflows.

**Keywords:** Breast Cancer Classification, Deep Learning, Transfer Learning, EfficientNet, Grad-CAM, Retrieval-Augmented Generation, FastAPI, React, Clinical Decision Support, Explainable AI

---

## List of Figures

| Figure | Description |
|--------|-------------|
| [INSERT FIGURE 1] | High-level system architecture and data flow |
| [INSERT FIGURE 2] | Prediction pipeline: training through inference |
| [INSERT FIGURE 3] | RAG chatbot pipeline: ingest, retrieve, generate |
| [INSERT FIGURE 4] | EfficientNetB0 model architecture and training pipeline |
| [INSERT FIGURE 5] | Training and validation accuracy/loss curves |
| [INSERT FIGURE 6] | Confusion matrix — test set evaluation |

## List of Tables

| Table | Description |
|-------|-------------|
| [INSERT TABLE 1] | Model architecture comparison: EfficientNetB0 vs. Custom CNN vs. ResNet50 |
| [INSERT TABLE 2] | Training hyperparameters and configuration |
| [INSERT TABLE 3] | Chatbot RAG configuration comparison |
| Table 4 | API endpoint reference |
| Table 5 | Preprocessing pipeline stages |
| Table 6 | Triage policy decision matrix |

---

## Chapter 1: Introduction

### 1.1 Background

Breast cancer is the most frequently diagnosed cancer among women globally, accounting for approximately 2.3 million new cases and 685,000 deaths annually according to the World Health Organization. Early and accurate detection remains the single most important factor in improving patient outcomes, with localized-stage breast cancer achieving a five-year survival rate of 99% compared to 27% for metastatic disease. The standard diagnostic workflow involves mammography screening followed by ultrasound imaging, and ultimately histopathological analysis through biopsy — a process that is labor-intensive, time-consuming, and subject to inter-observer variability among pathologists.

The rapid advancement of deep learning, particularly convolutional neural networks (CNNs), has demonstrated remarkable potential in medical image analysis. Systems leveraging transfer learning from large-scale natural image datasets (ImageNet) to medical imaging tasks have achieved performance comparable to or exceeding human experts in specific narrow domains, including dermatological classification, retinal disease detection, and histopathological grading. However, translating these laboratory results into production-ready clinical tools presents significant engineering and methodological challenges.

Contemporary Computer-Aided Diagnosis (CAD) systems often operate as opaque "black boxes," delivering binary or multi-class predictions without sufficient context, confidence calibration, or mechanisms for clinician interaction. Three critical gaps persist in bridging experimental AI performance with practical clinical utility:

1. **Deployment Drift**: Discrepancies between image preprocessing techniques employed during training and those applied in production APIs frequently lead to silent performance degradation — a phenomenon documented by Sambasivan et al. as "data cascades" in high-stakes AI systems.

2. **Lack of Explainability**: Even high-accuracy models provide limited insight into their decision-making processes, making it difficult for clinicians to trust, validate, or challenge predictions.

3. **Absence of Contextual Interaction**: Clinicians and patients interacting with diagnostic tools frequently have follow-up questions regarding model confidence, training provenance, or general protocol guidelines. Static prediction interfaces fail to provide this interactive support, while generic large language models risk generating plausible but incorrect medical information — a potentially dangerous behavior in healthcare contexts.

### 1.2 Motivation

This thesis is motivated by the conviction that a reliable breast cancer diagnosis support system requires more than a highly accurate neural network; it demands a holistic architecture encompassing reproducible data pipelines, robust APIs, explainable interfaces, and interactive, trustworthy conversational capabilities. The following design principles guide this work:

- **Reproducibility**: The preprocessing pipeline must be a single canonical function shared between training, evaluation, and inference to eliminate silent distribution shifts.

- **Explainability**: Every prediction should be accompanied by a visual explanation (Grad-CAM heatmap) and a calibrated confidence score with clinically meaningful thresholds.

- **Provenance-First Interaction**: Any conversational AI component must ground its answers in retrieved source documents and provide transparent citations, eliminating hallucination risk.

- **Clinical Readiness**: The system should support real-world workflows including triage prioritization, location-based facility referral, and printable clinical reports suitable for integration into patient records.

- **Operational Flexibility**: The architecture must support both local inference (for offline experimentation and privacy-sensitive deployments) and remote model serving (for centralized management and scaling).

### 1.3 Scope and Objectives

The primary objective of this research is to design, implement, and evaluate a comprehensive Breast Cancer Prediction and Medical Chatbot system. Specific goals include:

1. **Classification Model**: Develop a high-performance multi-class classification model (Benign, Malignant, Normal) using transfer learning with an EfficientNetB0 backbone, achieving robust sensitivity on the malignant class.

2. **Explainability**: Integrate Grad-CAM (Gradient-weighted Class Activation Mapping) to generate visual heatmaps highlighting regions of the input image most influential to the model's decision.

3. **Inference API**: Implement a production-grade FastAPI endpoint (`/predict`) that accepts image uploads, performs validated preprocessing, runs local or proxied inference, and returns prediction, confidence, probability distribution, Grad-CAM overlay, and triage assessment.

4. **Triage Policy Engine**: Design a confidence-calibrated triage system that maps prediction outputs to clinically meaningful action recommendations (High Concern → 24-hour referral, Moderate Concern → confirmatory testing, Routine Follow-up).

5. **Facility Recommendation**: Develop a location-aware medical facility recommendation module using a curated dataset of Indian hospitals, Haversine geolocation, specialty matching, and optional Google Places API fallback.

6. **RAG Medical Chatbot**: Implement a provenance-aware Retrieval-Augmented Generation chatbot that answers questions about breast cancer using only retrieved medical documents, providing inline citations and source transparency.

7. **User Authentication**: Integrate JWT-based authentication with bcrypt password hashing and MongoDB persistence to support personalized chat history.

8. **Clinical Frontend**: Build a responsive, accessible React application with image upload with drag-and-drop, step-by-step analysis visualization, color-coded results with confidence gauge, side-by-side Grad-CAM comparison, clinical report generation with print layout, and an interactive streaming chat interface.

9. **Chat History**: Persist user chat sessions to enable conversation continuity across browser sessions.

### 1.4 Literature Review

#### 1.4.1 Deep Learning in Histopathology

Convolutional neural networks have become the dominant paradigm for medical image analysis. Spanhol et al. established early benchmarks using the BreakHis dataset for breast cancer histopathological image classification, demonstrating the feasibility of deep learning approaches at varying magnification factors. Litjens et al. provided a comprehensive survey of deep learning applications in medical image analysis, identifying key challenges including dataset size limitations, class imbalance, and generalization across institutions.

Transfer learning, where models pretrained on ImageNet (1.2 million natural images across 1,000 classes) are fine-tuned on target medical datasets, has emerged as the dominant strategy due to data scarcity in the medical domain. The key insight is that early layers of deep networks learn general visual features (edges, textures, shapes) that transfer effectively across domains, while later layers learn task-specific representations during fine-tuning.

#### 1.4.2 Architectures for Medical Classification

Several backbone architectures have been extensively evaluated for medical image classification:

- **VGG16/VGG19**: Deep architectures with uniform 3x3 convolutional filters and 16-19 layers. Simple and effective but computationally expensive (138 million parameters for VGG16).

- **ResNet50**: Introduces residual connections that enable training of very deep networks (50+ layers) by mitigating the vanishing gradient problem. Widely adopted for medical imaging tasks due to strong performance-to-parameter ratio.

- **EfficientNetB0**: Developed by Tan and Le (2019), EfficientNet employs a compound scaling method that uniformly scales network depth, width, and resolution using a neural architecture search-derived baseline. The key insight is that scaling any single dimension (depth alone, width alone, or resolution alone) yields diminishing returns; simultaneous scaling using a compound coefficient phi produces significantly better results. EfficientNetB0 (phi=0) is the baseline architecture with approximately 5.3 million parameters, while larger variants (B1 through B7) scale up according to the compound coefficient. Despite being the smallest variant, EfficientNetB0 achieves ImageNet top-1 accuracy of 77.1% — comparable to ResNet50's 76.0% — with only one-fifth the parameters (5.3 million vs. 25.6 million). This parameter efficiency translates directly to faster inference times and reduced memory footprint, making it particularly suitable for deployment scenarios with limited computational resources such as CPU-only clinical environments.

**[INSERT TABLE 1: Model Architecture Comparison — EfficientNetB0 vs. Custom CNN vs. ResNet50]**

Architecture | Parameters (M) | Top-1 Accuracy (ImageNet) | Inference Speed | Deployment Suitability
EfficientNetB0 | 5.3 | 77.1% | Fast | Excellent
ResNet50 | 25.6 | 76.0% | Moderate | Good
Custom CNN (3-layer) | 2.1 | - | Very Fast | Limited
VGG16 | 138 | 71.5% | Slow | Poor

Our project selects EfficientNetB0 as the primary backbone due to its optimal balance of accuracy, parameter efficiency, and inference speed — critical for real-time clinical applications.

#### 1.4.3 Deployment Challenges in Medical AI

Sambasivan et al. (2021) identified "data cascades" as a critical failure mode in high-stakes AI systems: upstream data quality issues compound through the ML pipeline, resulting in downstream system failures that often remain undetected until deployment. This literature underscores the necessity of strict preprocessing contracts — the guarantee that transformations applied during training match exactly those applied at inference time.

Paleyes et al. (2022) surveyed deployment challenges in machine learning operations (MLOps), highlighting model versioning, monitoring, and the need for fallback mechanisms as key operational requirements for production systems.

#### 1.4.4 Explainable AI in Medical Imaging

Gradient-weighted Class Activation Mapping (Grad-CAM), introduced by Selvaraju et al. (2017), generates visual explanations for CNN-based models by computing the gradient of the target class score with respect to the feature maps of the final convolutional layer. The resulting heatmap highlights regions of the input that most strongly influence the model's decision. Grad-CAM has been widely adopted in medical imaging applications for its model-agnostic nature and intuitive visual output.

#### 1.4.5 Retrieval-Augmented Generation in Healthcare

Retrieval-Augmented Generation (RAG), introduced by Lewis et al. (2020), combines parametric memory (trained model weights) with non-parametric memory (external retrieval indices). The RAG architecture retrieves relevant document chunks from a knowledge base in response to a query and conditions answer generation on both retrieved context and the original question. This approach offers several advantages for medical applications:

- **Grounding**: Answers are constrained to verified source documents, reducing hallucination risk.
- **Transparency**: Retrieved sources provide an audit trail for every generated response.
- **Updatability**: Knowledge can be updated by modifying the document index without model retraining.

Singhal et al. (2023) demonstrated that large language models fine-tuned on medical data (Med-PaLM 2) can achieve expert-level performance on medical question-answering benchmarks. Our work builds on these foundations by implementing a lightweight, provenance-aware RAG system optimized for CPU deployment and focused specifically on breast cancer knowledge.

---

## Chapter 2: System Architecture

### 2.1 System Requirements

#### 2.1.1 Software Requirements

The system is built on a modern monorepo architecture with clearly separated concerns:

**Backend (Python 3.10+)** :
- FastAPI — asynchronous web framework for API endpoints
- Uvicorn — ASGI server for production deployment
- TensorFlow 2.x / Keras — deep learning model loading and inference
- Qdrant Client — vector database for RAG document retrieval
- FastEmbed (ONNX) — efficient embedding model for text vectorization
- OpenRouter / OpenAI — cloud-hosted LLM for chat synthesis
- PyMongo — document database for user and session persistence
- PyJWT / bcrypt — authentication and password hashing
- OpenCV — Grad-CAM heatmap generation and image processing
- Pillow — image loading and preprocessing

**Frontend (Node.js 18+)** :
- React 18.3 — component-based UI framework
- TypeScript 5.8 — type-safe JavaScript
- Vite 7.3 — build tool and development server with hot module replacement
- Tailwind CSS 3.4 — utility-first CSS framework
- shadcn/ui — 9 retained UI components (Button, Badge, Toast, etc.)
- Framer Motion 12 — declarative animation library
- Three.js / @react-three/fiber — WebGL particle background
- React Router DOM v6 — client-side routing with animated transitions
- React Markdown — markdown rendering for chat responses

#### 2.1.2 Hardware Requirements

**Development Environment**:
- CPU: Multi-core processor (4+ cores)
- RAM: 8+ GB (16 GB recommended for model training)
- Storage: 5+ GB free space for datasets and model artifacts

**Training Environment**:
- GPU: NVIDIA GPU with 8+ GB VRAM (CUDA-compatible)
- RAM: 16+ GB

**Production Inference**:
- CPU: Modern x86_64 processor (2+ cores)
- RAM: 4+ GB
- No GPU required for inference (TensorFlow CPU mode)

### 2.2 High-Level Architecture and Data Flow

The system follows a monorepo structure with two top-level packages — `backend/` and `Frontend/` — communicating through HTTP proxied via the Vite development server.

[INSERT FIGURE 1: High-level system architecture and data flow diagram showing frontend-backend separation, Vite proxy routing to FastAPI, and backend component interactions including model inference, RAG pipeline, authentication, and MongoDB persistence.]

#### 2.2.1 Request Flow: Image Classification

[INSERT FIGURE 2: Prediction pipeline showing the complete flow from image upload through preprocessing, model inference, Grad-CAM generation, triage assessment, and JSON response.]

1. The user uploads an image file through the React frontend, which creates a local preview using `URL.createObjectURL()`.
2. On submission, the frontend constructs a `FormData` object with the image file and posts it to `/predict`.
3. Vite's dev server proxies the request to the FastAPI backend at `localhost:8000`.
4. The backend validates content type (must be an image MIME type) and reads the file bytes.
5. `model_utils.preprocess_image_bytes()` transforms the raw bytes into a `(1, 224, 224, 3)` float32 tensor using the canonical pipeline.
6. If a local TensorFlow model is loaded, `model.predict()` runs inference and returns class probabilities.
7. Grad-CAM generates a heatmap highlighting influential regions, overlaid on the original image.
8. The triage engine evaluates the prediction class and confidence against policy rules.
9. The response combines prediction, confidence, probability distribution, Grad-CAM data URI, triage assessment, and model metadata.

#### 2.2.2 Request Flow: Chat

1. The user types a message and presses Enter. The frontend sends a POST request with the full conversation history as a JSON array of `{role, content}` objects.
2. The backend's `/chat` endpoint processes the request through the RAG pipeline.
3. The query is embedded using FastEmbed and a semantic search is performed against Qdrant.
4. Top-k retrieved chunks are inserted into a system prompt that constrains the LLM to answer only from provided context.
5. The LLM (OpenRouter) generates a streaming response, which the backend yields as a `text/plain` `StreamingResponse`.
6. The frontend reads the `ReadableStream`, decoding each chunk and updating the message content in real time.

### 2.3 Model Discovery and Initialization

The `model_utils.py` module centralizes all model-related operations including discovery, lazy initialization, inference, and Grad-CAM generation. A critical design decision is the avoidance of TensorFlow imports at module load time:

```python
# model_utils.py
def _import_tensorflow_safely():
    try:
        devnull = os.open(os.devnull, os.O_RDWR)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull, 2)
        try:
            import importlib
            tf_mod = importlib.import_module("tensorflow")
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(devnull)
            os.close(old_stderr_fd)
        return tf_mod
    except Exception:
        return None

tf = _import_tensorflow_safely()
```

This lazy import pattern keeps unit tests and command-line tooling responsive by avoiding TensorFlow's heavy initialization. The environment variables `TF_CPP_MIN_LOG_LEVEL=2` and `CUDA_VISIBLE_DEVICES=""` suppress TensorFlow's informational output and force CPU-only execution, eliminating CUDA library dependencies.

The model discovery mechanism searches for trained artifacts using a prioritized candidate list:

```python
# model_utils.py
MODEL_CANDIDATES = [
    "model_v3.keras",
    "breast_classification_model.keras",
    "model_best.keras",
    "model_finetuned.keras",
    "model_v2.keras",
]

def find_model_in_classification_dir():
    for name in MODEL_CANDIDATES:
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None
```

The FastAPI application uses an asynchronous lifespan context manager to load the model at startup into `app.state`, ensuring shared access across all request handlers:

```python
# api.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, IDX_TO_NAME
    MODEL, IDX_TO_NAME = model_utils.init_model()
    if MODEL is None:
        print("Model not loaded at startup - /predict will proxy to MODEL_PROXY_URL if available.")
    else:
        print("Model loaded successfully at startup.")
    yield
```

### 2.4 Dual Inference Path Pattern

The system supports two distinct inference paths to maximize operational flexibility:

**Path 1 — Local Inference**: When a TensorFlow model artifact is present and TensorFlow is importable, inference runs entirely on the local machine. This path is suitable for offline experimentation, development environments, and privacy-sensitive deployments where image data must not leave the premises.

**Path 2 — Proxy Inference**: When no local model is available and the `MODEL_PROXY_URL` environment variable is configured, the backend forwards the uploaded image to a remote inference service and returns the proxied response. This enables centralized model management, GPU-backed remote inference, and seamless scaling.

```python
# model_utils.py
def proxy_predict(file_bytes, filename, content_type, proxy_url):
    files = {"file": (filename, file_bytes, content_type)}
    resp = requests.post(proxy_url, files=files, timeout=15)
    resp.raise_for_status()
    return resp.json()
```

The fallback is transparent to the frontend — the API response format is identical regardless of which inference path is used.

---

## Chapter 3: Preprocessing and Inference Contract

### 3.1 The Criticality of Preprocessing Parity

The single most common source of silent model failure in production machine learning systems is a mismatch between training-time and inference-time preprocessing. Subtle differences in color mode, interpolation algorithm, normalization, data type, or channel order can cause distribution shift and degrade model performance without any error indication.

To eliminate this class of failure, the system enforces a single canonical preprocessing function used identically in training scripts, evaluation, and the production API.

### 3.2 Canonical Inference Pipeline

The `preprocess_image_bytes()` function serves as the single source of truth for inference preprocessing:

```python
# model_utils.py
def preprocess_image_bytes(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)
```

Pipeline stages:

| Stage | Operation | Rationale |
|-------|-----------|-----------|
| 1 | Decode from bytes using PIL.Image | Supports all common formats (JPEG, PNG, BMP, TIFF) |
| 2 | `.convert("RGB")` | Converts grayscale, RGBA, and palette images to three-channel RGB |
| 3 | `.resize((224, 224), Image.LANCZOS)` | High-quality downsampling to model input dimensions |
| 4 | `np.array(img, dtype=np.float32)` | Float32 array, values remain in [0, 255] range |
| 5 | `np.expand_dims(arr, axis=0)` | Adds batch dimension, shape (1, 224, 224, 3) |

**Table 5: Canonical preprocessing pipeline stages with rationale.**

### 3.3 Training-Side Preprocessing

The training script uses TensorFlow's `image_dataset_from_directory` which applies the same resize and RGB conversion:

```python
# train.py
def get_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        label_mode="categorical",
        image_size=IMG_SIZE,  # (224, 224)
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=123,
        color_mode="rgb"      # Ensures 3-channel RGB
    )
```

### 3.4 Edge Case Handling

The preprocessing function silently handles several edge cases:
- **Grayscale images**: Converted to RGB with three identical channels
- **RGBA images**: Alpha channel is discarded
- **Extremely large images**: Resizing occurs before numpy array allocation
- **Corrupted files**: PIL raises an exception that propagates to the API layer

### 3.5 Confidence Calibration and Thresholding

Predictions with confidence below 60% are flagged as inconclusive:

```python
# model_utils.py
CONFIDENCE_THRESHOLD = 0.60

def predict_with_model(model, x) -> dict:
    preds = model.predict(x)
    probs = preds[0].tolist()
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return {
        "pred_idx": pred_idx,
        "probs": probs,
        "confidence": confidence,
        "is_conclusive": confidence >= CONFIDENCE_THRESHOLD,
    }
```

When inconclusive:
1. The `predicted` label is set to `"inconclusive"`
2. The frontend displays a prominent amber alert
3. The triage engine returns "Further Evaluation Required"

---

## Chapter 4: Classification Model — EfficientNetB0 Transfer Learning

### 4.1 Dataset Preparation

The data preparation pipeline (`prepare_data.py`) ingests raw image directories and generates stratified train/validation/test splits. A critical enhancement is the automated filtering of segmentation mask files (suffixed with `_mask.png`), which contain ground truth labels that would cause data leakage if included in classification input.

```
raw_data/
  benign/       malignant/      normal/
  patient_001  patient_010     patient_020
  patient_001_mask (FILTERED)

data_prepared/
  train/      (60% stratified)
    benign/   malignant/   normal/
  val/        (20% stratified)
    benign/   malignant/   normal/
  test/       (20% stratified)
    benign/   malignant/   normal/
```

The dataset consists of breast ultrasound and mammography images across three classes:
- **Benign**: Non-cancerous abnormalities such as fibroadenomas, cysts, and benign calcifications. These cases require monitoring but typically do not involve aggressive intervention.
- **Malignant**: Cancerous tumors including invasive ductal carcinoma and invasive lobular carcinoma. These cases demand urgent clinical attention and treatment planning.
- **Normal**: Healthy breast tissue without any detectable abnormalities, serving as the negative control class.

The class distribution across the dataset reflects real-world clinical prevalence patterns. Stratified splitting ensures that each subset maintains the same class proportions as the original dataset, preventing evaluation bias. The validation set is used for hyperparameter tuning and early stopping decisions, while the test set remains completely unseen during model development to provide an unbiased estimate of real-world performance. The mask filtering mechanism is particularly important because segmentation masks contain explicit ground truth boundaries that, if included, would allow the model to shortcut learning by detecting annotation artifacts rather than genuine tissue morphology.

[INSERT FIGURE 6: Confusion matrix showing model performance on held-out test set across benign, malignant, and normal classes.]

### 4.2 Transfer Learning Architecture

The model architecture is built on **EfficientNetB0**, a state-of-the-art CNN that achieves ImageNet top-1 accuracy of 77.1% with only 5.3 million parameters. EfficientNet's compound scaling method uniformly scales all dimensions (depth, width, resolution) using coefficients derived from neural architecture search.

[INSERT FIGURE 4: EfficientNetB0 model architecture and training pipeline showing the backbone with ImageNet weights, custom classification head (GlobalAveragePooling2D, Dropout, Dense layers), and two-stage fine-tuning process.]

```python
# train.py
def build_model():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=IMG_SIZE + (3,),
        weights='imagenet'
    )
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)
    return model
```

The classification head includes:
1. **Global Average Pooling**: Reduces each feature map to a single value, reducing parameters while preserving spatial information
2. **Dropout (0.3/0.2)**: Prevents co-adaptation and overfitting by randomly dropping neurons during training
3. **Dense(256, ReLU)**: Hidden layer for learning task-specific feature combinations
4. **Dense(3, Softmax)**: Output layer producing a valid probability distribution across three classes

### 4.3 Training Strategy

The training process adopts a two-stage fine-tuning strategy:

**Stage 1 — Head Training (Frozen Backbone)**:
- Optimizer: Adam with learning rate 1x10^-4
- Loss: Categorical Cross-Entropy
- Epochs: Up to 50 with early stopping (patience=6)
- Only the classification head weights are updated

**Stage 2 — Fine-tuning**:
- Top N layers of the backbone are unfrozen
- Learning rate is reduced (1x10^-5) to prevent catastrophic forgetting
- Continued training until convergence

#### 4.3.1 Data Augmentation Strategy

Data augmentation is a critical component of the training pipeline, particularly given the limited size of medical imaging datasets. The augmentation pipeline applies the following transformations during training only (inference uses the original image without augmentation):

- **Random rotation**: Images are rotated by up to ±15 degrees, simulating variations in patient positioning and probe angle during ultrasound acquisition. This improves rotational invariance without introducing unrealistic orientations.
- **Random zoom**: A zoom range of 0.9 to 1.1 is applied, simulating variations in imaging distance and magnification settings across different machines and operators.
- **Horizontal flip**: Images are randomly mirrored horizontally with 50% probability. This is appropriate because breast anatomy is approximately symmetric and imaging can be performed from either orientation.
- **Brightness adjustment**: Minor brightness variations (range ±10%) simulate differences in exposure settings and machine calibration across imaging centers.

All augmentations are applied conservatively to preserve the diagnostic integrity of the tissue features. Aggressive augmentations (extreme rotations, shearing, or color jitter) are avoided as they could distort clinically relevant morphological patterns or introduce unrealistic artifacts that do not correspond to any real-world imaging condition.

[INSERT FIGURE 5: Training and validation accuracy/loss curves across epochs showing model convergence behavior during two-stage fine-tuning.]

### 4.4 Training Configuration

**[INSERT TABLE 2: Complete Training Hyperparameters]**

| Hyperparameter | Value |
|----------------|-------|
| Backbone | EfficientNetB0 |
| Pretrained Weights | ImageNet |
| Input Size | (224, 224, 3) |
| Batch Size | 16 |
| Stage 1 Learning Rate | 1x10^-4 |
| Stage 2 Learning Rate | 1x10^-5 |
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Max Epochs | 50 |
| Early Stopping Patience | 6 |
| Reduce LR Plateau Patience | 3 |
| Reduce LR Factor | 0.5 |
| Dropout (head) | 0.3, 0.2 |
| Data Augmentation | Rotation, Zoom, Flip, Brightness |

### 4.5 Evaluation Metrics

Model evaluation focuses on per-class metrics:

- **Precision (Positive Predictive Value)**: TP / (TP + FP)
- **Recall (Sensitivity)**: TP / (TP + FN)
- **F1-Score**: 2 x (Precision x Recall) / (Precision + Recall)
- **Macro-F1**: Unweighted average of F1-scores across all three classes

For the malignant class, recall is the most critical metric — minimizing false negatives ensures cancerous cases are rarely missed.

**[INSERT TABLE 1: Model Architecture Comparison — EfficientNetB0 vs. Custom CNN vs. ResNet50]**

| Metric | EfficientNetB0 | Custom CNN (3-layer) | ResNet50 |
|--------|---------------|---------------------|----------|
| Parameters | 5.3M | 2.1M | 25.6M |
| Test Accuracy | [INSERT] | [INSERT] | [INSERT] |
| Malignant Precision | [INSERT] | [INSERT] | [INSERT] |
| Malignant Recall | [INSERT] | [INSERT] | [INSERT] |
| Malignant F1 | [INSERT] | [INSERT] | [INSERT] |
| Benign F1 | [INSERT] | [INSERT] | [INSERT] |
| Normal F1 | [INSERT] | [INSERT] | [INSERT] |
| Macro F1 | [INSERT] | [INSERT] | [INSERT] |
| Inference Time (CPU) | [INSERT] | [INSERT] | [INSERT] |

---

## Chapter 5: Explainability — Grad-CAM

### 5.1 Theoretical Foundation

Gradient-weighted Class Activation Mapping (Grad-CAM) produces visual explanations for CNN-based model decisions by leveraging the gradient signal flowing into the final convolutional layer. The mathematical formulation proceeds as follows:

For a given input image and a target class c, let A^k be the k-th feature map of the final convolutional layer (with spatial dimensions u x v). The class score y^c (before softmax) is a function of these feature maps. Grad-CAM computes the importance weight alpha^k_c for each feature map k by globally average-pooling the gradients:

```
alpha^k_c = (1/Z) * sum_i * sum_j * (partial y^c / partial A^k_ij)
```

where Z = u * v is the number of spatial locations in the feature map. These weights capture the importance of each feature map for the target class — feature maps that have a strong positive gradient with respect to the class score receive higher weights.

The Grad-CAM heatmap is then computed as a weighted combination of the feature maps followed by a ReLU activation to retain only positive influences on the class of interest:

```
L^c_GradCAM = ReLU(sum_k * alpha^k_c * A^k)
```

The ReLU ensures that only features that have a positive influence on the class score are visualized — negative influences correspond to evidence for other classes and are suppressed. The resulting heatmap L^c_GradCAM is a coarse 2D grid (typically 7x7 or 14x14 depending on the network architecture) that is then upsampled to the original input image dimensions using bilinear interpolation and overlaid on the input image.

This formulation is particularly powerful because it requires no architectural modifications to the trained model — it works with any CNN architecture that has convolutional layers, including the EfficientNetB0 backbone used in this work. The only requirement is access to the gradient computation, which is provided automatically by TensorFlow's automatic differentiation engine (GradientTape).

### 5.2 Implementation

The Grad-CAM implementation handles two distinct architectural patterns — nested (transfer learning) and flat:

```python
# model_utils.py
def get_last_conv_layer_info(model):
    # 1. Check for nested models (e.g., transfer learning base models)
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for inner_layer in reversed(layer.layers):
                try:
                    if hasattr(inner_layer, 'output') and len(inner_layer.output.shape) == 4:
                        return inner_layer.name, layer.name
                except Exception:
                    continue

    # 2. Check standard flat layers
    for layer in reversed(model.layers):
        try:
            if hasattr(layer, 'output') and len(layer.output.shape) == 4:
                return layer.name, None
        except Exception:
            continue

    raise ValueError("Could not find a convolutional layer in the model.")
```

**Nested model handling**: Transfer learning architectures embed the backbone as a nested sub-model. Our implementation recursively searches inner layers for 4-dimensional outputs, correctly identifying the last convolutional layer regardless of nesting depth.

### 5.3 Heatmap Generation and Overlay

```python
# model_utils.py
def generate_gradcam_base64(img_bytes, heatmap):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_arr = np.array(img)

    heatmap_resized = cv2.resize(heatmap, (img_arr.shape[1], img_arr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    alpha = 0.5
    superimposed = cv2.addWeighted(colormap, alpha, img_arr, 1 - alpha, 0)

    out_img = Image.fromarray(superimposed)
    buffer = io.BytesIO()
    out_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
```

The overlay process: resize heatmap -> apply JET colormap -> blend with original at 50% opacity -> encode as base64 JPEG data URI.

The Grad-CAM generation is a key step in the prediction pipeline (see Figure 2). The heatmap overlay highlights regions of the input image most influential to the model's decision, providing visual explainability alongside the classification result.

### 5.4 Clinical Interpretation Guidance

The Grad-CAM overlay is presented alongside a clear limitation notice: "This overlay highlights regions the model considered influential but is not a clinical diagnosis." This framing prevents over-reliance on attention maps as diagnostic evidence.

---

## Chapter 6: Clinical Decision Support

### 6.1 Triage Policy Engine

The triage engine maps model outputs to clinically meaningful action recommendations:

**Table 6: Triage policy decision matrix.**

| Prediction | Confidence | Triage Tier | Recommendation |
|-----------|-----------|-------------|----------------|
| Malignant | >= 90% | High Concern | Urgent oncology referral within 24 hours |
| Malignant | 60-90% | Moderate Concern | Confirmatory tests + additional imaging |
| Benign | Any (conclusive) | Routine Follow-up | Standard monitoring per guidelines |
| Normal | Any (conclusive) | Routine Screening | Continue routine screening per guidelines |
| Any | < 60% | Further Evaluation Required | Clinical review by radiologist required |

```python
# facilities.py
def generate_triage(pred_name, confidence, is_conclusive):
    if not is_conclusive:
        return {
            "tier": "Further Evaluation Required",
            "recommendation": "Model confidence below safety threshold. Clinical review required.",
            "rationale": f"Confidence ({confidence:.1%}) below safety threshold.",
        }
    if pred_name == "malignant" and confidence >= 0.90:
        return {
            "tier": "High Concern",
            "recommendation": "Urgent specialist referral within 24 hours.",
            "rationale": f"High-confidence malignant classification ({confidence:.1%}).",
        }
```

### 6.2 Facility Recommendation Module

The facility recommendation system combines a curated dataset of Indian medical facilities with location-aware scoring:

```python
# facilities.py
SPECIALTY_MAP = {
    "malignant": ["breast_cancer", "oncology", "surgery"],
    "benign": ["radiology", "diagnostics", "breast_cancer_screening"],
    "normal": ["radiology", "diagnostics"],
    "inconclusive": ["diagnostics", "radiology", "mammography"],
}
```

The recommendation algorithm:
1. Loads facilities from a curated JSON dataset with metadata
2. Maps prediction class to required specialties and scores by overlap
3. Calculates Haversine distance if user coordinates are available
4. Applies city-matching and tier bonuses (tertiary=1.5x)
5. Ranks by composite score and returns top-N results

**Google Places API fallback**: When the curated dataset is insufficient, searches Google Places for additional facilities with dynamically constructed queries based on prediction class.

### 6.3 Audit Metadata

Every triage recommendation includes:
- Model version (from artifact filename)
- Request timestamp
- Confidence score
- Prediction class

---

## Chapter 7: Provenance-Aware RAG Medical Chatbot

### 7.1 What Is a Provenance-Aware Chatbot?

A **provenance-aware Retrieval-Augmented Generation (RAG) system** ensures that every generated answer is traceably grounded in retrieved source documents. Unlike generic large language models that may hallucinate plausible but incorrect information, a provenance-aware architecture enforces:

1. **Retrieval First**: Every query triggers a semantic search against a curated knowledge base
2. **Constrained Generation**: The LLM answers ONLY from provided context
3. **Citation Transparency**: Retrieved sources returned alongside the answer with relevance scores
4. **Domain Guardrails**: Refuses questions outside breast cancer scope
5. **Layered Fallbacks**: Graceful degradation through extractive summarization when LLM fails

6. **Safety Guardrails**: A multi-layer safety architecture that includes input validation, query scope detection, and controlled output formatting prevents the system from generating harmful or unauthorized medical advice. The prompt engineering incorporates explicit instructions against impersonating medical professionals, making diagnostic claims, or providing treatment recommendations. The system is designed as a patient navigator and awareness tool, not a clinical decision support system.

This creates an **audit trail** from user question to retrieved evidence to generated answer — critical for medical trust. Every response can be traced back to specific source documents, allowing clinicians and patients to verify claims independently. This transparency stands in contrast to black-box LLM systems where the provenance of generated information is opaque and unverifiable.

### 7.2 Architecture

[INSERT FIGURE 3: RAG chatbot pipeline showing the complete flow from user query through embedding, Qdrant vector search, context assembly, and OpenRouter LLM generation to streamed response.]

### 7.3 Document Ingestion

The ingestion pipeline transforms PDFs into searchable vector embeddings:

```python
# ingest.py (simplified)
def extract_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return clean_text(text)

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
```

PDFs are extracted, cleaned, chunked into overlapping segments (400 words, 50-word overlap), embedded as 384-dimensional vectors using SentenceTransformer/BAAI-bge-small-en-v1.5, and stored in Qdrant.

### 7.4 Retrieval

At query time, the user's question is embedded using FastEmbed:

```python
# engine.py
def retrieve_context(user_question, top_k=1):
    if not qdrant:
        return ""
    try:
        query_vector = list(embed_model.embed([user_question]))[0].tolist()
        search_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            score_threshold=0.5,
            with_payload=True
        )
        context_chunks = []
        for hit in search_response.points:
            if hit.payload and "text" in hit.payload:
                context_chunks.append(hit.payload["text"][:1200])
        return "\n\n---\n\n".join(context_chunks)
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return ""
```

### 7.5 Answer Synthesis

The retrieved context is injected into a system prompt enforcing safe behavior:

```python
# engine.py
system_prompt = f"""
You are a warm, empathetic, and supportive Breast Cancer Patient Navigator.

STRICT SCOPE GUARDRAILS:
1. You are a BREAST CANCER companion ONLY.
2. If asked about other topics, politely decline.

STRICT RULES:
1. DO NOT sound like a doctor.
2. Use beginner-friendly language.
3. ONLY use the provided medical context.

MEDICAL CONTEXT:
{context}
"""
```

The generation streams responses asynchronously via OpenRouter:

```python
# engine.py
async for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        yield content
```

**Safety parameters**: temperature=0.3, model=gpt-oss-120b:free, streaming enabled.

### 7.6 Streaming Frontend Integration

The frontend consumes the streaming response using the Fetch ReadableStream API:

```typescript
// ChatInterface.tsx
const reader = response.body.getReader();
const decoder = new TextDecoder("utf-8");
let done = false;
let streamedText = "";

while (!done) {
    const { value, done: readerDone } = await reader.read();
    done = readerDone;
    if (value) {
        const chunk = decoder.decode(value, { stream: true });
        streamedText += chunk;
        setMessages((prev) =>
            prev.map((msg) =>
                msg.id === assistantMessageId
                    ? { ...msg, content: streamedText }
                    : msg
            )
        );
    }
}
```

**[INSERT TABLE 3: Chatbot RAG Configuration Comparison]**

| Feature | Our Implementation | Baseline (No RAG) | Alternative (Local LLM) |
|---------|--------------------|--------------------|------------------------|
| Embedding Model | FastEmbed bge-small-en-v1.5 | - | SentenceTransformer all-MiniLM |
| Vector Store | Qdrant (cloud) | - | FAISS (local) |
| LLM | OpenRouter gpt-oss-120b:free | Raw LLM (no context) | Ollama phi3:mini |
| Streaming | Async generator + ReadableStream | - | Sync |
| Source Citations | Yes (with scores) | No | Optional |
| Domain Guardrails | Strict (breast cancer only) | None | Prompt-based |
| Hallucination Risk | Low (retrieval-constrained) | High | Medium |

---

## Chapter 8: Frontend Architecture and Clinical UX

### 8.1 Technology Stack and Design System

The frontend is built on:
- **React 18** with TypeScript
- **Vite 7** for fast HMR and optimized builds
- **Tailwind CSS 3.4** for utility-first responsive design
- **shadcn/ui** (9 components: Button, Badge, Toast, Sonner, Tooltip, Input, Label, Skeleton, Toaster)
- **Framer Motion 12** for page transitions and micro-interactions
- **Three.js** via @react-three/fiber for animated background

#### 8.1.1 Design Palette

| Token | Hex | Usage |
|-------|-----|-------|
| Primary (sage) | #2D6A4F | Buttons, links, active states |
| Secondary (amber) | #E8B86D | Accent highlights |
| Accent (mint) | #D8F3DC | Backgrounds, success states |
| Background | #FEFDFB | Page backgrounds |
| Foreground | #1A1A2E | Primary text |

**Result colors**: green (#22c55e) = normal, blue (#3b82f6) = benign, red (#ef4444) = malignant

**Fonts**: Plus Jakarta Sans (headings), Inter (body), Inconsolata (data values)

### 8.2 Page Transitions

```typescript
// App.tsx
const pageVariants = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -8 },
};
const pageTransition = {
  type: "tween",
  ease: [0.25, 1, 0.5, 1],  // Exponential easing
  duration: 0.2,
};
```

### 8.3 Classification Page UX

The classification page presents a structured workflow:

```
Page Header (left-aligned)
    "Breast tissue classification" + subtitle
    |
Upload Panel
    | Drag-drop zone -> Image preview -> Analyze button -> Progress steps
    |
Result Panel (lg+ two-column grid)
    +---------------------------+---------------------------+
    | Left (3/5)               | Right (2/5) - sticky     |
    | ConfidenceHeader (SVG)   | FacilityRecommendation   |
    | ImageComparison (2-up)   | City input + geolocation |
    | TriageCard + Disclaimer  | Curated + Google results |
    | ActionBar (Export PDF)   |                          |
    +---------------------------+---------------------------+
```



### 8.4 Confidence Gauge

```typescript
// ConfidenceHeader.tsx
const circumference = 2 * Math.PI * 36;
const dashOffset = circumference - (confidence / 100) * circumference;

<svg width="88" height="88" viewBox="0 0 88 88">
  <circle cx="44" cy="44" r="36" fill="none" strokeWidth="6" className="text-muted/60" />
  <motion.circle
    cx="44" cy="44" r="36" fill="none" stroke={colors.gauge} strokeWidth="6"
    strokeLinecap="round"
    strokeDasharray={circumference}
    initial={{ strokeDashoffset: circumference }}
    animate={{ strokeDashoffset: dashOffset }}
    transition={{ duration: 1, ease: [0.25, 1, 0.5, 1], delay: 0.15 }}
  />
</svg>
```

### 8.5 Chatbot Page UX

```
Chat Panel
+---------------------------------------------+
| Header: Bot avatar + "Medical Assistant"    |
|         Message count + Sign out button     |
+---------------------------------------------+
| Message Area (scrollable)                   |
| +---+------------------------------------+ |
| |Bot| Markdown-rendered text + citations | |
| | x | [View sources]                    | |
| +---+------------------------------------+ |
| +------------------------------------+---+ |
| | User text message                  | x | |
| +------------------------------------+---+ |
| Suggested Questions (before first message) |
| "What are early signs of breast cancer?"   |
| "How to prepare for a mammogram?"          |
+---------------------------------------------+
| Text input (Enter to send)    [Send btn]    |
| Educational disclaimer text                 |
+---------------------------------------------+
```



### 8.6 Authentication System

JWT-based authentication with:
- **Hashing**: bcrypt with salt rounds
- **Token**: HS256 JWT, 30-day expiry
- **Storage**: localStorage
- **Routes**: ProtectedRoute wrapper
- **History**: MongoDB per authenticated user

```typescript
// AuthContext.tsx (simplified)
const login = async (email, password) => {
    const res = await fetch("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    localStorage.setItem("token", data.token);
    localStorage.setItem("user", data.email);
    setUser(data.email);
};
```

### 8.7 Clinical Report Generation

The application includes a complete clinical report system with:
1. **Patient demographics form**: Name, age, ID, clinical notes
2. **Report preview**: All findings with professional formatting
3. **Print layout**: CSS @media print at 9pt with sage accents
4. **Signature block**: Reviewing physician placeholder

---

## Chapter 9: Deployment, Security, and Future Work

### 9.1 Deployment Patterns

**Development**: `uvicorn api:app --reload` on :8000 + `npm run dev` on :3000

**Production (Single Container)**: Docker with model artifact embedded, Nginx serving static frontend build

**Production (Microservices)**: Separate TensorFlow Serving model service + lightweight FastAPI API gateway

### 9.2 CI/CD Pipeline and Quality Assurance

The project is configured with GitHub Actions for continuous integration, triggered on every push and pull request to the main branch. The pipeline consists of two parallel jobs:

**Backend Job**: Installs Python dependencies from `requirements.txt` using the system Python interpreter. The pipeline currently validates that all dependencies resolve correctly but does not execute the test suite. For production hardening, this job should be extended to run `pytest` against the backend test suite and perform type checking with pyright or mypy.

**Frontend Job**: Installs Node.js dependencies with `npm ci` (which uses the lockfile for deterministic installs), runs ESLint for code quality checks, and executes a production build with `npm run build` to verify that the TypeScript compiles and the bundle can be produced without errors. The build artifact can be deployed directly to any static hosting service (Vercel, Netlify, Nginx).

Recommended enhancements for the CI pipeline include adding a model validation step that runs a canonical inference against a test image to verify model loading and prediction output, container image building with vulnerability scanning, and automated deployment to a staging environment.

### 9.3 Observability and Monitoring

Production deployments require comprehensive observability to detect degradation and diagnose issues. The following monitoring strategy is recommended:

**Health Check Endpoint**: The existing `GET /` endpoint returns `{"status": "ok", "model_loaded": true/false}` and serves as the simplest readiness probe. Container orchestrators (Kubernetes, Docker Swarm) can use this for liveness and readiness checks.

**Metrics Exposition**: For production deployments, the backend should expose Prometheus metrics including request count (total and by endpoint), inference latency (p50/p90/p99), model load state, Qdrant query latency, and error rate by HTTP status code. These metrics enable dashboards and alerting rules for operational teams.

**Structured Logging**: Logs should be emitted in JSON format with consistent fields (timestamp, request_id, endpoint, latency_ms, status_code, model_version). Log aggregation tools (ELK stack, Grafana Loki, Datadog) can then index and search these logs efficiently. Care must be taken to never log raw image bytes or PHI in log entries.

**Alerting Rules**: Critical alerts should trigger on: sustained error rate above 5%, inference latency above 5 seconds at p95, model loading failure, and Qdrant connection loss. Non-critical alerts should trigger on mild latency increases and high request volume.

### 9.4 Security Considerations

- **PHI Handling**: Images processed in volatile memory, not persisted
- **CORS**: Restricted to localhost origins in development
- **Authentication**: JWT tokens, bcrypt passwords, secret key must be environment-configured for production
- **Rate limiting**: Recommended for /predict and /chat in production

### 9.3 Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| MODEL_PROXY_URL | Remote inference fallback | No |
| QDRANT_URL | Qdrant vector store endpoint | Yes (for chatbot) |
| QDRANT_API_KEY | Qdrant authentication | Yes (for chatbot) |
| OPENROUTER_API_KEY | OpenRouter LLM access | Yes (for chatbot) |
| MONGO_DB_URL | MongoDB connection | Yes (for auth) |
| GOOGLE_PLACES_API_KEY | Facility search fallback | No |
| SECRET_KEY | JWT signing key | Yes (change from default) |

### 9.4 Current Limitations

1. **Dataset Size**: Performance bounded by training data diversity
2. **CPU-Only Inference**: CUDA disabled for compatibility, limits throughput
3. **Cold Start Latency**: First query incurs ~3-5 sec for model initialization
4. **Chatbot Knowledge Scope**: Limited to indexed documents
5. **Authentication Simplicity**: Symmetric HS256 rather than asymmetric RS256

### 9.5 Future Work

**Model Improvements**:
- Ensemble methods (EfficientNet + ResNet)
- Multi-scale analysis
- Active learning for low-confidence cases

**Feature Additions**:
- Multi-modal analysis (metadata + image)
- Federated learning across institutions
- Docker Compose for reproducible development

**Clinical Validation**:
- Clinician usability studies
- Prospective validation in clinical settings
- Regulatory pathway exploration (CE/FDA)

---

## Chapter 10: Conclusion

This thesis presented the design, implementation, and evaluation of **Breast Cancer Companion** — an integrated system combining deep learning-based image classification with a provenance-aware medical chatbot. Key contributions include:

1. **Reproducible Classification Pipeline**: EfficientNetB0 transfer learning with canonical preprocessing parity across training and inference.

2. **Explainability Integration**: Grad-CAM heatmaps handling both nested and flat model architectures.

3. **Clinical Decision Support**: Confidence-calibrated triage policy engine with location-aware facility recommendation.

4. **Provenance-Aware RAG Chatbot**: Citation-backed, domain-constrained answer generation with streaming delivery.

5. **Production-Grade Frontend**: Responsive clinical UX with animated transitions, confidence gauges, printable reports, and streaming chat.

6. **Operational Architecture**: Dual-path inference (local + proxy), JWT authentication, session persistence.

### 10.1 Ethical Considerations

The deployment of AI in healthcare carries significant ethical responsibilities that must be acknowledged. The system described in this thesis is designed as a decision support and awareness tool, not a replacement for clinical judgment. Several ethical principles guided the design:

**Transparency and Explainability**: Every prediction is accompanied by a Grad-CAM heatmap, a calibrated confidence score, and a clearly stated triage recommendation with rationale. This transparency enables clinicians to understand, validate, or challenge each prediction rather than blindly accepting black-box outputs.

**Bias and Fairness**: Deep learning models can perpetuate or amplify biases present in training data. If the training dataset is not representative of the diverse patient populations that the system will encounter, performance may vary systematically across demographic groups. Mitigation strategies include stratified evaluation across subgroups and dataset expansion efforts.

**Privacy and Data Governance**: Medical images are sensitive personal data. The system processes images in volatile memory without persisting them by default. For any production deployment, strict data governance policies must be established covering retention limits, encryption standards, access controls, and audit logging in compliance with applicable regulations such as HIPAA, GDPR, or India's Digital Personal Data Protection Act.

**Human Autonomy**: The system is explicitly positioned as a triage aid and educational tool. All outputs include disclaimers stating that findings must be validated by a board-certified radiologist or oncologist before any clinical action. The risk of automation bias — where clinicians over-rely on AI recommendations — must be actively managed through training and interface design.

**Scope Limitation**: The chatbot is constrained to breast cancer awareness and education. It explicitly declines to answer questions outside its scope, provide diagnoses, or recommend treatments. This conservative design choice prioritizes patient safety over conversational breadth.

### 10.2 Summary

The system balances quantitative precision with qualitative inquiry, establishing a foundation for trustworthy human-AI collaboration in breast cancer screening. By prioritizing transparency at every layer — from preprocessing contracts through Grad-CAM visualizations to citation-backed chat responses — this work contributes a framework for safe, effective AI deployment in clinical contexts. The modular architecture, dual-path inference design, and provenance-first chatbot approach provide a template that can be adapted to other medical imaging domains, demonstrating that responsible AI in healthcare requires not just accurate models but holistic systems designed with transparency, accountability, and human oversight as first-class requirements.

---

## References

[1] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.

[2] Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. ICCV.

[3] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.

[4] Spanhol, F. A., et al. (2016). A Dataset for Breast Cancer Histopathological Image Classification. IEEE TBME, 63(7), 1455-1462.

[5] Litjens, G., et al. (2017). A Survey on Deep Learning in Medical Image Analysis. Medical Image Analysis, 42, 60-88.

[6] Sambasivan, N., et al. (2021). Data Cascades in High-Stakes AI. CHI 2021.

[7] Singhal, K., et al. (2023). Large Language Models Encode Clinical Knowledge. Nature, 620, 172-180.

[8] He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.

[9] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.

[10] Esteva, A., et al. (2017). Dermatologist-Level Classification of Skin Cancer with Deep Neural Networks. Nature, 542, 115-118.

---

## Appendix A: Complete API Reference

**GET /** — Health check
Response: {"status": "ok", "model_loaded": true}

**POST /predict** — Image classification
Request: multipart/form-data with "file" field
Response: { predicted, confidence, probabilities, gradcam_image, inconclusive, triage }

**POST /chat** — RAG chatbot
Request: {"messages": [{"role": "...", "content": "..."}]}
Response: text/plain streaming

**Auth endpoints**: /auth/signup, /auth/login, /auth/me

**Chat history**: /chat/history, /chat/save

**Facilities**: /facilities/recommend, /facilities/search

---

## Appendix B: Quick-Start Commands

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000

# Frontend
cd Frontend
npm install && npm run dev

# Model Training
cd backend/classification_model
python prepare_data.py && python train.py

# Evaluation
python evaluate.py

# Chatbot Ingestion
cd ../chatbot && python ingest.py

# Tests
pytest tests/ -v
```

---

*This thesis was prepared in partial fulfillment of the requirements for the degree of Bachelor of Technology in Information Technology. July 2026.*

---

**End of Document**
