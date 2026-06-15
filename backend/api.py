
from fastapi import FastAPI, File, UploadFile, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import StreamingResponse
import os
from typing import Dict,List

from pydantic import BaseModel
import uvicorn

import model_utils
from contextlib import asynccontextmanager
import traceback
from chatbot.engine import generate_rag_response

@asynccontextmanager
async def lifespan(app: FastAPI):
 
    global MODEL, IDX_TO_NAME
    MODEL, IDX_TO_NAME = model_utils.init_model()
    if MODEL is None:
        print("Model not loaded at startup (no TF/model found) — /predict will proxy to MODEL_PROXY_URL if available.")
    else:
        print("Model loaded successfully at startup.")
    yield
   

app = FastAPI(title="Backend Classifier API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
MODEL = None
IDX_TO_NAME: Dict[int, str] = {}




@app.get("/")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # basic content-type check
    if not file.content_type or file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    data = await file.read()

    
    if model_utils.tf is not None and MODEL is not None:
        try:
            x = model_utils.preprocess_image_bytes(data)
            res = model_utils.predict_with_model(MODEL, x)
            pred_idx = res["pred_idx"]
            probs = res["probs"]
            pred_name = IDX_TO_NAME.get(pred_idx, str(pred_idx))
            name_prob = {IDX_TO_NAME.get(i, str(i)): float(probs[i]) for i in range(len(probs))}
            confidence = float(probs[pred_idx]) if probs else None

            heatmap = model_utils.make_gradcam_heatmap(x, MODEL, pred_idx)
            gradcam_b64 = model_utils.generate_gradcam_base64(data, heatmap) # type: ignore
            gradcam_data_uri = f"data:image/jpeg;base64,{gradcam_b64}"
            
            return {
                "predicted": pred_name,
                "predicted_idx": pred_idx,
                "confidence": confidence,
                "probabilities": name_prob,
                "gradcam_image": gradcam_data_uri
            }
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Local prediction failed: {e}")

    # proxy path

    proxy_url = os.environ.get("MODEL_PROXY_URL", "http://127.0.0.1:8001/predict")
    try:
        resp = model_utils.proxy_predict(data, getattr(file, "filename", "upload"), file.content_type or "application/octet-stream", proxy_url)
        return resp
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded locally and proxy failed: {e}")

class MessageItem(BaseModel):
    role: str
    content: str

# 2. Tell the API to expect a LIST of those messages
class ChatRequest(BaseModel):
    messages: List[MessageItem]

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Pass the question from React to your RAG engine
       return StreamingResponse(
        generate_rag_response(request.messages), 
        media_type="text/plain"
        )
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    print("Starting FastAPI Server...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, loop="asyncio")