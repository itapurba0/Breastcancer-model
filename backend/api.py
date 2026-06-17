
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends # type: ignore
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import StreamingResponse
import os
import json
import math
from typing import Dict, List, Optional

from pydantic import BaseModel
import uvicorn

import model_utils
from contextlib import asynccontextmanager
import traceback
from chatbot.engine import generate_rag_response
from auth.routes import router as auth_router
from auth.deps import get_current_user
from database import sessions_collection

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
app.include_router(auth_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
MODEL = None
IDX_TO_NAME: Dict[int, str] = {}


def generate_triage(pred_name: str, confidence: float, is_conclusive: bool) -> dict | None:
    """Generate a triage assessment based on prediction class and confidence."""
    if not is_conclusive:
        return {
            "tier": "Further Evaluation Required",
            "recommendation": "Model confidence is below the 60% safety threshold. Recommend clinical review by a radiologist for definitive diagnosis.",
            "rationale": f"The model's prediction confidence ({confidence:.1%}) falls below the safety threshold, indicating uncertainty in the classification. This result should not be used for clinical decisions without professional review.",
        }

    if pred_name == "malignant":
        if confidence >= 0.90:
            return {
                "tier": "High Concern",
                "recommendation": "Urgent specialist referral recommended. Schedule oncology consultation within 24 hours and consider confirmatory biopsy.",
                "rationale": f"High-confidence malignant classification ({confidence:.1%}) with strong model certainty. Prompt specialist evaluation is advised.",
            }
        else:
            return {
                "tier": "Moderate Concern",
                "recommendation": "Confirmatory tests recommended. Additional imaging (diagnostic mammography, ultrasound) and expert review advised.",
                "rationale": f"Moderate-confidence malignant classification ({confidence:.1%}). Confirmatory testing recommended before clinical action.",
            }

    if pred_name == "benign":
        return {
            "tier": "Routine Follow-up",
            "recommendation": "Standard monitoring recommended. Follow routine screening schedule as per clinical guidelines.",
            "rationale": f"Benign classification with {confidence:.1%} confidence. No immediate intervention required but regular follow-up advised.",
        }

    return None




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
            confidence = res["confidence"]
            is_conclusive = res["is_conclusive"]
            pred_name = IDX_TO_NAME.get(pred_idx, str(pred_idx))
            name_prob = {IDX_TO_NAME.get(i, str(i)): float(probs[i]) for i in range(len(probs))}

            # If below confidence threshold, mark as inconclusive
            if not is_conclusive:
                pred_name = "inconclusive"

            heatmap = model_utils.make_gradcam_heatmap(x, MODEL, res["pred_idx"])
            gradcam_b64 = model_utils.generate_gradcam_base64(data, heatmap) # type: ignore
            gradcam_data_uri = f"data:image/jpeg;base64,{gradcam_b64}"

            triage = generate_triage(
                pred_name if pred_name != "inconclusive" else IDX_TO_NAME.get(res["pred_idx"], str(res["pred_idx"])),
                confidence,
                is_conclusive,
            )

            response = {
                "predicted": pred_name,
                "predicted_idx": pred_idx,
                "confidence": confidence,
                "probabilities": name_prob,
                "gradcam_image": gradcam_data_uri,
                "inconclusive": not is_conclusive,
            }
            if triage:
                response["triage"] = triage

            return response
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Local prediction failed: {e}")

    # proxy path

    proxy_url = os.environ.get("MODEL_PROXY_URL", "http://127.0.0.1:8000/predict")
    try:
        resp = model_utils.proxy_predict(data, getattr(file, "filename", "upload"), file.content_type or "application/octet-stream", proxy_url)
        return resp
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded locally and proxy failed: {e}")

class MessageItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[MessageItem]

class SaveChatRequest(BaseModel):
    messages: list

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
       return StreamingResponse(
        generate_rag_response(request.messages), 
        media_type="text/plain"
        )
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/chat/history")
def get_chat_history(email: str = Depends(get_current_user)):
    session = sessions_collection.find_one({"email": email})
    if not session or not session.get("messages"):
        return {"messages": []}
    return {"messages": session["messages"]}


@app.post("/chat/save")
def save_chat_history(body: SaveChatRequest, email: str = Depends(get_current_user)):
    sessions_collection.update_one(
        {"email": email},
        {"$set": {"messages": body.messages, "updated_at": __import__("datetime").datetime.utcnow()}},
        upsert=True
    )
    return {"status": "ok"}


# --- Facility Recommendation ---

FACILITIES_PATH = os.path.join(BASE_DIR, "facilities.json")


def load_facilities() -> list:
    """Load the curated facility dataset."""
    if not os.path.exists(FACILITIES_PATH):
        return []
    with open(FACILITIES_PATH, "r") as f:
        data = json.load(f)
    return data.get("facilities", [])


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    R = 6371.0  # Earth's radius in km
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Mapping from prediction to relevant specialties
SPECIALTY_MAP = {
    "malignant": ["breast_cancer", "oncology", "surgery"],
    "benign": ["radiology", "diagnostics", "breast_cancer_screening"],
    "normal": ["radiology", "diagnostics"],
    "inconclusive": ["diagnostics", "radiology", "mammography"],
}


class FacilityRecommendRequest(BaseModel):
    prediction: str
    confidence: float
    inconclusive: bool = False
    city: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    limit: int = 5


class FacilitySearchRequest(BaseModel):
    query: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    radius: int = 20000


@app.post("/facilities/recommend")
def recommend_facilities(body: FacilityRecommendRequest):
    """Recommend medical facilities based on classification result and optional location."""
    facilities = load_facilities()
    if not facilities:
        return {"recommendations": [], "source": "curated", "error": "No facility data available"}

    # Determine which specialties to match
    pred_key = "inconclusive" if body.inconclusive else body.prediction.lower()
    target_specialties = set(SPECIALTY_MAP.get(pred_key, SPECIALTY_MAP["inconclusive"]))

    # Score each facility
    scored = []
    for f in facilities:
        # Calculate specialty match score
        facility_specs = set(f.get("specialties", []))
        spec_overlap = len(target_specialties & facility_specs)
        if spec_overlap == 0:
            continue

        # Calculate distance if coordinates provided
        distance_km = None
        if body.lat is not None and body.lng is not None:
            coords = f.get("coordinates", {})
            if "lat" in coords and "lng" in coords:
                distance_km = haversine_distance(body.lat, body.lng, coords["lat"], coords["lng"])

        # City match bonus
        city_match = False
        if body.city:
            city_match = body.city.lower() in f.get("city", "").lower()

        # Compute relevance score: specialty match (0-3) + city bonus (1) + tier bonus
        tier_bonus = {"tertiary": 1.5, "secondary": 0.5, "primary": 0}.get(f.get("tier", ""), 0)
        score = spec_overlap + (1.0 if city_match else 0) + tier_bonus

        # Distance penalty for sorting (closer is better, but don't penalize too heavily)
        if distance_km is not None and distance_km > 50:
            score -= 0.5

        relevance_reasons = []
        if spec_overlap >= 2:
            relevance_reasons.append("Specialized cancer care center")
        elif spec_overlap == 1:
            relevant_specs = list(target_specialties & facility_specs)
            relevance_reasons.append(f"Relevant specialty: {relevant_specs[0].replace('_', ' ')}")
        if city_match:
            relevance_reasons.append(f"Located in {f.get('city', '')}")
        if f.get("tier") == "tertiary":
            relevance_reasons.append("Tertiary care facility")

        scored.append({
            "id": f["id"],
            "name": f["name"],
            "type": f["type"],
            "specialties": f.get("specialties", []),
            "address": f["address"],
            "city": f["city"],
            "state": f["state"],
            "phone": f["phone"],
            "website": f.get("website", ""),
            "tier": f.get("tier", "primary"),
            "distance_km": round(distance_km, 1) if distance_km is not None else None,
            "relevance_reason": "; ".join(relevance_reasons[:2]),
            "score": score,
        })

    # Sort by score (descending), then by distance (ascending) if available
    scored.sort(key=lambda x: (-x["score"], x["distance_km"] or 0))

    # Limit results
    results = scored[: body.limit]

    # Remove internal score from response
    for r in results:
        del r["score"]

    return {"recommendations": results, "source": "curated"}


@app.post("/facilities/search")
def search_facilities(body: FacilitySearchRequest):
    """Search for facilities using Google Places API (fallback)."""
    api_key = os.environ.get("GOOGLE_PLACES_API_KEY", "")
    if not api_key:
        return {"recommendations": [], "source": "unavailable", "error": "Google Places API key not configured"}

    try:
        import requests as req

        params = {
            "query": body.query,
            "key": api_key,
            "type": "hospital",
        }
        if body.lat is not None and body.lng is not None:
            params["location"] = f"{body.lat},{body.lng}"
            params["radius"] = body.radius

        resp = req.get("https://maps.googleapis.com/maps/api/place/textsearch/json", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "OK":
            return {"recommendations": [], "source": "google", "error": f"Google Places API error: {data.get('status')}"}

        results = []
        for place in data.get("results", [])[:5]:
            results.append({
                "id": place.get("place_id", ""),
                "name": place.get("name", ""),
                "type": "hospital",
                "address": place.get("formatted_address", ""),
                "rating": place.get("rating"),
                "total_ratings": place.get("user_ratings_total"),
                "open_now": place.get("opening_hours", {}).get("open_now"),
                "relevance_reason": f"Google Places result ({place.get('rating', 'N/A')} stars)",
            })

        return {"recommendations": results, "source": "google"}

    except Exception as e:
        return {"recommendations": [], "source": "google", "error": f"Google Places search failed: {str(e)}"}


if __name__ == "__main__":
    print("Starting FastAPI Server...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, loop="asyncio")