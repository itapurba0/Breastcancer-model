import os
import json
import math
from typing import Dict, List, Optional

from pydantic import BaseModel

BASE_DIR = os.path.dirname(__file__)
FACILITIES_PATH = os.path.join(BASE_DIR, "facilities.json")


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

    if pred_name == "normal":
        return {
            "tier": "Routine Screening",
            "recommendation": "Continue routine screening as per guidelines. No abnormal findings detected.",
            "rationale": f"Normal classification with {confidence:.1%} confidence. Regular screening schedule recommended.",
        }

    return None


def load_facilities() -> list:
    """Load the curated facility dataset."""
    if not os.path.exists(FACILITIES_PATH):
        return []
    with open(FACILITIES_PATH, "r") as f:
        data = json.load(f)
    return data.get("facilities", [])


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    R = 6371.0
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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


def recommend_facilities(body: FacilityRecommendRequest) -> dict:
    """Recommend medical facilities based on classification result and optional location."""
    facilities = load_facilities()
    if not facilities:
        return {"recommendations": [], "source": "curated", "error": "No facility data available"}

    pred_key = "inconclusive" if body.inconclusive else body.prediction.lower()
    target_specialties = set(SPECIALTY_MAP.get(pred_key, SPECIALTY_MAP["inconclusive"]))

    scored = []
    for f in facilities:
        facility_specs = set(f.get("specialties", []))
        spec_overlap = len(target_specialties & facility_specs)
        if spec_overlap == 0:
            continue

        distance_km = None
        if body.lat is not None and body.lng is not None:
            coords = f.get("coordinates", {})
            if "lat" in coords and "lng" in coords:
                distance_km = haversine_distance(body.lat, body.lng, coords["lat"], coords["lng"])

        city_match = False
        if body.city:
            city_match = body.city.lower() in f.get("city", "").lower()

        tier_bonus = {"tertiary": 1.5, "secondary": 0.5, "primary": 0}.get(f.get("tier", ""), 0)
        score = spec_overlap + (1.0 if city_match else 0) + tier_bonus

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

    scored.sort(key=lambda x: (-x["score"], x["distance_km"] or 0))
    results = scored[: body.limit]

    for r in results:
        del r["score"]

    return {"recommendations": results, "source": "curated"}


async def search_facilities(body: FacilitySearchRequest) -> dict:
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