# app/routers/analyze.py
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any, Tuple
from enum import Enum
import os, json

router = APIRouter()

# --- Env / config ---
DS_MODEL_VERSION = os.getenv("DS_MODEL_VERSION", "ds-v0.1.0")
DS_USE_STUB = os.getenv("DS_USE_STUB", "true").lower() == "true"
ATTN_THRESH = float(os.getenv("DS_ATTENTION_THRESH", "0.65"))
RETAKE_QUALITY = float(os.getenv("DS_RETAKE_QUALITY", "0.50"))

# Try to import real inference (used only when DS_USE_STUB=false)
_real_inf = None
try:
    from app.services import inference as _real_inf
except Exception:
    _real_inf = None  # ok for stub mode

# --- Response Schemas (updated for 6 classes) ---
class OverallStatus(str, Enum):
    OK = "OK"
    ATTENTION = "ATTENTION"
    RETAKE = "RETAKE"

class Finding(BaseModel):
    type: Literal["Calculus", "Caries", "Gingivitis", "Mouth Ulcer", "Tooth Discoloration", "Hypodontia"]
    confidence: float = Field(ge=0.0, le=1.0)

class Quality(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasons: List[str] = []

class AnalyzeResponse(BaseModel):
    overall_status: OverallStatus
    quality: Quality
    findings: List[Finding]
    model_version: str
    recommendations: List[str] = []
    debug: Dict[str, Any] = {}

# --- Helpers (stub heuristics) ---
def _stub_quality_and_findings(url: str) -> Tuple[Quality, List[Finding], Dict[str, Any]]:
    """Stub implementation for testing - returns mock data based on URL keywords"""
    u = (url or "").lower()
    blur_score = 0.85
    brightness_score = 0.85
    mouth_open_score = 0.60
    reasons: List[str] = []

    if "blurry" in u or "blur" in u:
        blur_score = 0.35; reasons.append("blurry")
    if "dark" in u or "lowlight" in u:
        brightness_score = 0.35; reasons.append("low_light")
    if "mouthclosed" in u or "closed" in u:
        mouth_open_score = 0.25; reasons.append("mouth_closed")

    quality_score = round((blur_score + brightness_score + mouth_open_score) / 3, 4)

    # Initialize all 6 classes with low baseline confidence
    calculus = 0.15
    caries = 0.18
    gingivitis = 0.20
    mouth_ulcer = 0.12
    tooth_discoloration = 0.14
    hypodontia = 0.10

    # Adjust based on URL keywords
    if "calculus" in u or "plaque" in u or "tartar" in u:
        calculus = 0.82
    if "caries" in u or "cavity" in u or "spot" in u:
        caries = 0.85
    if "gingivitis" in u or "red" in u or "gums" in u:
        gingivitis = 0.80
    if "ulcer" in u or "sore" in u:
        mouth_ulcer = 0.78
    if "discoloration" in u or "stain" in u or "yellow" in u:
        tooth_discoloration = 0.75
    if "hypodontia" in u or "missing" in u:
        hypodontia = 0.88

    findings = [
        Finding(type="Calculus", confidence=round(calculus, 4)),
        Finding(type="Caries", confidence=round(caries, 4)),
        Finding(type="Gingivitis", confidence=round(gingivitis, 4)),
        Finding(type="Mouth Ulcer", confidence=round(mouth_ulcer, 4)),
        Finding(type="Tooth Discoloration", confidence=round(tooth_discoloration, 4)),
        Finding(type="Hypodontia", confidence=round(hypodontia, 4)),
    ]
    
    debug = {
        "quality": {
            "blur_score": round(blur_score, 4),
            "brightness_score": round(brightness_score, 4),
            "mouth_open_score": round(mouth_open_score, 4),
        },
        "best_frames": 0
    }
    return Quality(score=quality_score, reasons=reasons), findings, debug

def _triage(quality: Quality, findings: List[Finding]) -> OverallStatus:
    """Determine overall status based on quality and findings"""
    if quality.score < RETAKE_QUALITY:
        return OverallStatus.RETAKE
    if any(f.confidence >= ATTN_THRESH for f in findings):
        return OverallStatus.ATTENTION
    return OverallStatus.OK

def _recommendations(status: OverallStatus, quality: Quality, findings: List[Finding]) -> List[str]:
    """Generate recommendations based on status, quality, and findings"""
    recs: List[str] = []
    
    if status == OverallStatus.RETAKE:
        if "low_light" in quality.reasons: 
            recs.append("Increase lighting or face a bright window.")
        if "blurry" in quality.reasons: 
            recs.append("Hold the camera steady; use the rear camera if possible.")
        if "mouth_closed" in quality.reasons: 
            recs.append("Open mouth wider; include upper and lower teeth.")
        if not recs: 
            recs.append("Retake the photo with better lighting and a steady hand.")
    else:
        # Generate recommendations based on detected conditions
        for f in findings:
            if f.type == "Calculus" and f.confidence >= 0.6:
                recs.append("Consider professional teeth cleaning to remove calculus buildup.")
            elif f.type == "Caries" and f.confidence >= 0.6:
                recs.append("Schedule a dental exam to evaluate potential cavities or caries.")
            elif f.type == "Gingivitis" and f.confidence >= 0.6:
                recs.append("Improve gum health with gentle brushing and antiseptic mouthwash.")
            elif f.type == "Mouth Ulcer" and f.confidence >= 0.6:
                recs.append("Monitor mouth ulcers; consult a dentist if they persist beyond 2 weeks.")
            elif f.type == "Tooth Discoloration" and f.confidence >= 0.6:
                recs.append("Consider professional cleaning or whitening treatment for tooth discoloration.")
            elif f.type == "Hypodontia" and f.confidence >= 0.6:
                recs.append("Consult with a dentist about missing teeth and potential treatment options.")
        
        if status == OverallStatus.OK and not recs:
            recs.append("Maintain good oral hygiene and routine checkups.")
    
    # Deduplicate while preserving order
    seen, out = set(), []
    for r in recs:
        if r not in seen:
            seen.add(r); out.append(r)
    return out

def _aggregate_multi_image(per_view: List[Tuple[str, Quality, List[Finding], Dict[str, Any]]]) -> Tuple[Quality, List[Finding], Dict[str, Any]]:
    """Aggregate results from multiple images by taking max confidence per class"""
    if not per_view:
        return Quality(score=0.0, reasons=["no_images"]), [], {"per_view": []}

    # Average quality score
    avg_q = sum(q.score for _, q, _, _ in per_view) / len(per_view)
    
    # Collect all quality reasons
    reasons: List[str] = []
    for _, q, _, _ in per_view:
        for r in q.reasons:
            if r not in reasons:
                reasons.append(r)

    # Max confidence across views for each finding type
    max_conf = {
        "Calculus": 0.0,
        "Caries": 0.0,
        "Gingivitis": 0.0,
        "Mouth Ulcer": 0.0,
        "Tooth Discoloration": 0.0,
        "Hypodontia": 0.0
    }
    
    for _, _, fnds, _ in per_view:
        for f in fnds:
            if f.confidence > max_conf[f.type]:
                max_conf[f.type] = f.confidence
    
    findings = [Finding(type=k, confidence=round(v, 4)) for k, v in max_conf.items()]

    debug = {
        "per_view": [
            {
                "view": v,
                "quality": pv_dbg.get("quality", {}),
            }
            for v, _, _, pv_dbg in per_view
        ],
        "best_frames": sum(pv_dbg.get("best_frames", 0) for _, _, _, pv_dbg in per_view)
    }
    return Quality(score=round(avg_q, 4), reasons=reasons), findings, debug

VALID_VIEWS = {"front", "left", "right", "top", "bottom"}

# --- Route ---
@router.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(payload: Dict[str, Any] = Body(...)):
    """
    Accepts either:
      { "media_url": "https://..." }
    or:
      { "images": [ { "view": "front|left|right|top|bottom", "url": "https://..." }, ... ] }
    Optionally include "patient_id" to store result for GET retrieval.
    """
    media_url = payload.get("media_url")
    images = payload.get("images")
    patient_id = payload.get("patient_id")
    use_stub = DS_USE_STUB

    # -------------------------
    # MULTI-IMAGE FLOW
    # -------------------------
    if images is not None:
        if not isinstance(images, list):
            raise HTTPException(status_code=422, detail="'images' must be an array of {view,url} objects.")

        # Validate & collect URLs
        urls: List[str] = []
        views_norm: List[str] = []
        for item in images:
            if not isinstance(item, dict):
                raise HTTPException(status_code=422, detail="Each item in 'images' must be an object with 'view' and 'url'.")
            view = str(item.get("view", "")).strip().lower()
            url = str(item.get("url", "")).strip()
            if view not in VALID_VIEWS:
                raise HTTPException(status_code=422, detail=f"Invalid view '{item.get('view')}'. Must be one of {sorted(VALID_VIEWS)}.")
            if not (url.startswith("http://") or url.startswith("https://")):
                raise HTTPException(status_code=422, detail=f"Invalid url '{url}'. Must start with http:// or https://.")
            urls.append(url); views_norm.append(view)

        if use_stub:
            # Stub path (mock data)
            per_view: List[Tuple[str, Quality, List[Finding], Dict[str, Any]]] = []
            for url, v in zip(urls, views_norm):
                q, f, dbg = _stub_quality_and_findings(url)
                per_view.append((v, q, f, dbg))

            quality, findings, dbg = _aggregate_multi_image(per_view)
            status = _triage(quality, findings)
            recs = _recommendations(status, quality, findings)
            result = AnalyzeResponse(
                overall_status=status,
                quality=quality,
                findings=findings,
                model_version=DS_MODEL_VERSION,
                recommendations=recs,
                debug=dbg
            )
        else:
            # Real inference path - calls inference.aggregate_views()
            if _real_inf is None:
                raise HTTPException(
                    status_code=501,
                    detail="Real inference not available: app.services.inference not importable or missing dependencies."
                )
            try:
                # Returns dict with 6 classes: {"Calculus": 0.15, "Caries": 0.82, ...}
                max_conf = _real_inf.aggregate_views(urls)
            except FileNotFoundError as e:
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"One or more images failed: {e}")

            # Convert to Finding objects
            findings = [
                Finding(type="Calculus", confidence=round(max_conf.get("Calculus", 0.0), 4)),
                Finding(type="Caries", confidence=round(max_conf.get("Caries", 0.0), 4)),
                Finding(type="Gingivitis", confidence=round(max_conf.get("Gingivitis", 0.0), 4)),
                Finding(type="Mouth Ulcer", confidence=round(max_conf.get("Mouth Ulcer", 0.0), 4)),
                Finding(type="Tooth Discoloration", confidence=round(max_conf.get("Tooth Discoloration", 0.0), 4)),
                Finding(type="Hypodontia", confidence=round(max_conf.get("Hypodontia", 0.0), 4)),
            ]
            
            # Placeholder quality in real path; plug in real quality if available
            quality = Quality(score=1.0, reasons=[])
            status = _triage(quality, findings)
            recs = _recommendations(status, quality, findings)
            result = AnalyzeResponse(
                overall_status=status,
                quality=quality,
                findings=findings,
                model_version=DS_MODEL_VERSION,
                recommendations=recs,
                debug={"best_frames": 0}
            )

        # Save by patient_id if provided
        if patient_id:
            os.makedirs("outputs", exist_ok=True)
            with open(f"outputs/{patient_id}.json", "w") as f:
                json.dump(result.model_dump(), f, indent=2)
        return result

    # -------------------------
    # SINGLE-IMAGE FLOW
    # -------------------------
    if not media_url:
        raise HTTPException(status_code=422, detail="Provide either 'media_url' (single image) or 'images' (array).")
    if not (isinstance(media_url, str) and (media_url.startswith("http://") or media_url.startswith("https://"))):
        raise HTTPException(status_code=422, detail="media_url must be a valid http(s) URL.")

    if use_stub:
        # Stub single-image path (mock data)
        quality, findings, dbg = _stub_quality_and_findings(media_url)
        status = _triage(quality, findings)
        recs = _recommendations(status, quality, findings)
        result = AnalyzeResponse(
            overall_status=status,
            quality=quality,
            findings=findings,
            model_version=DS_MODEL_VERSION,
            recommendations=recs,
            debug=dbg
        )
    else:
        # Real single-image inference - calls inference.infer_image()
        if _real_inf is None:
            raise HTTPException(
                status_code=501,
                detail="Real inference not available: app.services.inference not importable or missing dependencies."
            )
        try:
            # Returns dict with 6 classes: {"Calculus": 0.15, "Caries": 0.82, ...}
            max_conf = _real_inf.infer_image(media_url)
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image fetch/decode failed: {e}")

        # Convert to Finding objects
        findings = [
            Finding(type="Calculus", confidence=round(max_conf.get("Calculus", 0.0), 4)),
            Finding(type="Caries", confidence=round(max_conf.get("Caries", 0.0), 4)),
            Finding(type="Gingivitis", confidence=round(max_conf.get("Gingivitis", 0.0), 4)),
            Finding(type="Mouth Ulcer", confidence=round(max_conf.get("Mouth Ulcer", 0.0), 4)),
            Finding(type="Tooth Discoloration", confidence=round(max_conf.get("Tooth Discoloration", 0.0), 4)),
            Finding(type="Hypodontia", confidence=round(max_conf.get("Hypodontia", 0.0), 4)),
        ]
        
        # Placeholder quality in real path; plug in real quality if available
        quality = Quality(score=1.0, reasons=[])
        status = _triage(quality, findings)
        recs = _recommendations(status, quality, findings)
        result = AnalyzeResponse(
            overall_status=status,
            quality=quality,
            findings=findings,
            model_version=DS_MODEL_VERSION,
            recommendations=recs,
            debug={"best_frames": 0}
        )

    # Save by patient_id if provided
    if patient_id:
        os.makedirs("outputs", exist_ok=True)
        with open(f"outputs/{patient_id}.json", "w") as f:
            json.dump(result.model_dump(), f, indent=2)

    return result

# --- GET: fetch stored output for patient ---
@router.get("/v1/analyze/{patient_id}")
def get_analysis_result(patient_id: str):
    filepath = f"outputs/{patient_id}.json"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Result not found for this patient ID")
    with open(filepath, "r") as f:
        data = json.load(f)
    return data