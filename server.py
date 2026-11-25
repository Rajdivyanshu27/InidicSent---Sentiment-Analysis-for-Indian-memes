import json
import os
import tempfile
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from constants import ARTIFACTS_DIR, EMOTION_DISPLAY, EMOTION_KEYS, IMAGE_DIRS, SENTIMENT_LABELS, STATIC_DIR, TASK_C_SCALES
from ocr_helper import extract_ocr_text
from predictor import MemotionPredictor

app = FastAPI(title="Memotion 3.0 Multimodal Analyzer")

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_PREDICTOR: Optional[MemotionPredictor] = None
_SAMPLES_CACHE: dict[str, list] = {}
_EVAL_REPORT_CACHE: Optional[dict] = None


def load_evaluation_report() -> Optional[dict]:
    global _EVAL_REPORT_CACHE
    if _EVAL_REPORT_CACHE is None:
        report_path = os.path.join(ARTIFACTS_DIR, "evaluation_report.json")
        if os.path.isfile(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                _EVAL_REPORT_CACHE = json.load(f)
    return _EVAL_REPORT_CACHE


def get_predictor() -> MemotionPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = MemotionPredictor()
    return _PREDICTOR


def load_split_records(split: str) -> list:
    if split not in ("val", "test"):
        raise HTTPException(status_code=400, detail="Split must be 'val' or 'test'.")
    if split not in _SAMPLES_CACHE:
        json_path = os.path.join(ARTIFACTS_DIR, f"{split}_predictions.json")
        if not os.path.isfile(json_path):
            raise HTTPException(status_code=503, detail=f"Missing {split}_predictions.json; run train_models.py first.")
        with open(json_path, "r", encoding="utf-8") as f:
            _SAMPLES_CACHE[split] = json.load(f)
    return _SAMPLES_CACHE[split]


@app.on_event("startup")
def startup_event():
    get_predictor()
    load_split_records("val")
    load_split_records("test")


@app.get("/", response_class=HTMLResponse)
def serve_home():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.isfile(index_path):
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read(), headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/api/overview")
def get_overview():
    pred = get_predictor()
    val_records = load_split_records("val")
    test_records = load_split_records("test")

    # Pick diverse featured validation samples across Positive / Negative / Neutral
    featured_val = []
    used_ids = set()
    for target in ("Positive", "Negative", "Neutral", "Positive", "Negative"):
        match = next(
            (r for r in val_records if r["id"] not in used_ids and r["prediction"]["task_a"]["label"] == target),
            None,
        )
        if match:
            featured_val.append(match)
            used_ids.add(match["id"])
    for r in val_records:
        if len(featured_val) >= 6:
            break
        if r["id"] not in used_ids:
            featured_val.append(r)
            used_ids.add(r["id"])

    return {
        "metrics": pred.metrics,
        "evaluation": load_evaluation_report(),
        "schema": {
            "sentiment_labels": SENTIMENT_LABELS,
            "emotions": [{"key": k, "name": EMOTION_DISPLAY[k], "scale": TASK_C_SCALES[k]["labels"]} for k in EMOTION_KEYS],
        },
        "featured_samples": {
            "val": featured_val,
            "test": test_records[:6],
        },
    }


@app.get("/api/evaluation")
def get_evaluation():
    report = load_evaluation_report()
    if report is None:
        raise HTTPException(status_code=503, detail="evaluation_report.json not found; run generate_evaluation.py first.")
    return report


@app.get("/api/samples")
def list_samples(
    split: str = Query("val"),
    offset: int = Query(0, ge=0),
    limit: int = Query(12, ge=1, le=48),
    sentiment: str = Query("all"),
    emotion: str = Query("all"),
    q: str = Query(""),
):
    records = load_split_records(split)
    q_low = q.strip().lower()
    filtered = []
    for r in records:
        p = r["prediction"]
        if sentiment != "all" and p["task_a"]["label"].lower() != sentiment.lower():
            continue
        if emotion != "all" and not p["task_b"].get(emotion, {}).get("active", False):
            continue
        if q_low and q_low not in r["ocr"].lower() and q_low != str(r["id"]):
            continue
        filtered.append(r)

    total = len(filtered)
    items = filtered[offset:offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "items": items}


def _resolve_sample_image(split: str, image_id: int) -> Optional[str]:
    """Find dataset image locally or in static/featured/ fallback for cloud hosting."""
    if split in IMAGE_DIRS:
        img_path = os.path.join(IMAGE_DIRS[split], f"{image_id}.jpg")
        if os.path.isfile(img_path):
            return img_path
    featured_path = os.path.join(STATIC_DIR, "featured", f"{split}_{image_id}.jpg")
    if os.path.isfile(featured_path):
        return featured_path
    return None


@app.get("/api/image/{split}/{image_id}")
def get_image(split: str, image_id: int):
    if split not in IMAGE_DIRS:
        raise HTTPException(status_code=400, detail="Invalid split")
    img_path = _resolve_sample_image(split, image_id)
    if not img_path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img_path, media_type="image/jpeg")


@app.post("/api/ocr")
async def run_ocr(image: UploadFile = File(...)):
    suffix = os.path.splitext(image.filename or "meme.jpg")[1] or ".jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await image.read())
            tmp_path = tmp.name
        text = extract_ocr_text(tmp_path)
        return {"ocr": text}
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            os.remove(tmp_path)


async def _resolve_input_and_run(
    image: Optional[UploadFile],
    ocr_text: str,
    split: Optional[str],
    sample_id: Optional[int],
    auto_ocr: bool,
    runner,
):
    tmp_path = None
    try:
        ground_truth = None
        if split is not None and sample_id is not None:
            if split not in IMAGE_DIRS:
                raise HTTPException(status_code=400, detail="Invalid split")
            img_path = _resolve_sample_image(split, sample_id)
            if not img_path:
                raise HTTPException(status_code=404, detail="Sample image not found")
            if split in ("val", "test"):
                recs = load_split_records(split)
                match = next((r for r in recs if r["id"] == sample_id), None)
                if match:
                    if not ocr_text.strip():
                        ocr_text = match["ocr"]
                    ground_truth = match.get("ground_truth")
        elif image is not None:
            suffix = os.path.splitext(image.filename or "meme.jpg")[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await image.read())
                tmp_path = tmp.name
            img_path = tmp_path
            if auto_ocr and not ocr_text.strip():
                ocr_text = extract_ocr_text(img_path)
        else:
            raise HTTPException(status_code=400, detail="Provide an uploaded image or a dataset sample.")

        result = runner(img_path, ocr_text)
        if ground_truth is not None:
            result["ground_truth"] = ground_truth
        return result
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            os.remove(tmp_path)


@app.post("/api/analyze")
async def analyze_meme(
    image: Optional[UploadFile] = File(None),
    ocr_text: str = Form(""),
    split: Optional[str] = Form(None),
    sample_id: Optional[int] = Form(None),
    auto_ocr: bool = Form(True),
):
    predictor = get_predictor()
    return await _resolve_input_and_run(image, ocr_text, split, sample_id, auto_ocr, predictor.predict)


@app.post("/api/explain")
async def explain_meme(
    image: Optional[UploadFile] = File(None),
    ocr_text: str = Form(""),
    split: Optional[str] = Form(None),
    sample_id: Optional[int] = Form(None),
    auto_ocr: bool = Form(True),
):
    predictor = get_predictor()
    return await _resolve_input_and_run(image, ocr_text, split, sample_id, auto_ocr, predictor.explain)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
