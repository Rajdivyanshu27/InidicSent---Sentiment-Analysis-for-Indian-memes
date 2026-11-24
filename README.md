---
title: MemeSense — Indian Meme Sentiment Analysis
emoji: 😎
colorFrom: purple
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
---

# MemeSense — Memotion 3.0 Multimodal Meme Analyzer

**MemeSense** is a multimodal web application for analyzing code-mixed Indian (Hinglish) and English internet memes across all three **Memotion 3.0 (AAAI)** shared-task objectives: **Sentiment Analysis (Task A)**, **Multi-Label Emotion Classification (Task B)**, and **Emotion Intensity Scales (Task C)**.

---

## Key Features

- **Task A — Sentiment Analysis:** Classifies memes as **Positive**, **Neutral**, or **Negative** with class probabilities, confidence score, and a natural-language mood summary.
- **Task B — Emotion Classification:** Simultaneously detects four emotion categories: **Humorous**, **Sarcastic**, **Offensive**, and **Motivational**.
- **Task C — Emotion Intensity Scales:** Quantifies expression intensity (`0–3` for Humorous, Sarcastic, and Offensive; `0–1` for Motivational) with **Task B ↔ Task C consistency enforcement**.
- **Auto-Analyzing Web UI (`MemeSense`):**
  - Drag-and-drop or browse meme upload (`JPG`, `PNG`, `WEBP`) with automatic OCR extraction and instant prediction (output stays hidden when cleared).
  - Editable **Detected Text (OCR)** drawer with live debounced re-analysis.
  - **Recent Analysis** strip, **History & Memotion 3.0 Dataset Explorer** (filterable across 1,500 validation & 1,500 test memes), and **About & Metrics** view.
- **Cross-Platform Multi-Pass OCR:** Runs upscaled, unsharp-masked, and high-contrast variants using Windows Media OCR (`ocr_run.ps1`) on Windows and `tesseract-ocr` on Linux/Docker.

---

## Multimodal Pipeline & Model Architecture

- **Visual Features (`features.py`):** `MobileNetV2` embeddings (`1280-d`) + OpenCV color/edge visual statistics (`18-d`) reduced with PCA (`128-d`) and standardized.
- **Text Features (`train_models.py` / `predictor.py`):** Word + character n-gram `TfidfVectorizer`, `TruncatedSVD` (`256-d`), and text statistics.
- **3-Model Ensemble per Target:** Blends sparse **Logistic Regression**, **XGBoost**, and a dual-branch **Keras Neural Network** (`artifacts/keras_models/`), tuned via out-of-fold validation weights.

### Validation Metrics (`n = 1,500`)

| Task | Metric | Score |
|---|---|---|
| **Task A — Sentiment** | Weighted F1 / Macro F1 / Accuracy | `31.87%` / `30.82%` / `35.40%` |
| **Task B — Emotions (Avg)** | Avg Weighted F1 | `80.37%` *(Humour `90.2%`, Sarcastic `87.9%`, Offensive `47.7%`, Motivational `95.7%`)* |
| **Task C — Intensity (Avg)** | Avg Weighted F1 | `55.39%` *(Humour `52.7%`, Sarcastic `31.7%`, Offensive `41.4%`, Motivational `95.7%`)* |

*(Dataset splits: 7,000 train / 1,500 validation / 1,500 test memes.)*

---

## Project Structure

```text
memotion_app/
├── server.py               # FastAPI server & REST endpoints
├── predictor.py            # Multimodal inference & Task B/C consistency
├── features.py             # MobileNetV2, OpenCV stats, & text feature extractors
├── ocr_helper.py           # Cross-platform multi-pass OCR (Windows OCR / Tesseract)
├── ocr_run.ps1             # Windows WinRT Media OCR runner
├── train_models.py         # Feature extraction, 5-fold OOF training & bundle export
├── constants.py            # Label mappings, scales, and paths
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build with tesseract-ocr & Uvicorn
├── artifacts/
│   ├── model_bundle.joblib # Trained TF-IDF, SVD, PCA, LR & XGBoost bundle
│   ├── metrics.json        # Validation metrics
│   ├── val_predictions.json
│   ├── test_predictions.json
│   └── keras_models/       # Saved Keras dual-branch models (.keras)
└── static/
    ├── index.html          # MemeSense single-page frontend
    ├── assets/             # Background wallpaper, hero illustrations & logo
    └── featured/           # Featured sample memes for quick preview
```

---

## Running Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(On Linux/macOS, also install `tesseract-ocr` for image caption extraction. On Windows, built-in Windows Media OCR is used automatically.)*

### 2. Start the Server
```bash
python server.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## Docker Deployment

```bash
docker build -t memesense .
docker run -p 8000:8000 memesense
```
The container binds automatically to `0.0.0.0:${PORT:-8000}`, making it compatible with **Railway**, **Render**, and **Hugging Face Docker Spaces**.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the MemeSense UI |
| `GET` | `/api/overview` | Validation metrics, label schema, and featured samples |
| `GET` | `/api/samples` | Paginated & filterable dataset explorer (`split`, `sentiment`, `emotion`, `q`) |
| `GET` | `/api/image/{split}/{image_id}` | Serves dataset or featured fallback meme JPEG |
| `POST` | `/api/ocr` | Runs multi-pass OCR on an uploaded meme image |
| `POST` | `/api/analyze` | Predicts Task A, Task B, and Task C for an uploaded image or sample |
