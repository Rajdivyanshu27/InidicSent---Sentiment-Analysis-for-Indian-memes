---
title: MemeSense — Indian Meme Sentiment Analysis
emoji: 😎
colorFrom: purple
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
---

# MemeSense — Memotion 3.0 Multimodal Analyzer

Multimodal sentiment, multi-label emotion, and intensity-scale analyzer for code-mixed Indian/English internet memes built on FastAPI, MobileNetV2, TF-IDF + SVD, XGBoost, and Keras dual-branch networks.

## Tasks Supported
- **Task A — Sentiment Analysis:** Positive / Neutral / Negative with confidence breakdown.
- **Task B — Emotion Classification:** Multi-label detection (`Humorous`, `Sarcastic`, `Offensive`, `Motivational`).
- **Task C — Emotion Intensity Scales:** Graded intensity prediction enforced consistently with Task B.

## Run Locally
```bash
pip install -r requirements.txt
python server.py
```
Open `http://127.0.0.1:8000`.

## Deploy with Docker
```bash
docker build -t memesense .
docker run -p 8000:8000 memesense
```
