import base64
import hashlib
import html
import json
import os
import tempfile

import streamlit as st

from constants import ARTIFACTS_DIR, EMOTION_DISPLAY, EMOTION_KEYS, IMAGE_DIRS, STATIC_DIR, TASK_C_SCALES
from ocr_helper import extract_ocr_text
from predictor import MemotionPredictor

st.set_page_config(
    page_title="MemeSense — Indian Meme Sentiment Analysis",
    page_icon="😎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading MemeSense multimodal models...")
def load_predictor() -> MemotionPredictor:
    return MemotionPredictor()


@st.cache_data
def load_split_records(split: str) -> list:
    path = os.path.join(ARTIFACTS_DIR, f"{split}_predictions.json")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def asset_b64(rel_path: str) -> str:
    full = os.path.join(STATIC_DIR, rel_path)
    if not os.path.isfile(full):
        return ""
    with open(full, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def resolve_sample_image(split: str, image_id: int) -> str | None:
    if split in IMAGE_DIRS:
        p = os.path.join(IMAGE_DIRS[split], f"{image_id}.jpg")
        if os.path.isfile(p):
            return p
    feat = os.path.join(STATIC_DIR, "featured", f"{split}_{image_id}.jpg")
    if os.path.isfile(feat):
        return feat
    return None


def get_featured_samples() -> list:
    val_records = load_split_records("val")
    picks = []
    used = set()
    for target in ("Positive", "Negative", "Neutral", "Positive", "Negative"):
        match = next(
            (r for r in val_records if r["id"] not in used and r["prediction"]["task_a"]["label"] == target and resolve_sample_image("val", r["id"])),
            None,
        )
        if match:
            picks.append(match)
            used.add(match["id"])
    for r in val_records:
        if len(picks) >= 5:
            break
        if r["id"] not in used and resolve_sample_image("val", r["id"]):
            picks.append(r)
            used.add(r["id"])
    return picks[:5]


def inject_theme_css():
    bg_b64 = asset_b64(os.path.join("assets", "bg_meme_faces.jpg"))
    bg_rule = (
        f'background-image: radial-gradient(ellipse 68% 46% at 50% 18%, rgba(109, 40, 217, 0.36), transparent 72%), '
        f'linear-gradient(180deg, rgba(19, 17, 32, 0.72), rgba(16, 14, 28, 0.80)), '
        f'url("data:image/jpeg;base64,{bg_b64}"); background-size: auto, auto, 1180px auto; background-repeat: no-repeat, no-repeat, repeat;'
        if bg_b64
        else "background-color: #131120;"
    )
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        header[data-testid="stHeader"] {{ background: transparent; }}
        #MainMenu, footer {{ visibility: hidden; }}

        .stApp {{
            {bg_rule}
            color: #f8fafc;
            font-family: "Plus Jakarta Sans", sans-serif;
        }}
        .block-container {{
            max-width: 1180px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }}
        .ms-navbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 18px;
            border-radius: 16px;
            background: rgba(19, 17, 32, 0.82);
            border: 1px solid rgba(139, 92, 246, 0.25);
            backdrop-filter: blur(12px);
            margin-bottom: 14px;
        }}
        .ms-brand {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-family: "Fredoka", sans-serif;
            font-size: 24px;
            font-weight: 700;
        }}
        .ms-brand img {{
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: #fff;
            padding: 2px;
        }}
        .ms-hero {{
            text-align: center;
            padding: 12px 10px 22px;
        }}
        .ms-hero-title {{
            font-family: "Fredoka", sans-serif;
            font-size: clamp(30px, 4vw, 46px);
            font-weight: 700;
            line-height: 1.08;
            text-shadow: 0 6px 22px rgba(0, 0, 0, 0.65);
        }}
        .ms-gold {{ color: #facc15; }}
        .ms-violet {{ color: #a855f7; }}
        .ms-hero-sub {{
            margin-top: 8px;
            font-family: "Fredoka", sans-serif;
            font-size: 17px;
            color: #e2e8f0;
        }}
        .ms-card {{
            background: rgba(29, 25, 48, 0.88);
            border: 1px solid rgba(139, 92, 246, 0.30);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 18px 42px rgba(7, 5, 16, 0.65);
        }}
        .ms-sent-banner {{
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 14px 16px;
            border-radius: 14px;
            margin-bottom: 10px;
        }}
        .ms-sent-Positive {{ background: rgba(34, 197, 94, 0.16); border: 1.5px solid rgba(74, 222, 128, 0.55); color: #4ade80; }}
        .ms-sent-Negative {{ background: rgba(244, 63, 94, 0.16); border: 1.5px solid rgba(251, 113, 133, 0.55); color: #fb7185; }}
        .ms-sent-Neutral  {{ background: rgba(148, 163, 184, 0.16); border: 1.5px solid rgba(148, 163, 184, 0.45); color: #cbd5e1; }}
        .ms-tri-bar {{
            width: 100%;
            height: 9px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            overflow: hidden;
            display: flex;
            margin: 8px 0 14px;
        }}
        .ms-chip {{
            display: flex;
            justify-content: space-between;
            padding: 6px 10px;
            border-radius: 9px;
            margin-bottom: 6px;
            font-size: 12.5px;
            font-weight: 600;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            color: #94a3b8;
        }}
        .ms-chip.active {{
            background: rgba(124, 58, 237, 0.28);
            border-color: rgba(167, 139, 250, 0.55);
            color: #ffffff;
        }}
        .ms-badge {{
            display: inline-block;
            padding: 3px 11px;
            border-radius: 999px;
            font-size: 11.5px;
            font-weight: 800;
            margin-top: 6px;
        }}
        .ms-badge-Positive {{ background: #4ade80; color: #052e16; }}
        .ms-badge-Negative {{ background: #fb7185; color: #4c0519; }}
        .ms-badge-Neutral  {{ background: #cbd5e1; color: #0f172a; }}
        div.stButton > button {{
            border-radius: 999px;
            font-weight: 700;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        div.stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(124, 58, 237, 0.35);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_mood_summary(ta: dict, tb: dict) -> str:
    active = [EMOTION_DISPLAY[k].lower() for k in EMOTION_KEYS if tb[k]["active"]]
    base = (
        "This meme expresses a positive sentiment"
        if ta["label"] == "Positive"
        else "This meme carries a negative or critical sentiment"
        if ta["label"] == "Negative"
        else "This meme reflects a balanced or neutral sentiment"
    )
    if not active:
        return base + "."
    if len(active) == 1:
        return f"{base}, highlighting a {active[0]} tone."
    return f"{base}, blending {', '.join(active[:-1])} and {active[-1]} tones."


def init_state():
    defaults = {
        "nav_tab": "Home",
        "uploader_nonce": 0,
        "last_upload_sig": None,
        "active_item": None,
        "session_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def analyze_bytes(predictor: MemotionPredictor, img_bytes: bytes, filename: str, ocr_override: str | None = None) -> dict:
    suffix = os.path.splitext(filename or "meme.jpg")[1] or ".jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        ocr_text = extract_ocr_text(tmp_path) if ocr_override is None else ocr_override
        pred = predictor.predict(tmp_path, ocr_text)
        return {
            "source": "upload",
            "title": filename or "Uploaded Meme",
            "img_bytes": img_bytes,
            "ocr": ocr_text,
            "prediction": {"task_a": pred["task_a"], "task_b": pred["task_b"], "task_c": pred["task_c"]},
            "ground_truth": None,
        }
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            os.remove(tmp_path)


def render_result_panel(predictor: MemotionPredictor, item: dict):
    with st.container():
        top_l, top_r = st.columns([4, 1])
        with top_l:
            st.markdown("#### ✨ Analysis Result")
        with top_r:
            if st.button("✕ Clear", key="clear_result_btn", use_container_width=True):
                st.session_state.active_item = None
                st.session_state.last_upload_sig = None
                st.session_state.uploader_nonce += 1
                st.rerun()

        img_col, sent_col = st.columns([0.92, 1.08], gap="medium")
        with img_col:
            if item.get("img_bytes"):
                st.image(item["img_bytes"], use_container_width=True)
            elif item.get("img_path"):
                st.image(item["img_path"], use_container_width=True)

        pred = item["prediction"]
        ta = pred["task_a"]
        tb = pred["task_b"]
        tc = pred["task_c"]
        p_neg = (ta["probs"][0] or 0.0) * 100
        p_neu = (ta["probs"][1] or 0.0) * 100
        p_pos = (ta["probs"][2] or 0.0) * 100
        conf = max(p_neg, p_neu, p_pos)
        icon = "😊" if ta["label"] == "Positive" else "😟" if ta["label"] == "Negative" else "😐"

        with sent_col:
            st.markdown(
                f"""
                <div class="ms-sent-banner ms-sent-{ta['label']}">
                  <div style="font-size:32px;line-height:1;">{icon}</div>
                  <div>
                    <div style="font-family:'Fredoka',sans-serif;font-size:26px;font-weight:700;line-height:1.05;">{ta['label']}</div>
                    <div style="font-size:13px;color:#e2e8f0;font-weight:600;">Confidence: {conf:.1f}%</div>
                  </div>
                </div>
                <div style="font-size:13px;color:#cbd5e1;margin-bottom:10px;">{html.escape(build_mood_summary(ta, tb))}</div>
                <div style="display:flex;justify-content:space-between;font-size:12.5px;font-weight:700;">
                  <span><span style="color:#22c55e;">●</span> Positive {p_pos:.1f}%</span>
                  <span><span style="color:#ef4444;">●</span> Negative {p_neg:.1f}%</span>
                  <span><span style="color:#94a3b8;">●</span> Neutral {p_neu:.1f}%</span>
                </div>
                <div class="ms-tri-bar">
                  <div style="width:{p_pos:.1f}%;background:#22c55e;"></div>
                  <div style="width:{p_neg:.1f}%;background:#ef4444;"></div>
                  <div style="width:{p_neu:.1f}%;background:#94a3b8;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        b_col, c_col = st.columns(2, gap="medium")
        with b_col:
            active_cnt = sum(1 for k in EMOTION_KEYS if tb[k]["active"])
            st.caption(f"**Task B · Emotions** ({active_cnt} active)")
            chips_html = "".join(
                f'<div class="ms-chip {"active" if tb[k]["active"] else ""}">'
                f'<span>{EMOTION_DISPLAY[k]}</span>'
                f'<span style="color:{"#facc15" if tb[k]["active"] else "#64748b"};">{round(tb[k]["prob"]*100)}%</span>'
                f"</div>"
                for k in EMOTION_KEYS
            )
            st.markdown(chips_html, unsafe_allow_html=True)

        with c_col:
            st.caption("**Task C · Intensity Scales**")
            for k in EMOTION_KEYS:
                lvl = tc[k]["level"]
                mx = tc[k]["max_level"]
                lbl = tc[k]["label"]
                pip_w = int(88 / max(mx, 1))
                pips = "".join(
                    f'<span style="display:inline-block;width:{pip_w}%;height:7px;border-radius:4px;margin-right:3px;'
                    f'background:{"linear-gradient(90deg,#8b5cf6,#facc15)" if lvl >= i else "rgba(255,255,255,0.12)"};"></span>'
                    for i in range(1, mx + 1)
                )
                st.markdown(
                    f'<div style="display:flex;align-items:center;justify-content:space-between;font-size:12px;margin-bottom:7px;">'
                    f'<span style="width:82px;color:#cbd5e1;font-weight:600;">{EMOTION_DISPLAY[k]}</span>'
                    f'<span style="flex:1;padding:0 6px;">{pips}</span>'
                    f'<span style="min-width:78px;text-align:right;color:{"#facc15" if lvl > 0 else "#64748b"};font-weight:700;">{lbl}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        if item.get("ground_truth"):
            gt = item["ground_truth"]
            tags = [f"Sentiment: {gt['task_a']['label']}"] + [f"{EMOTION_DISPLAY[k]}: {gt['task_c'][k]['label']}" for k in EMOTION_KEYS]
            st.info("**Ground Truth:** " + " · ".join(tags))

        with st.expander("📝 Detected Text (OCR)", expanded=False):
            edited = st.text_area(
                "Edit caption to re-analyze",
                value=item.get("ocr", ""),
                key=f"ocr_edit_{item.get('title')}_{hashlib.md5((item.get('ocr','') or '').encode()).hexdigest()[:6]}",
                label_visibility="collapsed",
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Update Prediction with Edited Text", key="reanalyze_ocr_btn", use_container_width=True):
                    if item["source"] == "upload":
                        updated = analyze_bytes(predictor, item["img_bytes"], item["title"], ocr_override=edited)
                    else:
                        pred_new = predictor.predict(item["img_path"], edited)
                        updated = {**item, "ocr": edited, "prediction": {"task_a": pred_new["task_a"], "task_b": pred_new["task_b"], "task_c": pred_new["task_c"]}}
                    st.session_state.active_item = updated
                    st.rerun()
            with c2:
                if item["source"] == "upload" and st.button("Re-extract OCR", key="re_ocr_btn", use_container_width=True):
                    updated = analyze_bytes(predictor, item["img_bytes"], item["title"], ocr_override=None)
                    st.session_state.active_item = updated
                    st.rerun()


def main():
    init_state()
    inject_theme_css()
    predictor = load_predictor()

    logo_b64 = asset_b64(os.path.join("assets", "logo_troll.png"))
    logo_img = f'<img src="data:image/png;base64,{logo_b64}" alt="MemeSense"/>' if logo_b64 else "😎"
    st.markdown(
        f"""
        <div class="ms-navbar">
          <div class="ms-brand">{logo_img}<span>Meme<span class="ms-gold">Sense</span></span></div>
          <div style="font-size:13px;color:#cbd5e1;font-weight:600;">🇮🇳 Indian Memes · Memotion 3.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n1, n2, n3 = st.columns([1, 1, 1])
    for col, label in ((n1, "🏠 Home"), (n2, "🕒 History"), (n3, "ℹ️ About")):
        name = label.split(" ", 1)[1]
        with col:
            if st.button(label, key=f"nav_{name}", type="primary" if st.session_state.nav_tab == name else "secondary", use_container_width=True):
                st.session_state.nav_tab = name
                st.rerun()

    if st.session_state.nav_tab == "Home":
        hero_l = asset_b64(os.path.join("assets", "hero_left.png"))
        hero_r = asset_b64(os.path.join("assets", "hero_right.png"))
        hl, hc, hr = st.columns([1, 2.2, 1])
        with hl:
            if hero_l:
                st.markdown(f'<img src="data:image/png;base64,{hero_l}" style="width:100%;max-height:175px;object-fit:contain;" />', unsafe_allow_html=True)
        with hc:
            st.markdown(
                """
                <div class="ms-hero">
                  <div class="ms-hero-title">Indian <span class="ms-gold">Meme</span><br/>Sentiment <span class="ms-violet">Analysis</span></div>
                  <div class="ms-hero-sub">Because every meme <span style="border-bottom:3px solid #facc15;padding-bottom:2px;">has a mood!</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with hr:
            if hero_r:
                st.markdown(f'<img src="data:image/png;base64,{hero_r}" style="width:100%;max-height:175px;object-fit:contain;" />', unsafe_allow_html=True)

        has_active = st.session_state.active_item is not None
        if has_active:
            up_col, res_col = st.columns([0.9, 1.25], gap="large")
        else:
            _, up_col, _ = st.columns([0.7, 1.4, 0.7])
            res_col = None

        with up_col:
            st.markdown("#### 📤 Upload Your Meme")
            st.caption("Drag & drop an image or click Browse — analyzes automatically (JPG, PNG, WEBP)")
            uploaded = st.file_uploader(
                "Choose Meme Image",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"meme_uploader_{st.session_state.uploader_nonce}",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                raw = uploaded.getvalue()
                sig = hashlib.md5(raw).hexdigest()
                if sig != st.session_state.last_upload_sig:
                    with st.spinner("Extracting caption (OCR) & analyzing meme..."):
                        item = analyze_bytes(predictor, raw, uploaded.name)
                    st.session_state.last_upload_sig = sig
                    st.session_state.active_item = item
                    st.session_state.session_history.insert(0, item)
                    st.rerun()

        if has_active and res_col is not None:
            with res_col:
                render_result_panel(predictor, st.session_state.active_item)

        st.markdown("---")
        st.markdown("#### 🕒 Recent Analysis")
        featured = get_featured_samples()
        combined_recent = st.session_state.session_history[:5] + featured
        recent_5 = combined_recent[:5]
        cols = st.columns(max(len(recent_5), 1))
        for idx, rec in enumerate(recent_5):
            with cols[idx]:
                if rec.get("source") == "upload":
                    st.image(rec["img_bytes"], use_container_width=True)
                    sent = rec["prediction"]["task_a"]["label"]
                    st.markdown(f'<div style="text-align:center;"><span class="ms-badge ms-badge-{sent}">{sent}</span></div>', unsafe_allow_html=True)
                    if st.button("Inspect", key=f"recent_up_{idx}", use_container_width=True):
                        st.session_state.active_item = rec
                        st.rerun()
                else:
                    p = resolve_sample_image(rec["split"], rec["id"])
                    if p:
                        st.image(p, use_container_width=True)
                    sent = rec["prediction"]["task_a"]["label"]
                    st.markdown(f'<div style="text-align:center;"><span class="ms-badge ms-badge-{sent}">{sent}</span></div>', unsafe_allow_html=True)
                    if st.button(f"Load #{rec['id']}", key=f"recent_val_{rec['id']}_{idx}", use_container_width=True):
                        st.session_state.active_item = {
                            "source": "sample",
                            "title": f"{rec['split'].upper()} #{rec['id']}",
                            "img_path": p,
                            "ocr": rec.get("ocr", ""),
                            "prediction": rec["prediction"],
                            "ground_truth": rec.get("ground_truth"),
                        }
                        st.rerun()

    elif st.session_state.nav_tab == "History":
        st.subheader("History & Featured Memotion 3.0 Samples")
        if st.session_state.session_history:
            st.markdown("##### Your Uploaded Memes")
            hcols = st.columns(4)
            for i, h in enumerate(st.session_state.session_history):
                with hcols[i % 4]:
                    st.image(h["img_bytes"], use_container_width=True)
                    st.caption(f"{h['title']} · **{h['prediction']['task_a']['label']}**")
                    if st.button("Open", key=f"hist_open_{i}", use_container_width=True):
                        st.session_state.active_item = h
                        st.session_state.nav_tab = "Home"
                        st.rerun()
        else:
            st.info("No memes uploaded yet in this session. Upload one on Home or inspect featured dataset samples below.")

        st.markdown("##### Featured Validation Memes")
        samples = [r for r in load_split_records("val") if resolve_sample_image("val", r["id"])][:12]
        scols = st.columns(4)
        for i, rec in enumerate(samples):
            p = resolve_sample_image("val", rec["id"])
            with scols[i % 4]:
                st.image(p, use_container_width=True)
                st.caption(f"VAL #{rec['id']} · **{rec['prediction']['task_a']['label']}**")
                if st.button("Analyze", key=f"exp_open_{rec['id']}", use_container_width=True):
                    st.session_state.active_item = {
                        "source": "sample",
                        "title": f"VAL #{rec['id']}",
                        "img_path": p,
                        "ocr": rec.get("ocr", ""),
                        "prediction": rec["prediction"],
                        "ground_truth": rec.get("ground_truth"),
                    }
                    st.session_state.nav_tab = "Home"
                    st.rerun()

    else:
        st.subheader("About MemeSense & Validation Metrics")
        m = predictor.metrics
        s = m["summary"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Task A — Sentiment (Weighted F1)", f"{m['task_a']['weighted_f1']*100:.1f}%", f"Acc {m['task_a']['accuracy']*100:.1f}%")
        m2.metric("Task B — Emotions (Avg Weighted F1)", f"{s['task_b_avg_weighted_f1']*100:.1f}%")
        m3.metric("Task C — Intensity (Avg Weighted F1)", f"{s['task_c_avg_weighted_f1']*100:.1f}%")
        st.markdown(
            """
            - **Task A — Sentiment Analysis:** Classifies memes as Positive, Negative, or Neutral.
            - **Task B — Emotion Classification:** Multi-label detection across Humorous, Sarcastic, Offensive, and Motivational.
            - **Task C — Emotion Intensity Scales:** Quantifies intensity levels consistently with Task B.
            """
        )


if __name__ == "__main__":
    main()
