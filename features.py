import os
import re
import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

POS_WORDS = {
    "love", "happy", "best", "good", "great", "awesome", "amazing", "cute",
    "thank", "thanks", "win", "winner", "success", "inspire", "dream", "hope",
    "smile", "wholesome", "proud", "blessed", "khush", "mast", "badiya", "pyaar",
}
NEG_WORDS = {
    "hate", "bad", "worst", "kill", "die", "dead", "stupid", "idiot", "cringe",
    "ugly", "angry", "sad", "pain", "cry", "shit", "fuck", "bitch", "ass",
    "trash", "hell", "wtf", "bkwas", "ganda", "bewakoof", "chutiya", "mar",
}
WATERMARK_WORDS = ("mematic", "rvcj", "imgflip", "memegenerator", "9gag", "sarcasm")

_MOBILENET = None


def get_mobilenet():
    """Lazy-load MobileNetV2 feature extractor once per process."""
    global _MOBILENET
    if _MOBILENET is None:
        import tensorflow as tf
        _MOBILENET = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(160, 160, 3),
        )
    return _MOBILENET


def clean_ocr_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_text_stats(texts) -> np.ndarray:
    """Extract 14 compact lexical and Hinglish/meme features per text."""
    rows = []
    for raw in texts:
        t = clean_ocr_text(raw)
        n_char = len(t)
        words = re.findall(r"\w+", t.lower())
        n_words = len(words)
        avg_w = (sum(len(w) for w in words) / n_words) if n_words else 0.0
        n_upper = sum(1 for c in t if c.isupper())
        n_alpha = sum(1 for c in t if c.isalpha())
        upper_ratio = (n_upper / n_alpha) if n_alpha else 0.0
        n_digit = sum(1 for c in t if c.isdigit())
        digit_ratio = (n_digit / n_char) if n_char else 0.0
        exclam = t.count("!")
        quest = t.count("?")
        newlines = t.count("\n")
        social = t.count("@") + t.count("#")
        quotes = t.count('"') + t.count("'")
        devanagari = sum(1 for c in t if "\u0900" <= c <= "\u097F")
        dev_ratio = (devanagari / n_char) if n_char else 0.0
        t_lower = t.lower()
        wm = sum(1 for w in WATERMARK_WORDS if w in t_lower)
        pos_c = sum(1 for w in words if w in POS_WORDS)
        neg_c = sum(1 for w in words if w in NEG_WORDS)
        rows.append([
            np.log1p(n_char),
            np.log1p(n_words),
            avg_w,
            upper_ratio,
            digit_ratio,
            min(exclam, 10),
            min(quest, 10),
            min(newlines, 20),
            min(social, 10),
            min(quotes, 10),
            dev_ratio,
            min(wm, 5),
            min(pos_c, 10),
            min(neg_c, 10),
        ])
    return np.asarray(rows, dtype=np.float32)


def extract_cv_stats_single(img: Image.Image) -> np.ndarray:
    """Extract 18 fast visual aesthetics and meme-layout features from a PIL image."""
    w, h = img.size
    aspect = float(w) / max(float(h), 1.0)
    log_area = float(np.log1p(w * h))

    rgb = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
    r_mean, g_mean, b_mean = rgb.mean(axis=(0, 1))
    r_std, g_std, b_std = rgb.std(axis=(0, 1))

    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    bright_mean = float(gray.mean())
    bright_std = float(gray.std())
    white_frac = float((gray > 0.90).mean())
    dark_frac = float((gray < 0.12).mean())
    top_white = float((gray[:24, :] > 0.88).mean())
    bot_white = float((gray[-24:, :] > 0.88).mean())

    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge_density = float(gx + gy)

    c_max = rgb.max(axis=2)
    c_min = rgb.min(axis=2)
    sat = np.where(c_max > 1e-5, (c_max - c_min) / np.maximum(c_max, 1e-5), 0.0)
    sat_mean = float(sat.mean())
    sat_std = float(sat.std())
    warm_ratio = float(r_mean - b_mean)

    return np.asarray([
        aspect, log_area,
        r_mean, g_mean, b_mean,
        r_std, g_std, b_std,
        bright_mean, bright_std,
        white_frac, dark_frac,
        top_white, bot_white,
        edge_density, sat_mean, sat_std, warm_ratio,
    ], dtype=np.float32)


def extract_visual_features_batch(image_paths, batch_size: int = 128):
    """Extract MobileNetV2 (1280-d) embeddings and CV stats (18-d) for a list of image paths."""
    import tensorflow as tf

    model = get_mobilenet()
    n = len(image_paths)
    emb_list = []
    cv_list = []

    for start in range(0, n, batch_size):
        chunk = image_paths[start:start + batch_size]
        arrs = []
        for p in chunk:
            try:
                with Image.open(p) as im:
                    im_rgb = im.convert("RGB")
                    cv_list.append(extract_cv_stats_single(im_rgb))
                    im_resized = im_rgb.resize((160, 160))
                    arrs.append(np.asarray(im_resized, dtype=np.float32))
            except Exception:
                cv_list.append(np.zeros(18, dtype=np.float32))
                arrs.append(np.zeros((160, 160, 3), dtype=np.float32))

        batch_x = tf.keras.applications.mobilenet_v2.preprocess_input(np.stack(arrs))
        feats = model.predict(batch_x, batch_size=len(arrs), verbose=0)
        emb_list.append(feats.astype(np.float32))
        if (start // batch_size) % 10 == 0:
            print(f"  Visual features: {min(start + batch_size, n)}/{n}", flush=True)

    return np.vstack(emb_list), np.vstack(cv_list)


def extract_single_image_features(image_path: str):
    """Extract (1, 1280) MobileNetV2 embedding and (1, 18) CV stats for one image."""
    return extract_visual_features_batch([image_path], batch_size=1)
