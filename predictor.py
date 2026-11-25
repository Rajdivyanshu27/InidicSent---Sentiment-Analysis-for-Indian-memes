import base64
import io
import os
import re
import cv2
import joblib
import numpy as np
from PIL import Image
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf

from constants import ARTIFACTS_DIR, EMOTION_KEYS, SENTIMENT_LABELS, TASK_C_SCALES
from features import clean_ocr_text, extract_single_image_features, extract_text_stats, get_mobilenet

SEQ_LEN = 48


class MemotionPredictor:
    """Loads the trained multimodal bundle + Keras NNs and scores meme image + OCR text.

    Supports both:
    - Old Task A schema: {"lin", "xgb", "nn_file", "weights", "n_classes"}
    - New Task A schema: {"lr", "cnn_file", "lr_weight", "has_aux"}
    """

    def __init__(self, bundle_path: str | None = None):
        path = bundle_path or os.path.join(ARTIFACTS_DIR, "model_bundle.joblib")
        self.bundle = joblib.load(path)
        self.metrics = self.bundle["metrics"]
        self.nn_models = {}
        nn_dir = os.path.join(ARTIFACTS_DIR, "keras_models")

        models = self.bundle["models"]
        task_a = models["task_a"]

        # Detect schema: new schema has "has_aux" key
        self._task_a_new_schema = bool(task_a.get("has_aux"))

        if self._task_a_new_schema:
            # New schema: LR + TextCNN blend
            self._load_nn(task_a["cnn_file"], nn_dir)
            # Load TextVectorization layer saved as separate file
            tv_path = os.path.join(nn_dir, "text_vec.keras")
            if os.path.isfile(tv_path):
                self._text_vec = tf.keras.models.load_model(tv_path)
            else:
                # Fallback: reconstruct from bundle config/weights if stored
                self._text_vec = self._rebuild_text_vec()
            # Load aux LR models for each emotion (stored in bundle)
            self._aux_lr_models = self.bundle.get("aux_lr_models", None)
        else:
            # Old schema: LR + XGB + NN ensemble
            self._load_nn(task_a["nn_file"], nn_dir)
            self._text_vec = None
            self._aux_lr_models = None

        for k in EMOTION_KEYS:
            self._load_nn(models["task_b"][k]["nn_file"], nn_dir)
            if k != "motivational":
                self._load_nn(models["task_c"][k]["nn_file"], nn_dir)

        mobilenet = get_mobilenet()
        self._mobilenet_conv = tf.keras.Model(
            inputs=mobilenet.input,
            outputs=mobilenet.get_layer("out_relu").output,
        )

    def _rebuild_text_vec(self):
        """Reconstruct TextVectorization from bundle config+weights if saved there."""
        cfg = self.bundle.get("text_vec_config")
        wts = self.bundle.get("text_vec_weights")
        if cfg is None or wts is None:
            return None
        vec = tf.keras.layers.TextVectorization.from_config(cfg)
        vec.set_weights(wts)
        return vec

    def _load_nn(self, filename: str, nn_dir: str):
        if filename not in self.nn_models:
            self.nn_models[filename] = tf.keras.models.load_model(os.path.join(nn_dir, filename))

    def _build_visual_parts(self, image_path: str):
        emb, cv_stat = extract_single_image_features(image_path)
        return self._transform_visual_parts(emb, cv_stat)

    def _transform_visual_parts(self, emb: np.ndarray, cv_stat: np.ndarray):
        emb_n = normalize(emb).astype(np.float32)
        v_pca = self.bundle["vis_pca_scaler"].transform(self.bundle["vis_pca"].transform(emb_n)).astype(np.float32)
        cv_s = self.bundle["cv_scaler"].transform(cv_stat).astype(np.float32)
        x_v_nn = np.hstack([emb_n, cv_s * 0.25]).astype(np.float32)
        return emb_n, v_pca, cv_s, x_v_nn

    def _build_text_parts(self, clean_text: str, emb_n: np.ndarray, v_pca: np.ndarray, cv_s: np.ndarray):
        w_vec = self.bundle["word_vec"].transform([clean_text])
        c_vec = self.bundle["char_vec"].transform([clean_text])
        tfidf = normalize(hstack([w_vec, c_vec]).tocsr())

        t_svd = self.bundle["svd_scaler"].transform(self.bundle["text_svd"].transform(tfidf)).astype(np.float32)
        t_stat = self.bundle["tstat_scaler"].transform(extract_text_stats([clean_text])).astype(np.float32)
        x_t_nn = np.hstack([t_svd, t_stat]).astype(np.float32)

        x_lin = normalize(hstack([
            tfidf,
            csr_matrix(emb_n * 0.25),
            csr_matrix(normalize(t_stat) * 0.10),
            csr_matrix(normalize(cv_s) * 0.08),
        ]).tocsr())
        x_dense = np.hstack([t_svd, t_stat, v_pca, cv_s]).astype(np.float32)

        x_seq = None
        if self._task_a_new_schema and self._text_vec is not None:
            x_seq = self._text_vec(np.array([[clean_text]])).numpy()

        return x_lin, x_t_nn, x_dense, x_seq, tfidf.shape[1]

    def _build_features(self, image_path: str, ocr_text: str):
        clean_text = clean_ocr_text(ocr_text)
        emb_n, v_pca, cv_s, x_v_nn = self._build_visual_parts(image_path)
        x_lin, x_t_nn, x_dense, x_seq, _ = self._build_text_parts(clean_text, emb_n, v_pca, cv_s)
        return clean_text, x_lin, x_t_nn, x_v_nn, x_dense, emb_n, x_seq

    def _build_aux_probs(self, x_lin) -> np.ndarray:
        """Build 8-d auxiliary emotion probs using aux_lr_models in bundle."""
        parts = []
        aux_lrs = self.bundle.get("aux_lr_models", {})
        for k in EMOTION_KEYS:
            if k in aux_lrs:
                probs = aux_lrs[k].predict_proba(x_lin)  # (1, 2)
            else:
                probs = np.array([[0.5, 0.5]], dtype=np.float32)
            parts.append(probs)
        return np.hstack(parts).astype(np.float32)  # (1, 8)

    def _score_task_a_new(self, task_a_dict, x_lin, emb_n, x_seq):
        """New Task A: LR+Aux stacked + TextCNN blend."""
        aux = self._build_aux_probs(x_lin)  # (1, 8)
        X_s = normalize(hstack([x_lin, csr_matrix(aux * 0.5)]))
        p_lr = task_a_dict["lr"].predict_proba(X_s)  # (1, 3)

        cnn = self.nn_models[task_a_dict["cnn_file"]]
        p_cnn = cnn([x_seq, emb_n], training=False).numpy()  # (1, 3)

        w = task_a_dict["lr_weight"]
        probs = (w * p_lr + (1.0 - w) * p_cnn)[0]
        pred = int(np.argmax(probs))
        return pred, probs

    def _score_model(self, model_dict, x_lin, x_t_nn, x_v_nn, x_dense):
        """Old schema: LR + XGB + NN blend."""
        w_lin, w_nn, w_xgb = model_dict["weights"]
        p_lin = model_dict["lin"].predict_proba(x_lin)
        p_xgb = model_dict["xgb"].predict_proba(x_dense)
        p_nn = self.nn_models[model_dict["nn_file"]]([x_t_nn, x_v_nn], training=False).numpy()
        probs = (w_lin * p_lin + w_nn * p_nn + w_xgb * p_xgb)[0]
        pred = int(np.argmax(probs))
        return pred, probs

    def predict(self, image_path: str, ocr_text: str) -> dict:
        clean_text, x_lin, x_t_nn, x_v_nn, x_dense, emb_n, x_seq = self._build_features(
            image_path, ocr_text
        )
        models = self.bundle["models"]

        # Task A
        if self._task_a_new_schema:
            pred_a, prob_a = self._score_task_a_new(models["task_a"], x_lin, emb_n, x_seq)
        else:
            pred_a, prob_a = self._score_model(models["task_a"], x_lin, x_t_nn, x_v_nn, x_dense)

        task_b = {}
        task_c = {}
        for k in EMOTION_KEYS:
            pb, pbp = self._score_model(models["task_b"][k], x_lin, x_t_nn, x_v_nn, x_dense)
            task_b[k] = {
                "active": bool(pb == 1),
                "prob": round(float(pbp[1]), 4),
            }
            if k == "motivational":
                pc, pcp = pb, pbp
            else:
                pc, pcp = self._score_model(models["task_c"][k], x_lin, x_t_nn, x_v_nn, x_dense)
                if pb == 0:
                    pc = 0
                elif pc == 0:
                    pc = 1 + int(np.argmax(pcp[1:]))

            task_c[k] = {
                "level": int(pc),
                "label": TASK_C_SCALES[k]["labels"][int(pc)],
                "max_level": len(TASK_C_SCALES[k]["labels"]) - 1,
                "probs": [round(float(x), 4) for x in pcp],
            }

        return {
            "ocr_used": clean_text,
            "task_a": {
                "label_id": int(pred_a),
                "label": SENTIMENT_LABELS[int(pred_a)],
                "probs": [round(float(x), 4) for x in prob_a],
            },
            "task_b": task_b,
            "task_c": task_c,
        }

    def _score_task_a_parts(self, clean_text: str, emb_n: np.ndarray, v_pca: np.ndarray, cv_s: np.ndarray, x_v_nn: np.ndarray):
        x_lin, x_t_nn, x_dense, x_seq, tfidf_dim = self._build_text_parts(clean_text, emb_n, v_pca, cv_s)
        task_a = self.bundle["models"]["task_a"]
        if self._task_a_new_schema:
            pred_a, prob_a = self._score_task_a_new(task_a, x_lin, emb_n, x_seq)
        else:
            pred_a, prob_a = self._score_model(task_a, x_lin, x_t_nn, x_v_nn, x_dense)
        return pred_a, prob_a, x_t_nn, x_seq, tfidf_dim

    def _compute_word_attributions(
        self,
        clean_text: str,
        target_cls: int,
        base_prob: float,
        emb_n: np.ndarray,
        v_pca: np.ndarray,
        cv_s: np.ndarray,
        x_v_nn: np.ndarray,
    ):
        """Leave-one-word-out perturbation attribution across the full Task A multimodal ensemble."""
        if not clean_text.strip():
            return [], []

        parts = re.findall(r"\w+|[^\w\s]+|\s+", clean_text)
        unique_words = []
        seen = set()
        for p in parts:
            if re.fullmatch(r"\w+", p) and len(p) >= 2:
                low = p.lower()
                if low not in seen:
                    seen.add(low)
                    unique_words.append(low)

        # Evaluate up to 24 unique words via leave-one-out on the full multimodal ensemble
        word_scores: dict[str, float] = {}
        for w in unique_words[:24]:
            masked_parts = [
                "" if (re.fullmatch(r"\w+", tok) and tok.lower() == w) else tok
                for tok in parts
            ]
            masked_text = clean_ocr_text("".join(masked_parts))
            _, m_probs, _, _, _ = self._score_task_a_parts(masked_text, emb_n, v_pca, cv_s, x_v_nn)
            word_scores[w] = float(base_prob - float(m_probs[target_cls]))

        tokens = []
        for p in parts:
            is_word = bool(re.fullmatch(r"\w+", p))
            score = word_scores.get(p.lower(), 0.0) if is_word else 0.0
            tokens.append({"text": p, "score": round(score, 5), "is_word": is_word})

        ranked = sorted(word_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_words = [
            {
                "word": w,
                "score": round(sc, 5),
                "direction": "supporting" if sc >= 0 else "opposing",
            }
            for w, sc in ranked[:10]
            if abs(sc) > 1e-5
        ]
        return tokens, top_words

    def _compute_gradcam(
        self,
        image_path: str,
        target_cls: int,
        cv_s: np.ndarray,
        x_t_nn: np.ndarray,
        x_seq: np.ndarray | None,
        tfidf_dim: int,
    ):
        """Compute spatial Grad-CAM heatmap over MobileNetV2 out_relu for Task A sentiment."""
        with Image.open(image_path) as im:
            im_rgb = im.convert("RGB")
            orig_w, orig_h = im_rgb.size
            orig_np = np.asarray(im_rgb, dtype=np.uint8)
            resized = np.asarray(im_rgb.resize((160, 160)), dtype=np.float32)

        batch_x = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(resized, axis=0))
        img_tensor = tf.constant(batch_x, dtype=tf.float32)
        cv_scaled_t = tf.constant(cv_s * 0.25, dtype=tf.float32)
        task_a = self.bundle["models"]["task_a"]

        with tf.GradientTape() as tape:
            conv_out = self._mobilenet_conv(img_tensor, training=False)
            tape.watch(conv_out)
            gap = tf.reduce_mean(conv_out, axis=(1, 2))
            emb_n_t = tf.math.l2_normalize(gap, axis=-1)

            if self._task_a_new_schema:
                cnn = self.nn_models[task_a["cnn_file"]]
                p_cnn = cnn([tf.constant(x_seq), emb_n_t], training=False)[0, target_cls]
                lin_coef = task_a["lr"].coef_[target_cls, tfidf_dim:tfidf_dim + 1280] * 0.25
                lin_score = tf.reduce_sum(emb_n_t[0] * tf.constant(lin_coef, dtype=tf.float32))
                w_lr = float(task_a["lr_weight"])
                score = (1.0 - w_lr) * p_cnn + w_lr * lin_score
            else:
                w_lin, w_nn, _ = task_a["weights"]
                nn_model = self.nn_models[task_a["nn_file"]]
                vis_in_t = tf.concat([emb_n_t, cv_scaled_t], axis=-1)
                p_nn = nn_model([tf.constant(x_t_nn, dtype=tf.float32), vis_in_t], training=False)[0, target_cls]
                lin_coef = task_a["lin"].coef_[target_cls, tfidf_dim:tfidf_dim + 1280] * 0.25
                lin_score = tf.reduce_sum(emb_n_t[0] * tf.constant(lin_coef, dtype=tf.float32))
                score = float(w_nn) * p_nn + float(w_lin) * lin_score

        grads = tape.gradient(score, conv_out)
        if grads is None:
            cam = np.ones((5, 5), dtype=np.float32)
        else:
            alpha = tf.reduce_mean(grads, axis=(0, 1, 2))
            raw_cam = tf.reduce_sum(conv_out[0] * alpha, axis=-1)
            cam = tf.nn.relu(raw_cam).numpy()
            if float(cam.max()) <= 1e-8:
                cam = np.abs(raw_cam.numpy())

        cam_max = float(cam.max())
        if cam_max > 1e-8:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        # 3x3 spatial region summary
        cam_3x3 = cv2.resize(cam, (3, 3), interpolation=cv2.INTER_AREA)
        region_names = [
            ["Top-Left", "Top-Center", "Top-Right"],
            ["Middle-Left", "Center", "Middle-Right"],
            ["Bottom-Left", "Bottom-Center", "Bottom-Right"],
        ]
        regions = []
        for r in range(3):
            for c in range(3):
                regions.append({
                    "region": region_names[r][c],
                    "importance": round(float(cam_3x3[r, c]), 4),
                })
        regions.sort(key=lambda x: x["importance"], reverse=True)

        # Smooth resize to original meme dimensions and blend heatmap
        cam_up = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        cam_up = np.clip(cam_up, 0.0, 1.0)
        heat_bgr = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        alpha_map = (0.18 + 0.42 * cam_up)[:, :, None]
        blended = np.uint8(np.clip((1.0 - alpha_map) * orig_np + alpha_map * heat_rgb, 0, 255))

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        return heatmap_b64, regions

    def explain(self, image_path: str, ocr_text: str) -> dict:
        """Compute multimodal XAI (Grad-CAM visual regions + word perturbation + modality share)."""
        clean_text = clean_ocr_text(ocr_text)
        emb_n, v_pca, cv_s, x_v_nn = self._build_visual_parts(image_path)
        pred_a, prob_a, x_t_nn, x_seq, tfidf_dim = self._score_task_a_parts(
            clean_text, emb_n, v_pca, cv_s, x_v_nn
        )
        base_prob = float(prob_a[pred_a])

        # Word-level attribution
        tokens, top_words = self._compute_word_attributions(
            clean_text, pred_a, base_prob, emb_n, v_pca, cv_s, x_v_nn
        )

        # Visual Grad-CAM
        heatmap_url, regions = self._compute_gradcam(
            image_path, pred_a, cv_s, x_t_nn, x_seq, tfidf_dim
        )

        # Modality contribution via single-modality ablation on this meme
        _, probs_no_text, _, _, _ = self._score_task_a_parts("", emb_n, v_pca, cv_s, x_v_nn)
        z_emb = np.zeros_like(emb_n)
        z_pca = np.zeros_like(v_pca)
        z_cv = np.zeros_like(cv_s)
        z_v_nn = np.zeros_like(x_v_nn)
        _, probs_no_img, _, _, _ = self._score_task_a_parts(clean_text, z_emb, z_pca, z_cv, z_v_nn)

        text_delta = abs(base_prob - float(probs_no_text[pred_a]))
        img_delta = abs(base_prob - float(probs_no_img[pred_a]))
        total_delta = text_delta + img_delta
        if total_delta > 1e-6:
            text_share = text_delta / total_delta
            img_share = img_delta / total_delta
        else:
            text_share, img_share = 0.5, 0.5

        return {
            "target_label": SENTIMENT_LABELS[int(pred_a)],
            "target_prob": round(base_prob, 4),
            "methods": {
                "text": "Leave-One-Word-Out Perturbation (Multimodal Ensemble)",
                "image": "Grad-CAM (MobileNetV2 out_relu + Multimodal Task A)",
                "modality": "Modality Occlusion Attribution",
            },
            "modality_contribution": {
                "text_pct": round(text_share * 100, 1),
                "image_pct": round(img_share * 100, 1),
                "no_text_prob": round(float(probs_no_text[pred_a]), 4),
                "no_image_prob": round(float(probs_no_img[pred_a]), 4),
            },
            "text_xai": {
                "tokens": tokens,
                "top_words": top_words,
            },
            "image_xai": {
                "heatmap_data_url": heatmap_url,
                "top_regions": regions[:4],
                "all_regions": regions,
            },
        }
