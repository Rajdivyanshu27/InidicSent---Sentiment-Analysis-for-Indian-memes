import os
import joblib
import numpy as np
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

        get_mobilenet()

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

    def _build_features(self, image_path: str, ocr_text: str):
        clean_text = clean_ocr_text(ocr_text)
        emb, cv_stat = extract_single_image_features(image_path)

        w_vec = self.bundle["word_vec"].transform([clean_text])
        c_vec = self.bundle["char_vec"].transform([clean_text])
        tfidf = normalize(hstack([w_vec, c_vec]).tocsr())

        t_svd = self.bundle["svd_scaler"].transform(self.bundle["text_svd"].transform(tfidf)).astype(np.float32)
        t_stat = self.bundle["tstat_scaler"].transform(extract_text_stats([clean_text])).astype(np.float32)
        x_t_nn = np.hstack([t_svd, t_stat]).astype(np.float32)

        emb_n = normalize(emb).astype(np.float32)
        v_pca = self.bundle["vis_pca_scaler"].transform(self.bundle["vis_pca"].transform(emb_n)).astype(np.float32)
        cv_s = self.bundle["cv_scaler"].transform(cv_stat).astype(np.float32)
        x_v_nn = np.hstack([emb_n, cv_s * 0.25]).astype(np.float32)

        x_lin = normalize(hstack([
            tfidf,
            csr_matrix(emb_n * 0.25),
            csr_matrix(normalize(t_stat) * 0.10),
            csr_matrix(normalize(cv_s) * 0.08),
        ]).tocsr())
        x_dense = np.hstack([t_svd, t_stat, v_pca, cv_s]).astype(np.float32)

        # Sequence tokens for TextCNN (new Task A schema)
        x_seq = None
        if self._task_a_new_schema and self._text_vec is not None:
            # text_vec is a Keras Model wrapping TextVectorization; call with string input
            x_seq = self._text_vec(np.array([[clean_text]])).numpy()  # (1, SEQ_LEN)

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
        p_cnn = cnn.predict([x_seq, emb_n], verbose=0)  # (1, 3)

        w = task_a_dict["lr_weight"]
        probs = (w * p_lr + (1.0 - w) * p_cnn)[0]
        pred = int(np.argmax(probs))
        return pred, probs

    def _score_model(self, model_dict, x_lin, x_t_nn, x_v_nn, x_dense):
        """Old schema: LR + XGB + NN blend."""
        w_lin, w_nn, w_xgb = model_dict["weights"]
        p_lin = model_dict["lin"].predict_proba(x_lin)
        p_xgb = model_dict["xgb"].predict_proba(x_dense)
        p_nn = self.nn_models[model_dict["nn_file"]].predict([x_t_nn, x_v_nn], verbose=0)
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
