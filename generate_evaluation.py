import json
import os
import time
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import normalize
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf

from constants import ARTIFACTS_DIR, EMOTION_KEYS, SENTIMENT_LABELS, TASK_C_SCALES
from features import extract_text_stats
from train_models import encode_targets, enforce_task_consistency, load_splits


# Pre-tuned ensemble weights from comparative_analysis.py (80/20 stratified holdout inside train)
TUNED_WEIGHTS = {
    "text_only": {
        "task_a": (0.6, 0.25, 0.15),
        "task_b": {
            "humour": (0.45, 0.45, 0.1),
            "sarcastic": (1.0, 0.0, 0.0),
            "offensive": (0.65, 0.15, 0.2),
            "motivational": (0.35, 0.5, 0.15),
        },
        "task_c": {
            "humour": (1.0, 0.0, 0.0),
            "sarcastic": (0.8, 0.1, 0.1),
            "offensive": (0.8, 0.1, 0.1),
            "motivational": (0.35, 0.5, 0.15),
        },
    },
    "image_only": {
        "task_a": (0.65, 0.15, 0.2),
        "task_b": {
            "humour": (1.0, 0.0, 0.0),
            "sarcastic": (1.0, 0.0, 0.0),
            "offensive": (0.7, 0.3, 0.0),
            "motivational": (1.0, 0.0, 0.0),
        },
        "task_c": {
            "humour": (1.0, 0.0, 0.0),
            "sarcastic": (0.65, 0.15, 0.2),
            "offensive": (0.35, 0.5, 0.15),
            "motivational": (1.0, 0.0, 0.0),
        },
    },
}


def compute_full_metrics(y_true, y_pred, labels: list[str]) -> dict:
    n_cls = len(labels)
    lbl_ids = list(range(n_cls))
    acc = float(accuracy_score(y_true, y_pred))
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=lbl_ids, average="weighted", zero_division=0
    )
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=lbl_ids, average="macro", zero_division=0
    )
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(
        y_true, y_pred, labels=lbl_ids, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=lbl_ids).tolist()
    per_class = [
        {
            "label": labels[i],
            "precision": round(float(p_c[i]), 4),
            "recall": round(float(r_c[i]), 4),
            "f1": round(float(f_c[i]), 4),
            "support": int(s_c[i]),
        }
        for i in range(n_cls)
    ]
    return {
        "accuracy": round(acc, 4),
        "precision_weighted": round(float(p_w), 4),
        "recall_weighted": round(float(r_w), 4),
        "f1_weighted": round(float(f_w), 4),
        "precision_macro": round(float(p_m), 4),
        "recall_macro": round(float(r_m), 4),
        "f1_macro": round(float(f_m), 4),
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def fit_single_branch_nn(X_tr, y_tr, n_classes: int, balanced: bool, dropout: float, seed: int = 100):
    cw_dict = None
    if balanced:
        classes = np.arange(n_classes)
        weights = compute_class_weight("balanced", classes=classes, y=y_tr)
        weights = np.clip(weights, 0.5, 4.0)
        cw_dict = {int(c): float(w) for c, w in zip(classes, weights)}

    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input(shape=(X_tr.shape[1],))
    h = tf.keras.layers.Dense(128, activation="gelu")(tf.keras.layers.Dropout(dropout)(inp))
    h = tf.keras.layers.Dense(96, activation="gelu")(tf.keras.layers.Dropout(0.35)(h))
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(h)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
    )
    model.fit(
        X_tr,
        y_tr,
        epochs=14,
        batch_size=128,
        validation_split=0.1,
        class_weight=cw_dict,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    )
    return model


def fit_and_eval_unimodal_target(
    X_lin_tr, X_lin_va,
    X_nn_tr, X_nn_va,
    X_dense_tr, X_dense_va,
    y_tr, n_classes: int, balanced: bool, dropout: float, weights: tuple[float, float, float],
):
    w_lin, w_nn, w_xgb = weights
    probs = np.zeros((X_lin_va.shape[0], n_classes), dtype=np.float32)

    if w_lin > 0.0:
        cw = "balanced" if balanced else None
        c_val = 0.35 if balanced else 1.0
        m_lin = LogisticRegression(C=c_val, max_iter=1000, class_weight=cw, random_state=42)
        m_lin.fit(X_lin_tr, y_tr)
        probs += w_lin * m_lin.predict_proba(X_lin_va)

    if w_nn > 0.0:
        m_nn = fit_single_branch_nn(X_nn_tr, y_tr, n_classes, balanced=balanced, dropout=dropout, seed=100)
        probs += w_nn * m_nn.predict(X_nn_va, verbose=0)
        tf.keras.backend.clear_session()

    if w_xgb > 0.0:
        if n_classes == 2:
            spw = 1.0
            if balanced:
                pos = max(int((y_tr == 1).sum()), 1)
                neg = max(int((y_tr == 0).sum()), 1)
                spw = min(max(neg / pos, 0.5), 4.0)
            m_xgb = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.06,
                subsample=0.85, colsample_bytree=0.75, reg_lambda=2.0,
                scale_pos_weight=spw, eval_metric="logloss",
                tree_method="hist", n_jobs=-1, random_state=42,
            )
        else:
            m_xgb = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.06,
                subsample=0.85, colsample_bytree=0.75, reg_lambda=2.0,
                objective="multi:softprob", num_class=n_classes,
                eval_metric="mlogloss", tree_method="hist", n_jobs=-1, random_state=42,
            )
        m_xgb.fit(X_dense_tr, y_tr)
        probs += w_xgb * m_xgb.predict_proba(X_dense_va)

    preds = probs.argmax(axis=1)
    return preds, probs


def summarize_mode(mode_eval: dict) -> dict:
    def avg(task_key: str, metric_key: str) -> float:
        return round(float(np.mean([mode_eval[task_key][k][metric_key] for k in EMOTION_KEYS])), 4)

    return {
        "task_a": {k: mode_eval["task_a"][k] for k in (
            "accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
            "precision_macro", "recall_macro", "f1_macro",
        )},
        "task_b_avg": {
            m: avg("task_b", m)
            for m in ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
                      "precision_macro", "recall_macro", "f1_macro")
        },
        "task_c_avg": {
            m: avg("task_c", m)
            for m in ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
                      "precision_macro", "recall_macro", "f1_macro")
        },
    }


def run_unimodal_eval(mode_name, feats, y_a_tr, y_b_tr, y_c_tr, y_a_va, y_b_va, y_c_va):
    print(f"Evaluating {mode_name}...", flush=True)
    t0 = time.time()
    X_lin_tr, X_lin_va, X_nn_tr, X_nn_va, X_dense_tr, X_dense_va, dropout = feats
    w_cfg = TUNED_WEIGHTS[mode_name]

    pa, _ = fit_and_eval_unimodal_target(
        X_lin_tr, X_lin_va, X_nn_tr, X_nn_va, X_dense_tr, X_dense_va,
        y_a_tr, 3, True, dropout, w_cfg["task_a"],
    )
    out = {"task_a": compute_full_metrics(y_a_va, pa, SENTIMENT_LABELS), "task_b": {}, "task_c": {}}

    balanced_cfg = {
        "task_b": {"humour": False, "sarcastic": False, "offensive": True, "motivational": False},
        "task_c": {"humour": False, "sarcastic": True, "offensive": True, "motivational": False},
    }
    preds_b, probs_b, preds_c, probs_c = {}, {}, {}, {}

    for k in EMOTION_KEYS:
        pb, pbp = fit_and_eval_unimodal_target(
            X_lin_tr, X_lin_va, X_nn_tr, X_nn_va, X_dense_tr, X_dense_va,
            y_b_tr[k], 2, balanced_cfg["task_b"][k], dropout, w_cfg["task_b"][k],
        )
        preds_b[k], probs_b[k] = pb, pbp
        out["task_b"][k] = compute_full_metrics(y_b_va[k], pb, ["Inactive", "Active"])

        if k == "motivational":
            preds_c[k], probs_c[k] = pb, pbp
        else:
            n_cls = len(TASK_C_SCALES[k]["labels"])
            pc, pcp = fit_and_eval_unimodal_target(
                X_lin_tr, X_lin_va, X_nn_tr, X_nn_va, X_dense_tr, X_dense_va,
                y_c_tr[k], n_cls, balanced_cfg["task_c"][k], dropout, w_cfg["task_c"][k],
            )
            preds_c[k], probs_c[k] = pc, pcp

    preds_c = enforce_task_consistency(preds_b, preds_c, probs_c)
    for k in EMOTION_KEYS:
        out["task_c"][k] = compute_full_metrics(y_c_va[k], preds_c[k], TASK_C_SCALES[k]["labels"])

    out["summary"] = summarize_mode(out)
    print(f"  {mode_name} done in {round(time.time() - t0, 1)}s", flush=True)
    return out


def eval_saved_multimodal() -> dict:
    val_path = os.path.join(ARTIFACTS_DIR, "val_predictions.json")
    with open(val_path, "r", encoding="utf-8") as f:
        recs = json.load(f)

    ya_t = np.array([r["ground_truth"]["task_a"]["label_id"] for r in recs], dtype=np.int64)
    ya_p = np.array([r["prediction"]["task_a"]["label_id"] for r in recs], dtype=np.int64)

    out = {"task_a": compute_full_metrics(ya_t, ya_p, SENTIMENT_LABELS), "task_b": {}, "task_c": {}}
    for k in EMOTION_KEYS:
        yb_t = np.array([1 if r["ground_truth"]["task_b"][k] else 0 for r in recs], dtype=np.int64)
        yb_p = np.array([1 if r["prediction"]["task_b"][k]["active"] else 0 for r in recs], dtype=np.int64)
        out["task_b"][k] = compute_full_metrics(yb_t, yb_p, ["Inactive", "Active"])

        yc_t = np.array([r["ground_truth"]["task_c"][k]["level"] for r in recs], dtype=np.int64)
        yc_p = np.array([r["prediction"]["task_c"][k]["level"] for r in recs], dtype=np.int64)
        out["task_c"][k] = compute_full_metrics(yc_t, yc_p, TASK_C_SCALES[k]["labels"])

    out["summary"] = summarize_mode(out)
    return out


def main():
    t0 = time.time()
    train_df, val_df, _ = load_splits()
    y_a_tr, y_b_tr, y_c_tr = encode_targets(train_df)
    y_a_va, y_b_va, y_c_va = encode_targets(val_df)

    cache = np.load(os.path.join(ARTIFACTS_DIR, "visual_cache.npz"))
    tr_emb, tr_cv = cache["train_emb"], cache["train_cv"]
    va_emb, va_cv = cache["val_emb"], cache["val_cv"]

    bundle = joblib.load(os.path.join(ARTIFACTS_DIR, "model_bundle.joblib"))
    word_vec = bundle["word_vec"]
    char_vec = bundle["char_vec"]
    text_svd = bundle["text_svd"]
    svd_scaler = bundle["svd_scaler"]
    tstat_scaler = bundle["tstat_scaler"]
    vis_pca = bundle["vis_pca"]
    vis_pca_scaler = bundle["vis_pca_scaler"]
    cv_scaler = bundle["cv_scaler"]

    X_tfidf_tr = normalize(hstack([word_vec.transform(train_df["ocr_clean"]), char_vec.transform(train_df["ocr_clean"])]).tocsr())
    X_tfidf_va = normalize(hstack([word_vec.transform(val_df["ocr_clean"]), char_vec.transform(val_df["ocr_clean"])]).tocsr())

    X_tsvd_tr = svd_scaler.transform(text_svd.transform(X_tfidf_tr)).astype(np.float32)
    X_tsvd_va = svd_scaler.transform(text_svd.transform(X_tfidf_va)).astype(np.float32)

    X_tstat_tr = tstat_scaler.transform(extract_text_stats(train_df["ocr_clean"])).astype(np.float32)
    X_tstat_va = tstat_scaler.transform(extract_text_stats(val_df["ocr_clean"])).astype(np.float32)

    X_t_lin_tr = normalize(hstack([X_tfidf_tr, csr_matrix(normalize(X_tstat_tr) * 0.10)]).tocsr())
    X_t_lin_va = normalize(hstack([X_tfidf_va, csr_matrix(normalize(X_tstat_va) * 0.10)]).tocsr())
    X_t_dense_tr = np.hstack([X_tsvd_tr, X_tstat_tr]).astype(np.float32)
    X_t_dense_va = np.hstack([X_tsvd_va, X_tstat_va]).astype(np.float32)

    tr_emb_n = normalize(tr_emb).astype(np.float32)
    va_emb_n = normalize(va_emb).astype(np.float32)

    X_vpca_tr = vis_pca_scaler.transform(vis_pca.transform(tr_emb_n)).astype(np.float32)
    X_vpca_va = vis_pca_scaler.transform(vis_pca.transform(va_emb_n)).astype(np.float32)

    X_cv_tr = cv_scaler.transform(tr_cv).astype(np.float32)
    X_cv_va = cv_scaler.transform(va_cv).astype(np.float32)

    X_v_lin_tr = normalize(hstack([csr_matrix(tr_emb_n * 0.25), csr_matrix(normalize(X_cv_tr) * 0.08)]).tocsr())
    X_v_lin_va = normalize(hstack([csr_matrix(va_emb_n * 0.25), csr_matrix(normalize(X_cv_va) * 0.08)]).tocsr())
    X_v_nn_tr = np.hstack([tr_emb_n, X_cv_tr * 0.25]).astype(np.float32)
    X_v_nn_va = np.hstack([va_emb_n, X_cv_va * 0.25]).astype(np.float32)
    X_v_dense_tr = np.hstack([X_vpca_tr, X_cv_tr]).astype(np.float32)
    X_v_dense_va = np.hstack([X_vpca_va, X_cv_va]).astype(np.float32)

    text_eval = run_unimodal_eval(
        "text_only",
        (X_t_lin_tr, X_t_lin_va, X_t_dense_tr, X_t_dense_va, X_t_dense_tr, X_t_dense_va, 0.35),
        y_a_tr, y_b_tr, y_c_tr, y_a_va, y_b_va, y_c_va,
    )
    image_eval = run_unimodal_eval(
        "image_only",
        (X_v_lin_tr, X_v_lin_va, X_v_nn_tr, X_v_nn_va, X_v_dense_tr, X_v_dense_va, 0.45),
        y_a_tr, y_b_tr, y_c_tr, y_a_va, y_b_va, y_c_va,
    )
    multimodal_eval = eval_saved_multimodal()

    report = {
        "val_count": int(len(val_df)),
        "train_count": int(len(train_df)),
        "modalities": {
            "text_only": text_eval,
            "image_only": image_eval,
            "multimodal": multimodal_eval,
        },
        "weights_used": TUNED_WEIGHTS,
    }
    out_path = os.path.join(ARTIFACTS_DIR, "evaluation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to {out_path} in {round(time.time() - t0, 1)}s", flush=True)


if __name__ == "__main__":
    main()
