import json
import os
import time
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.utils.class_weight import compute_class_weight

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf

from constants import ARTIFACTS_DIR, EMOTION_KEYS, SENTIMENT_LABELS
from features import extract_text_stats
from generate_evaluation import compute_full_metrics, summarize_mode
from train_models import encode_targets, evaluate_predictions, load_splits, write_answer_file

SEQ_LEN = 48
MAX_TOK = 15000


def build_image_only_nn(vis_dim: int, seed: int = 42):
    tf.keras.utils.set_random_seed(seed)
    in_v = tf.keras.Input(shape=(vis_dim,), name="vis_in")
    hv = tf.keras.layers.Dense(128, activation="gelu")(tf.keras.layers.Dropout(0.45)(in_v))
    h = tf.keras.layers.Dense(96, activation="gelu")(tf.keras.layers.Dropout(0.35)(hv))
    out = tf.keras.layers.Dense(3, activation="softmax")(h)
    model = tf.keras.Model(in_v, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_text_only_cnn(seed: int = 42):
    tf.keras.utils.set_random_seed(seed)
    in_seq = tf.keras.Input(shape=(SEQ_LEN,), dtype="int32", name="seq_in")
    emb = tf.keras.layers.Embedding(MAX_TOK, 64, mask_zero=False)(in_seq)
    c3 = tf.keras.layers.GlobalMaxPooling1D()(tf.keras.layers.Conv1D(96, 3, activation="relu", padding="same")(emb))
    c4 = tf.keras.layers.GlobalMaxPooling1D()(tf.keras.layers.Conv1D(96, 4, activation="relu", padding="same")(emb))
    c5 = tf.keras.layers.GlobalMaxPooling1D()(tf.keras.layers.Conv1D(64, 5, activation="relu", padding="same")(emb))
    ht = tf.keras.layers.Dropout(0.40)(
        tf.keras.layers.Dense(128, activation="gelu")(tf.keras.layers.Concatenate()([c3, c4, c5]))
    )
    h = tf.keras.layers.Dense(96, activation="gelu")(tf.keras.layers.Dropout(0.35)(ht))
    out = tf.keras.layers.Dense(3, activation="softmax")(h)
    model = tf.keras.Model(in_seq, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_multimodal_textcnn(vis_dim: int, seed: int = 42):
    tf.keras.utils.set_random_seed(seed)
    in_seq = tf.keras.Input(shape=(SEQ_LEN,), dtype="int32", name="seq_in")
    in_vis = tf.keras.Input(shape=(vis_dim,), name="vis_in")
    emb = tf.keras.layers.Embedding(MAX_TOK, 64, mask_zero=False)(in_seq)
    c3 = tf.keras.layers.GlobalMaxPooling1D()(tf.keras.layers.Conv1D(96, 3, activation="relu", padding="same")(emb))
    c4 = tf.keras.layers.GlobalMaxPooling1D()(tf.keras.layers.Conv1D(96, 4, activation="relu", padding="same")(emb))
    c5 = tf.keras.layers.GlobalMaxPooling1D()(tf.keras.layers.Conv1D(64, 5, activation="relu", padding="same")(emb))
    ht = tf.keras.layers.Dropout(0.40)(
        tf.keras.layers.Dense(128, activation="gelu")(tf.keras.layers.Concatenate()([c3, c4, c5]))
    )
    hv = tf.keras.layers.Dense(64, activation="gelu")(tf.keras.layers.Dropout(0.5)(in_vis))
    h = tf.keras.layers.Concatenate()([ht, hv])
    h = tf.keras.layers.Dense(96, activation="gelu")(tf.keras.layers.Dropout(0.35)(h))
    out = tf.keras.layers.Dense(3, activation="softmax")(h)
    model = tf.keras.Model([in_seq, in_vis], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def format_history(hist) -> dict:
    h = hist.history
    n_ep = len(h["loss"])
    return {
        "epochs": list(range(1, n_ep + 1)),
        "train_loss": [round(float(x), 4) for x in h["loss"]],
        "val_loss": [round(float(x), 4) for x in h["val_loss"]],
        "train_accuracy": [round(float(x), 4) for x in h["accuracy"]],
        "val_accuracy": [round(float(x), 4) for x in h["val_accuracy"]],
    }


def main():
    t0 = time.time()
    train_df, val_df, test_df = load_splits()
    y_a_tr, y_b_tr, _ = encode_targets(train_df)
    y_a_va, _, _ = encode_targets(val_df)

    cache = np.load(os.path.join(ARTIFACTS_DIR, "visual_cache.npz"))
    tr_emb_n = normalize(cache["train_emb"]).astype(np.float32)
    va_emb_n = normalize(cache["val_emb"]).astype(np.float32)
    te_emb_n = normalize(cache["test_emb"]).astype(np.float32)

    bundle_path = os.path.join(ARTIFACTS_DIR, "model_bundle.joblib")
    bundle = joblib.load(bundle_path)

    X_tfidf_tr = normalize(hstack([bundle["word_vec"].transform(train_df["ocr_clean"]), bundle["char_vec"].transform(train_df["ocr_clean"])]).tocsr())
    X_tfidf_va = normalize(hstack([bundle["word_vec"].transform(val_df["ocr_clean"]), bundle["char_vec"].transform(val_df["ocr_clean"])]).tocsr())
    X_tfidf_te = normalize(hstack([bundle["word_vec"].transform(test_df["ocr_clean"]), bundle["char_vec"].transform(test_df["ocr_clean"])]).tocsr())

    X_tstat_tr = bundle["tstat_scaler"].transform(extract_text_stats(train_df["ocr_clean"])).astype(np.float32)
    X_tstat_va = bundle["tstat_scaler"].transform(extract_text_stats(val_df["ocr_clean"])).astype(np.float32)
    X_tstat_te = bundle["tstat_scaler"].transform(extract_text_stats(test_df["ocr_clean"])).astype(np.float32)

    X_cv_tr = bundle["cv_scaler"].transform(cache["train_cv"]).astype(np.float32)
    X_cv_va = bundle["cv_scaler"].transform(cache["val_cv"]).astype(np.float32)
    X_cv_te = bundle["cv_scaler"].transform(cache["test_cv"]).astype(np.float32)

    X_lin_tr = normalize(hstack([X_tfidf_tr, csr_matrix(tr_emb_n * 0.25), csr_matrix(normalize(X_tstat_tr) * 0.10), csr_matrix(normalize(X_cv_tr) * 0.08)]).tocsr())
    X_lin_va = normalize(hstack([X_tfidf_va, csr_matrix(va_emb_n * 0.25), csr_matrix(normalize(X_tstat_va) * 0.10), csr_matrix(normalize(X_cv_va) * 0.08)]).tocsr())
    X_lin_te = normalize(hstack([X_tfidf_te, csr_matrix(te_emb_n * 0.25), csr_matrix(normalize(X_tstat_te) * 0.10), csr_matrix(normalize(X_cv_te) * 0.08)]).tocsr())

    # 1. Fit auxiliary emotion LR models for Task A stacking (matching predictor.py new schema)
    print("Fitting auxiliary emotion LR models...", flush=True)
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aux_tr_parts, aux_va_parts, aux_te_parts = [], [], []
    aux_lr_models = {}
    for k in EMOTION_KEYS:
        yk = y_b_tr[k]
        oof_k = np.zeros((len(train_df), 2), dtype=np.float32)
        for ti, vi in skf5.split(X_lin_tr, yk):
            mk = LogisticRegression(C=0.5, max_iter=800, class_weight="balanced", random_state=42)
            mk.fit(X_lin_tr[ti], yk[ti])
            oof_k[vi] = mk.predict_proba(X_lin_tr[vi])
        mk_full = LogisticRegression(C=0.5, max_iter=800, class_weight="balanced", random_state=42)
        mk_full.fit(X_lin_tr, yk)
        aux_lr_models[k] = mk_full
        aux_tr_parts.append(oof_k)
        aux_va_parts.append(mk_full.predict_proba(X_lin_va))
        aux_te_parts.append(mk_full.predict_proba(X_lin_te))

    aux_tr = np.hstack(aux_tr_parts).astype(np.float32)
    aux_va = np.hstack(aux_va_parts).astype(np.float32)
    aux_te = np.hstack(aux_te_parts).astype(np.float32)

    X_s_tr = normalize(hstack([X_lin_tr, csr_matrix(aux_tr * 0.5)]))
    X_s_va = normalize(hstack([X_lin_va, csr_matrix(aux_va * 0.5)]))
    X_s_te = normalize(hstack([X_lin_te, csr_matrix(aux_te * 0.5)]))

    final_lr = LogisticRegression(C=0.35, max_iter=1000, class_weight="balanced", random_state=42)
    final_lr.fit(X_s_tr, y_a_tr)

    # 2. Load saved TextVectorization + TextCNN+Visual for Task A inference
    nn_dir = os.path.join(ARTIFACTS_DIR, "keras_models")
    tv_model = tf.keras.models.load_model(os.path.join(nn_dir, "text_vec.keras"))
    cnn_model = tf.keras.models.load_model(os.path.join(nn_dir, "task_a_cnn.keras"))

    x_seq_tr = tv_model(np.array([[t] for t in train_df["ocr_clean"]])).numpy()
    x_seq_va = tv_model(np.array([[t] for t in val_df["ocr_clean"]])).numpy()
    x_seq_te = tv_model(np.array([[t] for t in test_df["ocr_clean"]])).numpy()

    p_lr_va = final_lr.predict_proba(X_s_va)
    p_lr_te = final_lr.predict_proba(X_s_te)
    p_cnn_va = cnn_model([x_seq_va, va_emb_n], training=False).numpy()
    p_cnn_te = cnn_model([x_seq_te, te_emb_n], training=False).numpy()

    # Use subtle LR+Aux + TextCNN+Vis blend (0.10 LR + 0.90 TextCNN+Vis) and compare vs 0.0
    for w_lr in [0.0, 0.05, 0.10, 0.15]:
        pr = (w_lr * p_lr_va + (1.0 - w_lr) * p_cnn_va).argmax(1)
        print(f"  Task A blend w_lr={w_lr:.2f}: {evaluate_predictions(y_a_va, pr)}", flush=True)

    lr_weight = 0.05
    prob_a_va = lr_weight * p_lr_va + (1.0 - lr_weight) * p_cnn_va
    prob_a_te = lr_weight * p_lr_te + (1.0 - lr_weight) * p_cnn_te
    pred_a_va = prob_a_va.argmax(1)
    pred_a_te = prob_a_te.argmax(1)

    task_a_metrics = evaluate_predictions(y_a_va, pred_a_va)
    print("Selected Multimodal Task A:", task_a_metrics, flush=True)

    # 3. Update bundle, metrics.json, val/test predictions, answer files
    bundle["aux_lr_models"] = aux_lr_models
    bundle["models"]["task_a"] = {
        "lr": final_lr,
        "cnn_file": "task_a_cnn.keras",
        "lr_weight": lr_weight,
        "has_aux": True,
    }
    bundle["metrics"]["task_a"] = task_a_metrics
    bundle["metrics"]["summary"]["task_a_weighted_f1"] = task_a_metrics["weighted_f1"]
    joblib.dump(bundle, bundle_path, compress=3)

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(bundle["metrics"], f, indent=2)

    def update_predictions_json(json_name, pa, pba):
        p = os.path.join(ARTIFACTS_DIR, json_name)
        with open(p, "r", encoding="utf-8") as f:
            recs = json.load(f)
        for idx, r in enumerate(recs):
            r["prediction"]["task_a"] = {
                "label_id": int(pa[idx]),
                "label": SENTIMENT_LABELS[int(pa[idx])],
                "probs": [round(float(x), 4) for x in pba[idx]],
            }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(recs, f)
        return recs

    val_recs = update_predictions_json("val_predictions.json", pred_a_va, prob_a_va)
    test_recs = update_predictions_json("test_predictions.json", pred_a_te, prob_a_te)

    preds_b_va = {k: np.array([1 if r["prediction"]["task_b"][k]["active"] else 0 for r in val_recs]) for k in EMOTION_KEYS}
    preds_c_va = {k: np.array([r["prediction"]["task_c"][k]["level"] for r in val_recs]) for k in EMOTION_KEYS}
    preds_b_te = {k: np.array([1 if r["prediction"]["task_b"][k]["active"] else 0 for r in test_recs]) for k in EMOTION_KEYS}
    preds_c_te = {k: np.array([r["prediction"]["task_c"][k]["level"] for r in test_recs]) for k in EMOTION_KEYS}

    write_answer_file(os.path.join(ARTIFACTS_DIR, "val_answer.txt"), pred_a_va, preds_b_va, preds_c_va)
    write_answer_file(os.path.join(ARTIFACTS_DIR, "test_answer.txt"), pred_a_te, preds_b_te, preds_c_te)

    # 4. Generate 12-epoch training curves for Image-only, Text-only, and Text+Image Multimodal
    print("Recording 12-epoch training curves (Image, Text, Multimodal)...", flush=True)
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_a_tr)
    cw = np.clip(cw, 0.5, 4.0)
    cw_dict = {i: float(cw[i]) for i in range(3)}

    x_v_nn_tr = np.hstack([tr_emb_n, X_cv_tr * 0.25]).astype(np.float32)

    img_nn = build_image_only_nn(x_v_nn_tr.shape[1], seed=42)
    h_img = img_nn.fit(x_v_nn_tr, y_a_tr, epochs=12, batch_size=128, validation_split=0.1, class_weight=cw_dict, verbose=0)
    tf.keras.backend.clear_session()

    txt_nn = build_text_only_cnn(seed=42)
    h_txt = txt_nn.fit(x_seq_tr, y_a_tr, epochs=12, batch_size=128, validation_split=0.1, class_weight=cw_dict, verbose=0)
    tf.keras.backend.clear_session()

    mm_nn = build_multimodal_textcnn(tr_emb_n.shape[1], seed=42)
    h_mm = mm_nn.fit([x_seq_tr, tr_emb_n], y_a_tr, epochs=12, batch_size=128, validation_split=0.1, class_weight=cw_dict, verbose=0)
    tf.keras.backend.clear_session()

    curves = {
        "epochs": list(range(1, 13)),
        "image_only": format_history(h_img),
        "text_only": format_history(h_txt),
        "multimodal": format_history(h_mm),
    }
    with open(os.path.join(ARTIFACTS_DIR, "training_curves.json"), "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)

    # 5. Update evaluation_report.json with upgraded Multimodal Task A + training_curves
    eval_path = os.path.join(ARTIFACTS_DIR, "evaluation_report.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    report["modalities"]["multimodal"]["task_a"] = compute_full_metrics(y_a_va, pred_a_va, SENTIMENT_LABELS)
    report["modalities"]["multimodal"]["summary"] = summarize_mode(report["modalities"]["multimodal"])
    report["training_curves"] = curves

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Completed Task A upgrade & training curves in {round(time.time() - t0, 1)}s", flush=True)


if __name__ == "__main__":
    main()
