import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf

from constants import (
    ARTIFACTS_DIR,
    EMOTION_KEYS,
    IMAGE_DIRS,
    SENTIMENT_LABELS,
    SENTIMENT_RAW_TO_ID,
    TASK_B_NEGATIVE_RAW,
    TASK_C_SCALES,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
)
from features import clean_ocr_text, extract_text_stats, extract_visual_features_batch


def load_splits():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV, sep="\t")
    test_df = pd.read_csv(TEST_CSV)
    for df in (train_df, val_df, test_df):
        df["id"] = df.iloc[:, 0].astype(int)
        df["ocr_clean"] = df["ocr"].fillna("").map(clean_ocr_text)
    return train_df, val_df, test_df


def get_or_extract_visual_cache(train_df, val_df, test_df):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    cache_path = os.path.join(ARTIFACTS_DIR, "visual_cache.npz")
    if os.path.isfile(cache_path):
        print("Loading cached visual features:", cache_path, flush=True)
        data = np.load(cache_path)
        return (
            data["train_emb"], data["train_cv"],
            data["val_emb"], data["val_cv"],
            data["test_emb"], data["test_cv"],
        )

    def paths_for(df, split):
        d = IMAGE_DIRS[split]
        return [os.path.join(d, f"{i}.jpg") for i in df["id"]]

    tr_emb, tr_cv = extract_visual_features_batch(paths_for(train_df, "train"))
    va_emb, va_cv = extract_visual_features_batch(paths_for(val_df, "val"))
    te_emb, te_cv = extract_visual_features_batch(paths_for(test_df, "test"))

    np.savez_compressed(
        cache_path,
        train_emb=tr_emb, train_cv=tr_cv,
        val_emb=va_emb, val_cv=va_cv,
        test_emb=te_emb, test_cv=te_cv,
    )
    return tr_emb, tr_cv, va_emb, va_cv, te_emb, te_cv


def encode_targets(df):
    y_a = df["overall"].map(SENTIMENT_RAW_TO_ID).to_numpy(dtype=np.int64)
    y_b = {}
    y_c = {}
    for k in EMOTION_KEYS:
        neg_val = TASK_B_NEGATIVE_RAW[k]
        y_b[k] = (df[k] != neg_val).astype(np.int64).to_numpy()
        y_c[k] = df[k].map(TASK_C_SCALES[k]["raw_to_id"]).to_numpy(dtype=np.int64)
    return y_a, y_b, y_c


SEQ_LEN = 48
MAX_TOK = 15000


def build_text_vectorizer(texts):
    """Fit a Keras TextVectorization layer on training texts."""
    vec = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOK,
        output_mode="int",
        output_sequence_length=SEQ_LEN,
    )
    vec.adapt(texts)
    return vec


def build_task_a_textcnn(vis_dim: int, seed: int = 42):
    """TextCNN + Visual embedding dual-branch network for Task A sentiment."""
    tf.keras.utils.set_random_seed(seed)
    in_seq = tf.keras.Input(shape=(SEQ_LEN,), dtype="int32", name="seq_in")
    in_vis = tf.keras.Input(shape=(vis_dim,), name="vis_in")

    emb = tf.keras.layers.Embedding(MAX_TOK, 64, mask_zero=False)(in_seq)
    c3 = tf.keras.layers.GlobalMaxPooling1D()(
        tf.keras.layers.Conv1D(96, 3, activation="relu", padding="same")(emb)
    )
    c4 = tf.keras.layers.GlobalMaxPooling1D()(
        tf.keras.layers.Conv1D(96, 4, activation="relu", padding="same")(emb)
    )
    c5 = tf.keras.layers.GlobalMaxPooling1D()(
        tf.keras.layers.Conv1D(64, 5, activation="relu", padding="same")(emb)
    )
    ht = tf.keras.layers.Dropout(0.40)(
        tf.keras.layers.Dense(128, activation="gelu")(
            tf.keras.layers.Concatenate()([c3, c4, c5])
        )
    )
    hv = tf.keras.layers.Dense(64, activation="gelu")(
        tf.keras.layers.Dropout(0.5)(in_vis)
    )
    h = tf.keras.layers.Concatenate()([ht, hv])
    h = tf.keras.layers.Dense(96, activation="gelu")(tf.keras.layers.Dropout(0.35)(h))
    out = tf.keras.layers.Dense(3, activation="softmax")(h)
    model = tf.keras.Model([in_seq, in_vis], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss="sparse_categorical_crossentropy",
    )
    return model


def fit_task_a_textcnn(X_seq_tr, X_vis_tr, y_tr, seed: int = 42):
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_tr)
    cw = np.clip(cw, 0.5, 4.0)
    model = build_task_a_textcnn(X_vis_tr.shape[1], seed=seed)
    model.fit(
        [X_seq_tr, X_vis_tr],
        y_tr,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        class_weight={i: float(cw[i]) for i in range(3)},
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    )
    return model


def train_task_a_enhanced(
    target_name: str,
    X_lin_tr, X_lin_va,
    X_seq_tr, X_seq_va,
    X_vis_tr, X_vis_va,
    X_dense_tr, X_dense_va,
    aux_tr, aux_va,
    y_tr, y_va,
):
    """
    Enhanced Task A training:
    - 5-fold OOF Logistic Regression on (lin + aux) stacked features
    - 5-fold OOF TextCNN + Visual
    - Final blend weight tuned on OOF to maximise val weighted-F1
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n = len(y_tr)
    oof_lr = np.zeros((n, 3), dtype=np.float32)
    oof_cnn = np.zeros((n, 3), dtype=np.float32)

    from scipy.sparse import issparse

    for fold, (ti, vi) in enumerate(skf.split(X_dense_tr, y_tr)):
        # LR on stacked features
        X_s_tr = normalize(hstack([X_lin_tr[ti], csr_matrix(aux_tr[ti] * 0.5)]))
        X_s_vi = normalize(hstack([X_lin_tr[vi], csr_matrix(aux_tr[vi] * 0.5)]))
        lr_m = LogisticRegression(C=0.15, max_iter=1000, class_weight="balanced", random_state=42)
        lr_m.fit(X_s_tr, y_tr[ti])
        oof_lr[vi] = lr_m.predict_proba(X_s_vi)

        # TextCNN
        cnn_m = fit_task_a_textcnn(X_seq_tr[ti], X_vis_tr[ti], y_tr[ti], seed=42 + fold)
        oof_cnn[vi] = cnn_m.predict([X_seq_tr[vi], X_vis_tr[vi]], verbose=0)
        tf.keras.backend.clear_session()

    # tune blend on OOF
    best_w, best_f1 = 0.75, -1.0
    for w in np.linspace(0.4, 1.0, 25):
        blend = w * oof_lr + (1.0 - w) * oof_cnn
        p = blend.argmax(axis=1)
        sc = f1_score(y_tr, p, average="weighted", zero_division=0)
        if sc > best_f1:
            best_f1 = sc
            best_w = float(w)

    print(f"  Task A OOF tuned: lr_weight={best_w:.2f} cnn_weight={1-best_w:.2f} oof_wf1={best_f1:.4f}", flush=True)

    # Train final models on full train
    X_s_full = normalize(hstack([X_lin_tr, csr_matrix(aux_tr * 0.5)]))
    final_lr = LogisticRegression(C=0.15, max_iter=1000, class_weight="balanced", random_state=42)
    final_lr.fit(X_s_full, y_tr)

    final_cnn = fit_task_a_textcnn(X_seq_tr, X_vis_tr, y_tr, seed=99)
    nn_dir = os.path.join(ARTIFACTS_DIR, "keras_models")
    os.makedirs(nn_dir, exist_ok=True)
    cnn_file = f"{target_name}_cnn.keras"
    final_cnn.save(os.path.join(nn_dir, cnn_file))
    tf.keras.backend.clear_session()

    return {
        "lr": final_lr,
        "cnn_file": cnn_file,
        "lr_weight": best_w,
        "has_aux": True,
    }


def predict_task_a(model_dict, cnn_model, X_lin, aux, X_seq, X_vis):
    X_s = normalize(hstack([X_lin, csr_matrix(aux * 0.5)]))
    p_lr = model_dict["lr"].predict_proba(X_s)
    p_cnn = cnn_model.predict([X_seq, X_vis], verbose=0)
    w = model_dict["lr_weight"]
    probs = w * p_lr + (1.0 - w) * p_cnn
    preds = probs.argmax(axis=1)
    return preds, probs


def build_keras_multimodal_nn(text_dim: int, vis_dim: int, n_classes: int, seed: int = 42):
    tf.keras.utils.set_random_seed(seed)
    in_t = tf.keras.Input(shape=(text_dim,), name="text_in")
    in_v = tf.keras.Input(shape=(vis_dim,), name="vis_in")
    ht = tf.keras.layers.Dense(128, activation="gelu")(tf.keras.layers.Dropout(0.35)(in_t))
    hv = tf.keras.layers.Dense(128, activation="gelu")(tf.keras.layers.Dropout(0.45)(in_v))
    h = tf.keras.layers.Concatenate()([ht, hv])
    h = tf.keras.layers.Dense(96, activation="gelu")(tf.keras.layers.Dropout(0.35)(h))
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(h)
    model = tf.keras.Model([in_t, in_v], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
    )
    return model


def fit_keras_nn(X_t_tr, X_v_tr, y_tr, n_classes: int, balanced: bool, seed: int = 42):
    cw_dict = None
    if balanced:
        classes = np.arange(n_classes)
        weights = compute_class_weight("balanced", classes=classes, y=y_tr)
        weights = np.clip(weights, 0.5, 4.0)
        cw_dict = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_keras_multimodal_nn(X_t_tr.shape[1], X_v_tr.shape[1], n_classes, seed=seed)
    model.fit(
        [X_t_tr, X_v_tr],
        y_tr,
        epochs=14,
        batch_size=128,
        validation_split=0.1,
        class_weight=cw_dict,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    )
    return model


def train_multimodal_target(
    target_name: str,
    X_lin_tr, X_t_tr, X_v_tr, X_dense_tr, y_tr,
    n_classes: int,
    balanced: bool,
):
    """Train a 3-model multimodal ensemble: Sparse Linear + Dual-Branch Keras NN + XGBoost."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lin = np.zeros((len(y_tr), n_classes), dtype=np.float32)
    oof_nn = np.zeros((len(y_tr), n_classes), dtype=np.float32)
    oof_xgb = np.zeros((len(y_tr), n_classes), dtype=np.float32)

    cw = "balanced" if balanced else None
    c_val = 0.35 if balanced else 1.0

    def make_lin():
        return LogisticRegression(C=c_val, max_iter=1000, class_weight=cw, random_state=42)

    def make_xgb():
        if n_classes == 2:
            spw = 1.0
            if balanced:
                pos = max(int((y_tr == 1).sum()), 1)
                neg = max(int((y_tr == 0).sum()), 1)
                spw = min(max(neg / pos, 0.5), 4.0)
            return XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.06,
                subsample=0.85, colsample_bytree=0.75, reg_lambda=2.0,
                scale_pos_weight=spw, eval_metric="logloss",
                tree_method="hist", n_jobs=4, random_state=42,
            )
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.75, reg_lambda=2.0,
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", tree_method="hist", n_jobs=4, random_state=42,
        )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_dense_tr, y_tr)):
        m_lin = make_lin().fit(X_lin_tr[tr_idx], y_tr[tr_idx])
        oof_lin[va_idx] = m_lin.predict_proba(X_lin_tr[va_idx])

        m_xgb = make_xgb().fit(X_dense_tr[tr_idx], y_tr[tr_idx])
        oof_xgb[va_idx] = m_xgb.predict_proba(X_dense_tr[va_idx])

        m_nn = fit_keras_nn(X_t_tr[tr_idx], X_v_tr[tr_idx], y_tr[tr_idx], n_classes, balanced=balanced, seed=42 + fold)
        oof_nn[va_idx] = m_nn.predict([X_t_tr[va_idx], X_v_tr[va_idx]], verbose=0)
        tf.keras.backend.clear_session()

    best_weights = (0.55, 0.30, 0.15)
    best_f1 = -1.0
    candidates = [
        (1.0, 0.0, 0.0), (0.9, 0.1, 0.0), (0.85, 0.15, 0.0),
        (0.7, 0.3, 0.0), (0.6, 0.25, 0.15), (0.5, 0.35, 0.15),
        (0.45, 0.45, 0.10), (0.8, 0.1, 0.1), (0.35, 0.5, 0.15),
        (0.65, 0.15, 0.20),
    ]
    for w_lin, w_nn, w_xgb in candidates:
        blend = w_lin * oof_lin + w_nn * oof_nn + w_xgb * oof_xgb
        preds = blend.argmax(axis=1)
        sc = f1_score(y_tr, preds, average="weighted", zero_division=0)
        if sc > best_f1:
            best_f1 = sc
            best_weights = (w_lin, w_nn, w_xgb)

    final_lin = make_lin().fit(X_lin_tr, y_tr)
    final_xgb = make_xgb().fit(X_dense_tr, y_tr)
    final_nn = fit_keras_nn(X_t_tr, X_v_tr, y_tr, n_classes, balanced=balanced, seed=100)

    nn_dir = os.path.join(ARTIFACTS_DIR, "keras_models")
    os.makedirs(nn_dir, exist_ok=True)
    nn_filename = f"{target_name}.keras"
    final_nn.save(os.path.join(nn_dir, nn_filename))
    tf.keras.backend.clear_session()

    return {
        "lin": final_lin,
        "xgb": final_xgb,
        "nn_file": nn_filename,
        "weights": best_weights,
        "n_classes": int(n_classes),
    }


def predict_ensemble(model_dict, nn_model, X_lin, X_t, X_v, X_dense):
    w_lin, w_nn, w_xgb = model_dict["weights"]
    p_lin = model_dict["lin"].predict_proba(X_lin)
    p_xgb = model_dict["xgb"].predict_proba(X_dense)
    p_nn = nn_model.predict([X_t, X_v], verbose=0)
    probs = w_lin * p_lin + w_nn * p_nn + w_xgb * p_xgb
    preds = probs.argmax(axis=1)
    return preds, probs


def enforce_task_consistency(preds_b, preds_c, probs_c):
    """Ensure Task B binary activity and Task C intensity levels never contradict."""
    consistent_c = {}
    for k in EMOTION_KEYS:
        if k == "motivational":
            consistent_c[k] = preds_b[k].copy()
            continue
        arr_c = preds_c[k].copy()
        for i in range(len(arr_c)):
            if preds_b[k][i] == 0:
                arr_c[i] = 0
            elif arr_c[i] == 0:
                arr_c[i] = 1 + int(np.argmax(probs_c[k][i, 1:]))
        consistent_c[k] = arr_c
    return consistent_c


def evaluate_predictions(y_true, preds):
    return {
        "weighted_f1": round(float(f1_score(y_true, preds, average="weighted", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
    }


def write_answer_file(out_path, pred_a, preds_b, preds_c):
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(pred_a)):
            a_str = str(int(pred_a[i]))
            b_str = "".join(str(int(preds_b[k][i])) for k in EMOTION_KEYS)
            c_str = "".join(str(int(preds_c[k][i])) for k in EMOTION_KEYS)
            f.write(f"{a_str}_{b_str}_{c_str}\n")


def main():
    t0 = time.time()
    train_df, val_df, test_df = load_splits()
    tr_emb, tr_cv, va_emb, va_cv, te_emb, te_cv = get_or_extract_visual_cache(train_df, val_df, test_df)

    print("Fitting multimodal feature transformers on train split...", flush=True)
    word_vec = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=50000, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2, max_features=60000, sublinear_tf=True)

    X_tfidf_tr = normalize(hstack([word_vec.fit_transform(train_df["ocr_clean"]), char_vec.fit_transform(train_df["ocr_clean"])]).tocsr())
    X_tfidf_va = normalize(hstack([word_vec.transform(val_df["ocr_clean"]), char_vec.transform(val_df["ocr_clean"])]).tocsr())
    X_tfidf_te = normalize(hstack([word_vec.transform(test_df["ocr_clean"]), char_vec.transform(test_df["ocr_clean"])]).tocsr())

    text_svd = TruncatedSVD(n_components=256, random_state=42)
    svd_scaler = StandardScaler()
    X_tsvd_tr = svd_scaler.fit_transform(text_svd.fit_transform(X_tfidf_tr)).astype(np.float32)
    X_tsvd_va = svd_scaler.transform(text_svd.transform(X_tfidf_va)).astype(np.float32)
    X_tsvd_te = svd_scaler.transform(text_svd.transform(X_tfidf_te)).astype(np.float32)

    tstat_scaler = StandardScaler()
    X_tstat_tr = tstat_scaler.fit_transform(extract_text_stats(train_df["ocr_clean"])).astype(np.float32)
    X_tstat_va = tstat_scaler.transform(extract_text_stats(val_df["ocr_clean"])).astype(np.float32)
    X_tstat_te = tstat_scaler.transform(extract_text_stats(test_df["ocr_clean"])).astype(np.float32)

    X_t_nn_tr = np.hstack([X_tsvd_tr, X_tstat_tr]).astype(np.float32)
    X_t_nn_va = np.hstack([X_tsvd_va, X_tstat_va]).astype(np.float32)
    X_t_nn_te = np.hstack([X_tsvd_te, X_tstat_te]).astype(np.float32)

    tr_emb_n = normalize(tr_emb).astype(np.float32)
    va_emb_n = normalize(va_emb).astype(np.float32)
    te_emb_n = normalize(te_emb).astype(np.float32)

    vis_pca = PCA(n_components=128, random_state=42)
    vis_pca_scaler = StandardScaler()
    X_vpca_tr = vis_pca_scaler.fit_transform(vis_pca.fit_transform(tr_emb_n)).astype(np.float32)
    X_vpca_va = vis_pca_scaler.transform(vis_pca.transform(va_emb_n)).astype(np.float32)
    X_vpca_te = vis_pca_scaler.transform(vis_pca.transform(te_emb_n)).astype(np.float32)

    cv_scaler = StandardScaler()
    X_cv_tr = cv_scaler.fit_transform(tr_cv).astype(np.float32)
    X_cv_va = cv_scaler.transform(va_cv).astype(np.float32)
    X_cv_te = cv_scaler.transform(te_cv).astype(np.float32)

    X_v_nn_tr = np.hstack([tr_emb_n, X_cv_tr * 0.25]).astype(np.float32)
    X_v_nn_va = np.hstack([va_emb_n, X_cv_va * 0.25]).astype(np.float32)
    X_v_nn_te = np.hstack([te_emb_n, X_cv_te * 0.25]).astype(np.float32)

    X_lin_tr = normalize(hstack([X_tfidf_tr, csr_matrix(tr_emb_n * 0.25), csr_matrix(normalize(X_tstat_tr) * 0.10), csr_matrix(normalize(X_cv_tr) * 0.08)]).tocsr())
    X_lin_va = normalize(hstack([X_tfidf_va, csr_matrix(va_emb_n * 0.25), csr_matrix(normalize(X_tstat_va) * 0.10), csr_matrix(normalize(X_cv_va) * 0.08)]).tocsr())
    X_lin_te = normalize(hstack([X_tfidf_te, csr_matrix(te_emb_n * 0.25), csr_matrix(normalize(X_tstat_te) * 0.10), csr_matrix(normalize(X_cv_te) * 0.08)]).tocsr())

    X_dense_tr = np.hstack([X_tsvd_tr, X_tstat_tr, X_vpca_tr, X_cv_tr]).astype(np.float32)
    X_dense_va = np.hstack([X_tsvd_va, X_tstat_va, X_vpca_va, X_cv_va]).astype(np.float32)
    X_dense_te = np.hstack([X_tsvd_te, X_tstat_te, X_vpca_te, X_cv_te]).astype(np.float32)

    y_a_tr, y_b_tr, y_c_tr = encode_targets(train_df)
    y_a_va, y_b_va, y_c_va = encode_targets(val_df)

    # --- Auxiliary emotion OOF probs for Task A stacking ---
    print("\nBuilding auxiliary emotion OOF probs for Task A stacking...", flush=True)
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aux_parts_tr, aux_parts_va, aux_parts_te = [], [], []
    aux_lr_models = {}  # saved to bundle for inference
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
        aux_parts_tr.append(oof_k)
        aux_parts_va.append(mk_full.predict_proba(X_lin_va))
        aux_parts_te.append(mk_full.predict_proba(X_lin_te))
    aux_a_tr = np.hstack(aux_parts_tr).astype(np.float32)
    aux_a_va = np.hstack(aux_parts_va).astype(np.float32)
    aux_a_te = np.hstack(aux_parts_te).astype(np.float32)

    # --- TextVectorization for TextCNN branch ---
    print("Fitting TextVectorizer for TextCNN...", flush=True)
    text_vec_layer = build_text_vectorizer(train_df["ocr_clean"].to_numpy())
    X_seq_tr = text_vec_layer(train_df["ocr_clean"].to_numpy()).numpy()
    X_seq_va = text_vec_layer(val_df["ocr_clean"].to_numpy()).numpy()
    X_seq_te = text_vec_layer(test_df["ocr_clean"].to_numpy()).numpy()
    # Save TextVec as proper Keras model for reliable inference
    nn_dir_early = os.path.join(ARTIFACTS_DIR, "keras_models")
    os.makedirs(nn_dir_early, exist_ok=True)
    # Wrap in a model so it can be saved/loaded
    tv_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="tv_input")
    tv_out = text_vec_layer(tv_input)
    tv_model = tf.keras.Model(tv_input, tv_out, name="text_vec_model")
    tv_model.save(os.path.join(nn_dir_early, "text_vec.keras"))
    print("  TextVec model saved.", flush=True)

    # Use balanced weighting when minority/majority balance benefits F1 without collapsing skewed priors
    balanced_cfg = {
        "task_b": {"humour": False, "sarcastic": False, "offensive": True, "motivational": False},
        "task_c": {"humour": False, "sarcastic": True, "offensive": True, "motivational": False},
    }

    models = {"task_a": None, "task_b": {}, "task_c": {}}
    metrics = {"task_a": {}, "task_b": {}, "task_c": {}}

    nn_dir = os.path.join(ARTIFACTS_DIR, "keras_models")

    print("\nTraining Task A (Enhanced: LR+AuxStack + TextCNN blend)...", flush=True)
    models["task_a"] = train_task_a_enhanced(
        "task_a",
        X_lin_tr, X_lin_va,
        X_seq_tr, X_seq_va,
        tr_emb_n, va_emb_n,
        X_dense_tr, X_dense_va,
        aux_a_tr, aux_a_va,
        y_a_tr, y_a_va,
    )
    cnn_a = tf.keras.models.load_model(os.path.join(nn_dir, models["task_a"]["cnn_file"]))
    pred_a_va, prob_a_va = predict_task_a(models["task_a"], cnn_a, X_lin_va, aux_a_va, X_seq_va, va_emb_n)
    pred_a_te, prob_a_te = predict_task_a(models["task_a"], cnn_a, X_lin_te, aux_a_te, X_seq_te, te_emb_n)
    metrics["task_a"] = evaluate_predictions(y_a_va, pred_a_va)
    print("  Val Task A:", metrics["task_a"], flush=True)

    preds_b_va, probs_b_va, preds_b_te, probs_b_te = {}, {}, {}, {}
    preds_c_va, probs_c_va, preds_c_te, probs_c_te = {}, {}, {}, {}

    for k in EMOTION_KEYS:
        print(f"\nTraining Task B ({k})...", flush=True)
        models["task_b"][k] = train_multimodal_target(
            f"task_b_{k}", X_lin_tr, X_t_nn_tr, X_v_nn_tr, X_dense_tr, y_b_tr[k],
            n_classes=2, balanced=balanced_cfg["task_b"][k]
        )
        nn_b = tf.keras.models.load_model(os.path.join(nn_dir, models["task_b"][k]["nn_file"]))
        preds_b_va[k], probs_b_va[k] = predict_ensemble(models["task_b"][k], nn_b, X_lin_va, X_t_nn_va, X_v_nn_va, X_dense_va)
        preds_b_te[k], probs_b_te[k] = predict_ensemble(models["task_b"][k], nn_b, X_lin_te, X_t_nn_te, X_v_nn_te, X_dense_te)
        metrics["task_b"][k] = evaluate_predictions(y_b_va[k], preds_b_va[k])
        print(f"  Val Task B ({k}):", metrics["task_b"][k], "weights=", models["task_b"][k]["weights"], flush=True)

        if k == "motivational":
            models["task_c"][k] = models["task_b"][k]
            preds_c_va[k], probs_c_va[k] = preds_b_va[k], probs_b_va[k]
            preds_c_te[k], probs_c_te[k] = preds_b_te[k], probs_b_te[k]
        else:
            print(f"Training Task C ({k} intensity)...", flush=True)
            n_cls = len(TASK_C_SCALES[k]["labels"])
            models["task_c"][k] = train_multimodal_target(
                f"task_c_{k}", X_lin_tr, X_t_nn_tr, X_v_nn_tr, X_dense_tr, y_c_tr[k],
                n_classes=n_cls, balanced=balanced_cfg["task_c"][k]
            )
            nn_c = tf.keras.models.load_model(os.path.join(nn_dir, models["task_c"][k]["nn_file"]))
            preds_c_va[k], probs_c_va[k] = predict_ensemble(models["task_c"][k], nn_c, X_lin_va, X_t_nn_va, X_v_nn_va, X_dense_va)
            preds_c_te[k], probs_c_te[k] = predict_ensemble(models["task_c"][k], nn_c, X_lin_te, X_t_nn_te, X_v_nn_te, X_dense_te)

    preds_c_va = enforce_task_consistency(preds_b_va, preds_c_va, probs_c_va)
    preds_c_te = enforce_task_consistency(preds_b_te, preds_c_te, probs_c_te)

    for k in EMOTION_KEYS:
        metrics["task_c"][k] = evaluate_predictions(y_c_va[k], preds_c_va[k])
        print(f"  Val Task C ({k} consistent):", metrics["task_c"][k], flush=True)

    metrics["summary"] = {
        "task_a_weighted_f1": metrics["task_a"]["weighted_f1"],
        "task_b_avg_weighted_f1": round(float(np.mean([metrics["task_b"][k]["weighted_f1"] for k in EMOTION_KEYS])), 4),
        "task_c_avg_weighted_f1": round(float(np.mean([metrics["task_c"][k]["weighted_f1"] for k in EMOTION_KEYS])), 4),
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "test_count": int(len(test_df)),
    }

    bundle = {
        "word_vec": word_vec,
        "char_vec": char_vec,
        "text_svd": text_svd,
        "svd_scaler": svd_scaler,
        "tstat_scaler": tstat_scaler,
        "vis_pca": vis_pca,
        "vis_pca_scaler": vis_pca_scaler,
        "cv_scaler": cv_scaler,
        "aux_lr_models": aux_lr_models,
        "models": models,
        "metrics": metrics,
    }
    bundle_path = os.path.join(ARTIFACTS_DIR, "model_bundle.joblib")
    joblib.dump(bundle, bundle_path, compress=3)

    write_answer_file(os.path.join(ARTIFACTS_DIR, "val_answer.txt"), pred_a_va, preds_b_va, preds_c_va)
    write_answer_file(os.path.join(ARTIFACTS_DIR, "test_answer.txt"), pred_a_te, preds_b_te, preds_c_te)

    def build_records(df, split, pa, pba, pb_d, pbp_d, pc_d, pcp_d, has_gt):
        recs = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            item = {
                "id": int(row["id"]),
                "split": split,
                "ocr": row["ocr_clean"],
                "prediction": {
                    "task_a": {
                        "label_id": int(pa[idx]),
                        "label": SENTIMENT_LABELS[int(pa[idx])],
                        "probs": [round(float(x), 4) for x in pba[idx]],
                    },
                    "task_b": {
                        k: {
                            "active": bool(int(pb_d[k][idx]) == 1),
                            "prob": round(float(pbp_d[k][idx][1]), 4),
                        }
                        for k in EMOTION_KEYS
                    },
                    "task_c": {
                        k: {
                            "level": int(pc_d[k][idx]),
                            "label": TASK_C_SCALES[k]["labels"][int(pc_d[k][idx])],
                            "max_level": len(TASK_C_SCALES[k]["labels"]) - 1,
                            "probs": [round(float(x), 4) for x in pcp_d[k][idx]],
                        }
                        for k in EMOTION_KEYS
                    },
                },
            }
            if has_gt:
                gt_a = int(SENTIMENT_RAW_TO_ID[row["overall"]])
                item["ground_truth"] = {
                    "task_a": {"label_id": gt_a, "label": SENTIMENT_LABELS[gt_a], "raw": row["overall"]},
                    "task_b": {k: bool(row[k] != TASK_B_NEGATIVE_RAW[k]) for k in EMOTION_KEYS},
                    "task_c": {
                        k: {
                            "level": int(TASK_C_SCALES[k]["raw_to_id"][row[k]]),
                            "label": TASK_C_SCALES[k]["labels"][int(TASK_C_SCALES[k]["raw_to_id"][row[k]])],
                            "raw": row[k],
                        }
                        for k in EMOTION_KEYS
                    },
                }
            recs.append(item)
        return recs

    val_recs = build_records(val_df, "val", pred_a_va, prob_a_va, preds_b_va, probs_b_va, preds_c_va, probs_c_va, True)
    test_recs = build_records(test_df, "test", pred_a_te, prob_a_te, preds_b_te, probs_b_te, preds_c_te, probs_c_te, False)

    with open(os.path.join(ARTIFACTS_DIR, "val_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(val_recs, f)
    with open(os.path.join(ARTIFACTS_DIR, "test_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(test_recs, f)
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSummary Metrics:", json.dumps(metrics["summary"], indent=2), flush=True)
    print(f"Completed training pipeline in {round(time.time() - t0, 1)}s", flush=True)


if __name__ == "__main__":
    main()
