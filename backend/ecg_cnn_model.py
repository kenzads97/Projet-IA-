from __future__ import annotations

import ast
import json
import re
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

try:
    from ecg_pipeline import WINDOW_SAMPLES, N_LEADS, build_cnn_model, preprocess_ecg
except ModuleNotFoundError:
    from backend.ecg_pipeline import WINDOW_SAMPLES, N_LEADS, build_cnn_model, preprocess_ecg

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_ECG_DIR = BASE_DIR / "data" / "ecg" / "raw"
META_PATH = BASE_DIR / "data" / "ecg" / "df_meta.pkl"
CNN_MODEL_PATH = BASE_DIR / "backend" / "ecg_cnn_model.keras"
CNN_META_PATH = BASE_DIR / "backend" / "ecg_cnn_model_meta.json"


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_diagnosis_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [text]


def _clean_diagnosis_item(value: str) -> str:
    text = _normalize_text(value)
    text = re.sub(r"^userinsert\s*:\s*", "", text)
    return text.strip()


def _is_ignorable_diagnosis_item(text: str) -> bool:
    ignored_tokens = [
        "valide par le cardiologue",
        "lors d",
        "aucun ecg precedent",
        "aucun changement significatif",
        "comparaison",
    ]
    return any(token in text for token in ignored_tokens)


def _classify_diagnosis_item(text: str) -> str | None:
    if not text or _is_ignorable_diagnosis_item(text):
        return None
    if "tachycardie ventriculaire" in text:
        return "tachycardie_ventriculaire"
    if "infarctus" in text and "possible" not in text:
        return "infarctus_du_myocarde"
    if "isch" in text and "possible" not in text:
        return "ischemie_myocardique"
    if any(
        token in text
        for token in [
            "bloc de branche",
            "bbg",
            "bbd",
            "bid",
            "bbdi",
            "bloc complet gauche",
            "bloc incomplet droit",
        ]
    ):
        return "bloc_auriculo_ventriculaire"
    if "hypertrophie ventriculaire gauche" in text or "diagnostic d'hvg" in text:
        return "hypertrophie_ventriculaire_gauche"
    if any(
        token in text
        for token in [
            "fibrillation auriculaire",
            "fibrillation atriale",
            "flutter auriculaire",
            "fibrillo flutter",
        ]
    ):
        return "fibrillation_auriculaire"
    if "bradycardie sinusale" in text or "bradycardie sinusal" in text:
        return "bradycardie"
    if (
        "rythme sinusal normal" in text
        or "ecg normal" in text
        or "dans les limites de la normale" in text
        or "normal sinus" in text
        or "rythme sinusal avec arythmie sinusale" in text
    ):
        return "normal"
    return None


def normalize_cnn_ecg_label(value: Any) -> str | None:
    items = [_clean_diagnosis_item(item) for item in _parse_diagnosis_items(value)]
    labels = [label for label in (_classify_diagnosis_item(item) for item in items) if label]
    if not labels:
        return None
    primary_label = labels[0]
    if any(label != primary_label for label in labels[1:]):
        return None
    return primary_label


def _load_metadata():
    if pd is None:
        raise ModuleNotFoundError("pandas est requis pour entraîner le CNN ECG.")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Fichier metadata introuvable: {META_PATH}")
    return pd.read_pickle(META_PATH)


def _group_train_test_split(groups: np.ndarray, test_ratio: float = 0.2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(random_state)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    test_count = max(1, int(len(shuffled) * test_ratio))
    test_groups = set(shuffled[:test_count].tolist())
    test_idx = np.array([idx for idx, group in enumerate(groups) if group in test_groups], dtype=int)
    train_idx = np.array([idx for idx, group in enumerate(groups) if group not in test_groups], dtype=int)
    if len(train_idx) == 0 or len(test_idx) == 0:
        split = max(1, int(len(groups) * test_ratio))
        test_idx = np.arange(0, split, dtype=int)
        train_idx = np.arange(split, len(groups), dtype=int)
    return train_idx, test_idx


def _accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def _macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1_scores: list[float] = []
    for label in range(n_classes):
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def _extract_signal_from_file(file_name: str) -> np.ndarray:
    ecg_path = RAW_ECG_DIR / Path(str(file_name)).name
    if not ecg_path.exists():
        raise FileNotFoundError(f"ECG introuvable: {ecg_path}")
    preprocessed = preprocess_ecg(ecg_path.read_bytes())
    signal = preprocessed["signal"][:WINDOW_SAMPLES]
    if signal.shape[0] < WINDOW_SAMPLES:
        pad = np.zeros((WINDOW_SAMPLES - signal.shape[0], signal.shape[1]), dtype=signal.dtype)
        signal = np.vstack([signal, pad])
    signal = signal[:, :N_LEADS]
    return signal.astype("float32")


def _build_cnn_dataset(min_class_size: int = 40, max_per_class: int = 80):
    df = _load_metadata().copy()
    df["target_label"] = df["diagnosis"].apply(normalize_cnn_ecg_label)
    df = df[df["target_label"].notna()].copy()

    class_counts = df["target_label"].value_counts().to_dict()
    kept_labels = [label for label, count in sorted(class_counts.items()) if count >= min_class_size]
    if len(kept_labels) < 2:
        raise ValueError("Pas assez de classes valides pour entraîner le CNN ECG.")

    sampled_groups = []
    for label in kept_labels:
        group = df[df["target_label"] == label].copy()
        if len(group) > max_per_class:
            group = group.sample(n=max_per_class, random_state=42)
        sampled_groups.append(group)
    sampled = pd.concat(sampled_groups, ignore_index=True)

    label_to_index = {label: index for index, label in enumerate(sorted(kept_labels))}
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    groups: list[str] = []
    kept_counter: dict[str, int] = {}

    for row in sampled.itertuples(index=False):
        try:
            signal = _extract_signal_from_file(getattr(row, "ecg_file_path", ""))
        except Exception:
            continue
        label = str(getattr(row, "target_label"))
        X_rows.append(signal)
        y_rows.append(label_to_index[label])
        groups.append(str(getattr(row, "patient_id", "")))
        kept_counter[label] = kept_counter.get(label, 0) + 1

    if len(set(y_rows)) < 2:
        raise ValueError("Le dataset CNN obtenu ne contient pas assez de classes valides.")

    X = np.stack(X_rows).astype("float32")
    y = np.array(y_rows, dtype="int32")
    groups_arr = np.array(groups, dtype=object)
    classes = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    return X, y, groups_arr, classes, kept_counter


def clear_cnn_model_cache() -> None:
    load_cnn_model.cache_clear()


def cnn_model_available() -> bool:
    return CNN_MODEL_PATH.exists()


@lru_cache(maxsize=1)
def load_cnn_model():
    if tf is None or not CNN_MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(CNN_MODEL_PATH)


def get_cnn_model_status() -> dict[str, Any]:
    status = {
        "model_exists": CNN_MODEL_PATH.exists(),
        "model_path": str(CNN_MODEL_PATH),
        "metadata_exists": META_PATH.exists(),
        "metadata_path": str(META_PATH),
        "tensorflow_available": tf is not None,
    }
    if CNN_META_PATH.exists():
        try:
            status["training_info"] = json.loads(CNN_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            status["training_info_error"] = "Impossible de lire le fichier meta du modèle CNN."
    return status


def train_cnn_ecg_model(
    test_ratio: float = 0.2,
    min_class_size: int = 40,
    max_per_class: int = 80,
    epochs: int = 8,
    batch_size: int = 8,
    save_model: bool = True,
) -> dict[str, Any]:
    if tf is None:
        raise ModuleNotFoundError("TensorFlow est requis pour entraîner le CNN ECG.")

    X, y, groups, classes, kept_counter = _build_cnn_dataset(
        min_class_size=max(5, int(min_class_size)),
        max_per_class=max(10, int(max_per_class)),
    )

    train_idx, test_idx = _group_train_test_split(groups, test_ratio=min(max(test_ratio, 0.1), 0.4), random_state=42)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = build_cnn_model(num_classes=len(classes))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=max(1, int(epochs)),
        batch_size=max(4, int(batch_size)),
        verbose=0,
        callbacks=callbacks,
    )

    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    metrics = {
        "accuracy": round(_accuracy_score(y_test, y_pred), 3),
        "macro_f1": round(_macro_f1_score(y_test, y_pred, len(classes)), 3),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "classes": classes,
        "samples_per_class": kept_counter,
        "epochs": int(max(1, int(epochs))),
        "batch_size": int(max(4, int(batch_size))),
    }

    trained_at = datetime.now().isoformat(timespec="seconds")
    if save_model:
        model.save(CNN_MODEL_PATH)
        CNN_META_PATH.write_text(
            json.dumps(
                {
                    "trained_at": trained_at,
                    "classes": classes,
                    "metrics": metrics,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        clear_cnn_model_cache()

    return {
        "status": "ok",
        "saved": bool(save_model),
        "model_path": str(CNN_MODEL_PATH),
        "metrics": metrics,
    }


def predict_cnn_probabilities(signal: np.ndarray, classes: list[str] | None = None) -> dict[str, float] | None:
    model = load_cnn_model()
    if model is None:
        return None

    signal_window = signal[:WINDOW_SAMPLES]
    if signal_window.shape[0] < WINDOW_SAMPLES:
        pad = np.zeros((WINDOW_SAMPLES - signal_window.shape[0], signal_window.shape[1]), dtype=signal_window.dtype)
        signal_window = np.vstack([signal_window, pad])
    signal_window = signal_window[:, :N_LEADS].astype("float32")

    if classes is None:
        if CNN_META_PATH.exists():
            info = json.loads(CNN_META_PATH.read_text(encoding="utf-8"))
            classes = info.get("classes", [])
        else:
            classes = []

    probs = model.predict(signal_window[np.newaxis], verbose=0)[0]
    by_class = {label: 0.0 for label in classes}
    for idx, label in enumerate(classes):
        if idx < len(probs):
            by_class[label] = float(probs[idx])

    total = sum(by_class.values()) or 1.0
    return {label: value / total for label, value in by_class.items()}
