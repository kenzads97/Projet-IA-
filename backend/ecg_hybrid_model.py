from __future__ import annotations

import ast
import json
import re
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupShuffleSplit
except ModuleNotFoundError:
    RandomForestClassifier = None
    GroupShuffleSplit = None
    accuracy_score = None
    f1_score = None

try:
    from ecg_pipeline import preprocess_ecg
except ModuleNotFoundError:
    from backend.ecg_pipeline import preprocess_ecg


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_ECG_DIR = BASE_DIR / "data" / "ecg" / "raw"
META_PATH = BASE_DIR / "data" / "ecg" / "df_meta.pkl"
HYBRID_MODEL_PATH = BASE_DIR / "backend" / "ecg_hybrid_model.joblib"
HYBRID_META_PATH = BASE_DIR / "backend" / "ecg_hybrid_model_meta.json"

HYBRID_FEATURE_ORDER = [
    "heart_rate_bpm",
    "rr_mean_ms",
    "rr_std_ms",
    "rr_irregularity_ratio",
    "dominant_slow_heart_rate_bpm",
    "alternating_rr_ratio",
    "n_r_peaks",
    "qrs_width_ms",
    "r_amplitude_mean",
    "v5_r_amplitude",
    "st_max_deviation_mm",
    "st_mean_deviation_mm",
    "qrs_wide_leads_count",
    "qrs_borderline_wide_leads_count",
    "v1_qrs_polarity",
    "v6_qrs_polarity",
    "fibrillation_power_ratio",
    "signal_quality_score",
    "baseline_wander_score",
    "snr_like_ratio",
]

TARGET_CLASSES = [
    "normal",
    "fibrillation_auriculaire",
    "tachycardie_ventriculaire",
    "bradycardie",
    "bloc_auriculo_ventriculaire",
    "hypertrophie_ventriculaire_gauche",
    "ischemie_myocardique",
    "infarctus_du_myocarde",
]


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


def normalize_hybrid_ecg_label(value: Any) -> str | None:
    items = [_clean_diagnosis_item(item) for item in _parse_diagnosis_items(value)]
    labels = [label for label in (_classify_diagnosis_item(item) for item in items) if label]
    if not labels:
        return None
    primary_label = labels[0]
    if any(label != primary_label for label in labels[1:]):
        return None
    return primary_label


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _load_metadata():
    if pd is None:
        raise ModuleNotFoundError("pandas est requis pour entraîner le modèle hybride ECG.")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Fichier metadata introuvable: {META_PATH}")
    return pd.read_pickle(META_PATH)


def _extract_hybrid_features_from_file(file_name: str) -> dict[str, float]:
    ecg_path = RAW_ECG_DIR / Path(str(file_name)).name
    if not ecg_path.exists():
        raise FileNotFoundError(f"ECG introuvable: {ecg_path}")
    preprocessed = preprocess_ecg(ecg_path.read_bytes())
    features = dict(preprocessed["features"])
    return {name: _safe_float(features.get(name), 0.0) for name in HYBRID_FEATURE_ORDER}


def _build_training_rows(min_class_size: int = 40, max_per_class: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    df = _load_metadata().copy()
    df["target_label"] = df["diagnosis"].apply(normalize_hybrid_ecg_label)
    df = df[df["target_label"].notna()].copy()

    class_counts = df["target_label"].value_counts().to_dict()
    kept_labels = {label for label, count in class_counts.items() if count >= min_class_size}
    if not kept_labels:
        raise ValueError("Aucune classe suffisante pour entraîner le modèle hybride.")

    sampled_groups = []
    for label in sorted(kept_labels):
        group = df[df["target_label"] == label].copy()
        if len(group) > max_per_class:
            group = group.sample(n=max_per_class, random_state=42)
        sampled_groups.append(group)
    sampled = pd.concat(sampled_groups, ignore_index=True)

    X_rows: list[list[float]] = []
    y_rows: list[str] = []
    groups: list[str] = []
    kept_counter: dict[str, int] = {}

    for row in sampled.itertuples(index=False):
        try:
            feature_dict = _extract_hybrid_features_from_file(getattr(row, "ecg_file_path", ""))
        except Exception:
            continue
        X_rows.append([feature_dict[name] for name in HYBRID_FEATURE_ORDER])
        label = str(getattr(row, "target_label"))
        y_rows.append(label)
        groups.append(str(getattr(row, "patient_id", "")))
        kept_counter[label] = kept_counter.get(label, 0) + 1

    if len(set(y_rows)) < 2:
        raise ValueError("Le dataset hybride obtenu ne contient pas assez de classes valides.")

    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=object), np.array(groups, dtype=object), kept_counter


def clear_hybrid_model_cache() -> None:
    load_hybrid_model.cache_clear()


@lru_cache(maxsize=1)
def load_hybrid_model() -> dict[str, Any] | None:
    if not HYBRID_MODEL_PATH.exists():
        return None
    return joblib.load(HYBRID_MODEL_PATH)


def hybrid_model_available() -> bool:
    return HYBRID_MODEL_PATH.exists()


def get_hybrid_model_status() -> dict[str, Any]:
    status = {
        "model_exists": HYBRID_MODEL_PATH.exists(),
        "model_path": str(HYBRID_MODEL_PATH),
        "metadata_exists": META_PATH.exists(),
        "metadata_path": str(META_PATH),
    }
    if HYBRID_META_PATH.exists():
        try:
            status["training_info"] = json.loads(HYBRID_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            status["training_info_error"] = "Impossible de lire le fichier meta du modèle hybride."
    return status


def train_hybrid_ecg_model(
    test_ratio: float = 0.2,
    min_class_size: int = 40,
    max_per_class: int = 400,
    save_model: bool = True,
) -> dict[str, Any]:
    if RandomForestClassifier is None or GroupShuffleSplit is None:
        raise ModuleNotFoundError("scikit-learn est requis pour entraîner le modèle hybride ECG.")

    X, y, groups, kept_counter = _build_training_rows(
        min_class_size=max(5, int(min_class_size)),
        max_per_class=max(20, int(max_per_class)),
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=min(max(test_ratio, 0.1), 0.4), random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 3) if accuracy_score is not None else None,
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 3) if f1_score is not None else None,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "classes": sorted({str(label) for label in y.tolist()}),
        "samples_per_class": kept_counter,
    }

    payload = {
        "model": classifier,
        "feature_order": HYBRID_FEATURE_ORDER,
        "classes": classifier.classes_.tolist(),
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    if save_model:
        joblib.dump(payload, HYBRID_MODEL_PATH)
        HYBRID_META_PATH.write_text(
            json.dumps(
                {
                    "trained_at": payload["trained_at"],
                    "feature_order": HYBRID_FEATURE_ORDER,
                    "classes": payload["classes"],
                    "metrics": metrics,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        clear_hybrid_model_cache()

    return {
        "status": "ok",
        "saved": bool(save_model),
        "model_path": str(HYBRID_MODEL_PATH),
        "metrics": metrics,
    }


def predict_hybrid_probabilities(features: dict[str, Any]) -> dict[str, float] | None:
    payload = load_hybrid_model()
    if not payload:
        return None

    feature_order = payload.get("feature_order", HYBRID_FEATURE_ORDER)
    classifier = payload["model"]
    vector = np.array([[_safe_float(features.get(name), 0.0) for name in feature_order]], dtype=float)
    probabilities = classifier.predict_proba(vector)[0]

    by_class = {label: 0.0 for label in TARGET_CLASSES}
    for index, label in enumerate(payload.get("classes", [])):
        by_class[str(label)] = float(probabilities[index])

    total = sum(by_class.values()) or 1.0
    return {label: value / total for label, value in by_class.items()}
