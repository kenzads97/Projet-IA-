"""
Pipeline de prediction patient a partir de plusieurs ECG + cas cliniques.

Objectif :
- relier plusieurs ECG a un patient/cas clinique
- construire un dataset d'apprentissage
- separer train/test au niveau patient
- entrainer un modele simple adapte a de petits jeux de donnees tabulaires
- predire un risque / une maladie cible pour un patient

Modele choisi :
- classifieur par centroïdes (nearest centroid)

Pourquoi :
- robuste sur petits datasets
- interpretable
- ne demande pas de dependances externes supplementaires
- bien adapte a des features structurees extraites du pipeline ECG
"""

from __future__ import annotations

import ast
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None
try:
    from sklearn.ensemble import RandomForestClassifier
except ModuleNotFoundError:
    RandomForestClassifier = None

try:
    from ecg_pipeline import preprocess_ecg
except ModuleNotFoundError:
    from backend.ecg_pipeline import preprocess_ecg


BASE_DIR = Path(__file__).resolve().parents[1]
PARSED_CASES_DIR = BASE_DIR / "data" / "cas_cliniques" / "parsed"
RAW_ECG_DIR = BASE_DIR / "data" / "ecg" / "raw"
MAPPING_PATH = BASE_DIR / "data" / "ecg" / "patient_ecg_mapping.json"
META_PATH = BASE_DIR / "data" / "ecg" / "df_meta.pkl"
MODEL_PATH = BASE_DIR / "backend" / "prediction_model.json"
MIN_CLASS_SIZE = 200


FEATURE_ORDER = [
    "heart_rate_bpm",
    "rr_mean_ms",
    "rr_std_ms",
    "rr_irregularity_ratio",
    "n_r_peaks",
    "qrs_width_ms",
    "r_amplitude_mean",
    "v5_r_amplitude",
    "st_max_deviation_mm",
    "st_mean_deviation_mm",
    "fibrillation_power_ratio",
    "age",
    "sex_male",
    "symptom_count",
    "antecedent_count",
    "exam_count",
]


@dataclass
class PredictionSample:
    patient_id: str
    case_id: str
    ecg_file: str
    target_label: str
    features: dict[str, float]
    history_size: int = 1


def load_ecg_metadata():
    """Charge les metadonnees ECG fournies par le prof."""
    if pd is None:
        raise ModuleNotFoundError(
            "pandas est requis pour lire df_meta.pkl. Installe-le dans l'environnement utilise par le backend."
        )
    if not META_PATH.exists():
        raise FileNotFoundError(f"Fichier de metadonnees introuvable: {META_PATH}")
    return pd.read_pickle(META_PATH)


def _metadata_patient_payload_from_rows(rows) -> dict[str, Any]:
    if len(rows) == 0:
        return {"age": None, "sexe": "", "examens_realises": ["ecg"], "symptomes": [], "antecedents": []}
    first = rows.iloc[0]
    gender = _normalize_text(getattr(first, "gender", ""))
    sexe = "M" if gender == "male" else "F" if gender == "female" else ""
    return {
        "age": _safe_float(getattr(first, "age", None), 0.0),
        "sexe": sexe,
        "examens_realises": ["ecg"],
        "symptomes": [],
        "antecedents": [],
    }


def get_metadata_patient_index(limit: int = 500) -> list[dict[str, Any]]:
    df = load_ecg_metadata().copy()
    df["target_label"] = df["diagnosis"].apply(normalize_diagnosis_label)

    rows: list[dict[str, Any]] = []
    for patient_id, group in df.groupby("patient_id", sort=False):
        labels = [label for label in group["target_label"].dropna().tolist() if label]
        if not labels:
            continue
        label_counts: dict[str, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        primary_label = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        first = group.iloc[0]
        rows.append(
            {
                "patient_id": str(patient_id),
                "age": getattr(first, "age", None),
                "gender": getattr(first, "gender", None),
                "ecg_count": int(len(group)),
                "primary_label": primary_label,
                "labels": sorted(label_counts.keys()),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def get_metadata_patient_details(patient_id: str) -> dict[str, Any] | None:
    df = load_ecg_metadata().copy()
    group = df[df["patient_id"].astype(str) == str(patient_id)].copy()
    if group.empty:
        return None

    group["target_label"] = group["diagnosis"].apply(normalize_diagnosis_label)
    if pd is not None:
        group["date_parsed"] = pd.to_datetime(group["date"], errors="coerce")
        group = group.sort_values(by=["date_parsed", "ecg_file_path"], kind="stable")
    ecg_files = [Path(str(path)).name for path in group["ecg_file_path"].dropna().tolist()]
    labels = [label for label in group["target_label"].dropna().tolist() if label]
    first = group.iloc[0]
    return {
        "patient_id": str(patient_id),
        "age": getattr(first, "age", None),
        "gender": getattr(first, "gender", None),
        "ecg_files": ecg_files,
        "labels": sorted(set(labels)),
        "primary_label": labels[0] if labels else None,
        "dates": [str(value) for value in group["date"].tolist()],
        "patient_payload": _metadata_patient_payload_from_rows(group),
    }


def load_case_index() -> dict[str, dict[str, Any]]:
    """Charge les cas cliniques parses."""
    cases: dict[str, dict[str, Any]] = {}
    if not PARSED_CASES_DIR.exists():
        return cases

    for path in sorted(PARSED_CASES_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            cases[payload["id"]] = payload
        except Exception:
            continue
    return cases


def load_patient_ecg_mapping() -> list[dict[str, Any]]:
    """
    Charge le mapping patient <-> ECG.

    Format attendu :
    [
      {
        "patient_id": "patient_001",
        "case_id": "10_petits_cas_cliniques_cas_001",
        "target_label": "fibrillation_auriculaire",
        "ecg_files": ["file1.csv", "file2.csv"]
      }
    ]
    """
    if not MAPPING_PATH.exists():
        return []

    payload = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("patients"), list):
        return payload["patients"]
    return []


def infer_target_label(mapping_item: dict[str, Any], case_payload: dict[str, Any] | None) -> str | None:
    """Recupere la cible a predire depuis le mapping ou le cas clinique."""
    explicit = mapping_item.get("target_label") or mapping_item.get("future_disease")
    if explicit:
        return str(explicit).strip().lower()

    if case_payload:
        reference = case_payload.get("diagnostic_reference")
        if reference:
            return str(reference).strip().lower()
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


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


def normalize_diagnosis_label(value: Any) -> str | None:
    """
    Ramene les diagnostics ECG a quelques classes stables et frequentes.
    """
    items = _parse_diagnosis_items(value)
    text = _normalize_text(" | ".join(items))

    if not text:
        return None
    if "fibrillation auriculaire" in text or "fibrillation atriale" in text:
        return "fibrillation_auriculaire"
    if "flutter auriculaire" in text or "fibrillo-flutter" in text or "fibrillo flutter" in text:
        return "flutter_auriculaire"
    if "tachycardie sinusale" in text:
        return "tachycardie_sinusale"
    if "bradycardie sinusale" in text or "bradycardie sinusal" in text:
        return "bradycardie_sinusale"
    if "bloc a-v du premier degre" in text or "bloc auriculo ventriculaire de type 1" in text:
        return "bloc_auriculo_ventriculaire_premier_degre"
    if "rythme ventriculaire entraine" in text:
        return "rythme_ventriculaire_entraine"
    if (
        "rythme sinusal normal" in text
        or "ecg normal" in text
        or "dans les limites de la normale" in text
        or "normal sinus" in text
        or "normal rest ecg" in text
    ):
        return "rythme_normal_sinusal"
    return None


def _patient_context_features(patient: dict[str, Any]) -> dict[str, float]:
    sexe = str(patient.get("sexe", "")).strip().upper()
    return {
        "age": _safe_float(patient.get("age"), 0.0),
        "sex_male": 1.0 if sexe == "M" else 0.0,
        "symptom_count": float(len(patient.get("symptomes", []) or [])),
        "antecedent_count": float(len(patient.get("antecedents", []) or [])),
        "exam_count": float(len(patient.get("examens_realises", []) or [])),
    }


def _metadata_context_features(row: Any) -> dict[str, float]:
    gender = _normalize_text(getattr(row, "gender", ""))
    return {
        "age": _safe_float(getattr(row, "age", None), 0.0),
        "sex_male": 1.0 if gender == "male" else 0.0,
        "symptom_count": 0.0,
        "antecedent_count": 0.0,
        "exam_count": 1.0,
    }


def extract_prediction_features(ecg_file: str, patient: dict[str, Any]) -> dict[str, float]:
    """Extrait les features pour la prediction a partir d'un ECG + contexte patient."""
    ecg_path = RAW_ECG_DIR / Path(ecg_file).name
    if not ecg_path.exists():
        raise FileNotFoundError(f"ECG introuvable: {ecg_file}")

    preprocessed = preprocess_ecg(ecg_path.read_bytes())
    features = dict(preprocessed["features"])
    features.update(_patient_context_features(patient))
    return {name: _safe_float(features.get(name), 0.0) for name in FEATURE_ORDER}


def extract_prediction_features_from_metadata_row(row: Any) -> dict[str, float]:
    ecg_path_value = getattr(row, "ecg_file_path", "")
    ecg_path = RAW_ECG_DIR / Path(str(ecg_path_value)).name
    if not ecg_path.exists():
        raise FileNotFoundError(f"ECG introuvable: {ecg_path_value}")

    preprocessed = preprocess_ecg(ecg_path.read_bytes())
    features = dict(preprocessed["features"])
    features.update(_metadata_context_features(row))
    return {name: _safe_float(features.get(name), 0.0) for name in FEATURE_ORDER}


def _aggregate_feature_dicts(feature_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not feature_dicts:
        return {name: 0.0 for name in FEATURE_ORDER}
    return {
        name: float(np.mean([feature_dict.get(name, 0.0) for feature_dict in feature_dicts]))
        for name in FEATURE_ORDER
    }


def build_prediction_dataset_from_metadata(min_class_size: int = MIN_CLASS_SIZE) -> list[PredictionSample]:
    """
    Construit un dataset temporel a partir de df_meta.pkl.

    Pour chaque patient :
    - on trie les ECG par date
    - on utilise l'historique precedent comme entree
    - on predit le diagnostic normalise de l'ECG suivant
    """
    df = load_ecg_metadata().copy()
    df["target_label"] = df["diagnosis"].apply(normalize_diagnosis_label)
    df = df[df["target_label"].notna()].copy()
    if df.empty:
        return []

    if pd is None:
        return []

    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(by=["patient_id", "date_parsed", "ecg_file_path"], kind="stable")

    feature_cache: dict[str, dict[str, float]] = {}
    temporal_samples: list[PredictionSample] = []

    for patient_id, group in df.groupby("patient_id", sort=False):
        if len(group) < 2:
            continue

        history_features: list[dict[str, float]] = []
        for row in group.itertuples(index=False):
            ecg_path_value = getattr(row, "ecg_file_path", "")
            ecg_file = Path(str(ecg_path_value)).name
            try:
                row_features = feature_cache.get(ecg_file)
                if row_features is None:
                    row_features = extract_prediction_features_from_metadata_row(row)
                    feature_cache[ecg_file] = row_features
            except FileNotFoundError:
                continue

            if history_features:
                temporal_samples.append(
                    PredictionSample(
                        patient_id=str(patient_id),
                        case_id=str(patient_id),
                        ecg_file=ecg_file,
                        target_label=str(getattr(row, "target_label")),
                        features=_aggregate_feature_dicts(history_features),
                        history_size=len(history_features),
                    )
                )

            history_features.append(row_features)

    if not temporal_samples:
        return []

    class_counts: dict[str, int] = {}
    for sample in temporal_samples:
        class_counts[sample.target_label] = class_counts.get(sample.target_label, 0) + 1

    allowed_labels = {label for label, count in class_counts.items() if count >= min_class_size}
    return [sample for sample in temporal_samples if sample.target_label in allowed_labels]


def build_prediction_dataset() -> list[PredictionSample]:
    """
    Construit le dataset supervise.

    Note :
    - avec `df_meta.pkl`, on privilegie un apprentissage temporel par patient/date
    - `patient_ecg_mapping.json` reste un mode secours legacy
    """
    if META_PATH.exists():
        metadata_samples = build_prediction_dataset_from_metadata()
        if metadata_samples:
            return metadata_samples

    case_index = load_case_index()
    mapping = load_patient_ecg_mapping()
    samples: list[PredictionSample] = []

    for item in mapping:
        case_id = item.get("case_id")
        patient_id = item.get("patient_id") or case_id
        ecg_files = item.get("ecg_files", []) or []
        case_payload = case_index.get(case_id, {})
        patient = case_payload.get("patient", {})
        target_label = infer_target_label(item, case_payload)

        if not case_id or not patient or not ecg_files or not target_label:
            continue

        for ecg_file in ecg_files:
            try:
                features = extract_prediction_features(ecg_file, patient)
            except FileNotFoundError:
                continue
            samples.append(
                PredictionSample(
                    patient_id=str(patient_id),
                    case_id=str(case_id),
                    ecg_file=str(ecg_file),
                    target_label=target_label,
                    features=features,
                )
            )
    return samples


def split_train_test_by_patient(
    samples: list[PredictionSample],
    test_ratio: float = 0.2,
) -> tuple[list[PredictionSample], list[PredictionSample]]:
    """Split train/test sans fuite entre ECG d'un meme patient."""
    patient_ids = sorted({sample.patient_id for sample in samples})
    if not patient_ids:
        return [], []

    test_count = max(1, int(math.ceil(len(patient_ids) * test_ratio))) if len(patient_ids) > 1 else 0
    test_patients = set(patient_ids[-test_count:]) if test_count else set()

    train = [sample for sample in samples if sample.patient_id not in test_patients]
    test = [sample for sample in samples if sample.patient_id in test_patients]
    return train, test


def vectorize_samples(samples: list[PredictionSample]) -> tuple[np.ndarray, list[str]]:
    """Transforme les samples en matrice numerique."""
    if not samples:
        return np.empty((0, len(FEATURE_ORDER))), []

    x = np.array(
        [[sample.features[name] for name in FEATURE_ORDER] for sample in samples],
        dtype=float,
    )
    y = [sample.target_label for sample in samples]
    return x, y


def train_random_forest_model(train_samples: list[PredictionSample]) -> dict[str, Any]:
    """Entraine un RandomForest sur les features temporelles."""
    if RandomForestClassifier is None:
        raise ModuleNotFoundError(
            "scikit-learn est requis pour entrainer le modele RandomForest. Installe-le dans l'environnement du backend."
        )

    x_train, y_train = vectorize_samples(train_samples)
    if len(train_samples) < 2 or len(set(y_train)) < 2:
        raise ValueError(
            "Dataset insuffisant pour l'entrainement. Il faut au moins 2 classes valides apres normalisation des diagnostics."
        )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    classifier.fit(x_train, y_train)

    model = {
        "model_type": "random_forest",
        "prediction_mode": "temporal_next_diagnosis",
        "feature_order": FEATURE_ORDER,
        "labels": [str(label) for label in classifier.classes_.tolist()],
        "rf_params": classifier.get_params(),
        "classes": [str(label) for label in classifier.classes_.tolist()],
        "feature_importances": {
            name: round(float(score), 6)
            for name, score in zip(FEATURE_ORDER, classifier.feature_importances_)
        },
        "estimators": [
            {
                "feature": tree.tree_.feature.tolist(),
                "threshold": tree.tree_.threshold.tolist(),
                "children_left": tree.tree_.children_left.tolist(),
                "children_right": tree.tree_.children_right.tolist(),
                "value": tree.tree_.value.tolist(),
                "node_count": int(tree.tree_.node_count),
            }
            for tree in classifier.estimators_
        ],
    }
    return model


def predict_with_model(model: dict[str, Any], feature_dict: dict[str, float]) -> dict[str, Any]:
    """Predictions + probabilites a partir du modele entraine."""
    feature_order = model["feature_order"]
    x = np.array([[feature_dict.get(name, 0.0) for name in feature_order]], dtype=float)

    if model.get("model_type") != "random_forest":
        raise ValueError(f"Modele non supporte: {model.get('model_type')}")

    classes = model["classes"]
    votes = np.zeros(len(classes), dtype=float)
    for estimator_payload in model["estimators"]:
        node = 0
        while True:
            feature_index = int(estimator_payload["feature"][node])
            if feature_index < 0:
                leaf_values = np.array(estimator_payload["value"][node][0], dtype=float)
                total = float(np.sum(leaf_values)) or 1.0
                votes += leaf_values / total
                break
            threshold = float(estimator_payload["threshold"][node])
            next_node = (
                int(estimator_payload["children_left"][node])
                if x[0, feature_index] <= threshold
                else int(estimator_payload["children_right"][node])
            )
            node = next_node

    probability_vector = votes / (len(model["estimators"]) or 1.0)
    probabilities = {
        label: round(float(probability_vector[index]), 3)
        for index, label in enumerate(classes)
    }
    best_label = max(probabilities, key=probabilities.get)
    distances = {label: round(1.0 - prob, 4) for label, prob in probabilities.items()}

    return {
        "prediction": best_label,
        "confidence": probabilities[best_label],
        "probabilities": probabilities,
        "distances": distances,
    }


def evaluate_model(model: dict[str, Any], test_samples: list[PredictionSample]) -> dict[str, Any]:
    """Evaluation simple accuracy sur le split test."""
    if not test_samples:
        return {
            "test_samples": 0,
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "per_label": {},
            "details": [],
        }

    details = []
    correct = 0
    for sample in test_samples:
        result = predict_with_model(model, sample.features)
        is_correct = result["prediction"] == sample.target_label
        correct += int(is_correct)
        details.append(
            {
                "patient_id": sample.patient_id,
                "case_id": sample.case_id,
                "ecg_file": sample.ecg_file,
                "expected": sample.target_label,
                "predicted": result["prediction"],
                "confidence": result["confidence"],
                "correct": is_correct,
            }
        )

    labels = sorted({sample.target_label for sample in test_samples} | set(model.get("labels", [])))
    per_label: dict[str, dict[str, Any]] = {}
    for label in labels:
        tp = sum(1 for item in details if item["expected"] == label and item["predicted"] == label)
        fp = sum(1 for item in details if item["expected"] != label and item["predicted"] == label)
        fn = sum(1 for item in details if item["expected"] == label and item["predicted"] != label)
        support = sum(1 for item in details if item["expected"] == label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_label[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    metric_labels = [stats for stats in per_label.values() if stats["support"] > 0]
    macro_precision = sum(item["precision"] for item in metric_labels) / len(metric_labels) if metric_labels else 0.0
    macro_recall = sum(item["recall"] for item in metric_labels) / len(metric_labels) if metric_labels else 0.0
    macro_f1 = sum(item["f1"] for item in metric_labels) / len(metric_labels) if metric_labels else 0.0

    return {
        "test_samples": len(test_samples),
        "accuracy": round(correct / len(test_samples), 3),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3),
        "per_label": per_label,
        "details": details,
    }


def train_prediction_pipeline(test_ratio: float = 0.2, save_model: bool = True) -> dict[str, Any]:
    """Pipeline complet train/test."""
    samples = build_prediction_dataset()
    if not samples:
        raise ValueError(
            "Aucun sample entrainable. Verifie df_meta.pkl ou, en secours, patient_ecg_mapping.json."
        )

    train_samples, test_samples = split_train_test_by_patient(samples, test_ratio=test_ratio)
    model = train_random_forest_model(train_samples)
    evaluation = evaluate_model(model, test_samples)

    if save_model:
        MODEL_PATH.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "unique_patients": len({sample.patient_id for sample in samples}),
        "labels": model["labels"],
        "model_type": model["model_type"],
        "model_path": str(MODEL_PATH),
        "data_source": "df_meta.pkl" if META_PATH.exists() else "patient_ecg_mapping.json",
        "evaluation": evaluation,
    }


def load_trained_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Aucun modele entraine trouve. Lance d'abord train_prediction_pipeline().")
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def predict_future_disease(
    patient: dict[str, Any],
    ecg_files: list[str],
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Agrege plusieurs ECG historiques d'un meme patient et predit le diagnostic de la periode suivante.
    """
    if not ecg_files:
        raise ValueError("Aucun ECG fourni pour la prediction")

    model = model or load_trained_model()
    feature_vectors = [extract_prediction_features(ecg_file, patient) for ecg_file in ecg_files]

    aggregated_features = {
        name: float(np.mean([vector[name] for vector in feature_vectors]))
        for name in FEATURE_ORDER
    }
    result = predict_with_model(model, aggregated_features)
    result["n_ecg_used"] = len(ecg_files)
    result["ecg_files"] = ecg_files
    result["prediction_mode"] = model.get("prediction_mode", "unknown")
    return result


if __name__ == "__main__":
    try:
        summary = train_prediction_pipeline()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"Prediction pipeline non lance: {exc}")
