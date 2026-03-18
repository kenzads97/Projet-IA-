"""
EXOFIT — Backend FastAPI
Partie 1 : Diagnostic IA (infirmière + LLM + télémédecin)
Partie 2 : Analyse ECG automatique
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
import json
import os

try:
    from prompt_engine import (
        ACTIVE_LLM,
        LLM_CONFIGS,
        build_prompt,
        call_llm,
        diagnose_case_with_providers,
        get_available_cases,
        get_case_by_id,
    )
    from ecg_pipeline import (
        get_preview_signal,
        get_preview_12_leads,
        list_available_ecg_files,
        load_ecg_file_bytes,
        preprocess_ecg,
        predict_pathology,
        read_ecg_csv,
    )
    from prediction_pipeline import (
        META_PATH,
        MODEL_PATH,
        MAPPING_PATH,
        FEATURE_ORDER,
        get_metadata_patient_details,
        get_metadata_patient_index,
        load_case_index,
        load_ecg_metadata,
        load_patient_ecg_mapping,
        predict_future_disease,
        train_prediction_pipeline,
    )
except ModuleNotFoundError:
    from backend.prompt_engine import (
        ACTIVE_LLM,
        LLM_CONFIGS,
        build_prompt,
        call_llm,
        diagnose_case_with_providers,
        get_available_cases,
        get_case_by_id,
    )
    from backend.ecg_pipeline import (
        get_preview_signal,
        get_preview_12_leads,
        list_available_ecg_files,
        load_ecg_file_bytes,
        preprocess_ecg,
        predict_pathology,
        read_ecg_csv,
    )
    from backend.prediction_pipeline import (
        META_PATH,
        MODEL_PATH,
        MAPPING_PATH,
        FEATURE_ORDER,
        get_metadata_patient_details,
        get_metadata_patient_index,
        load_case_index,
        load_ecg_metadata,
        load_patient_ecg_mapping,
        predict_future_disease,
        train_prediction_pipeline,
    )

app = FastAPI(title="EXOFIT API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODÈLES DE DONNÉES ────────────────────────────────────────────────────

class PatientData(BaseModel):
    age: int
    sexe: str                        # "M" | "F"
    taille_cm: Optional[int] = None
    poids_kg: Optional[float] = None
    symptomes: list[str]             # ex: ["douleur_thoracique", "dyspnee"]
    symptome_libre: Optional[str] = None
    tension_systolique: Optional[int] = None
    tension_diastolique: Optional[int] = None
    frequence_cardiaque: Optional[int] = None
    temperature: Optional[float] = None
    spo2: Optional[int] = None
    antecedents: list[str] = []
    traitements_en_cours: list[str] = []
    examens_realises: list[str] = []  # ex: ["ecg", "nfs", "crp"]
    resultats_examens: dict = {}      # ex: {"crp": "45 mg/L", "nfs": "Hb 10g/dL"}

class DiagnosticDifferentielItem(BaseModel):
    nom: str
    explication: str
    credibilite: float

class DiagnosticResponse(BaseModel):
    diagnostic_preliminaire: str
    explication_diagnostic: str
    diagnostics_differentiels: list[DiagnosticDifferentielItem]
    questions_complementaires: list[str]
    examens_proposes: list[str]
    traitements_proposes: list[str]
    niveau_urgence: str              # "faible" | "modere" | "eleve" | "critique"
    confiance: float                 # 0.0 à 1.0
    modele_utilise: str

class ValidationMedecin(BaseModel):
    cas_id: str
    diagnostic_valide: bool
    diagnostic_corrige: Optional[str] = None
    commentaire: Optional[str] = None
    medecin_id: str

class ECGDiagnosticResponse(BaseModel):
    pathologie_detectee: str
    pathologie_id: Optional[str] = None
    probabilites: dict[str, float]   # {"fibrillation_auriculaire": 0.87, ...}
    features_extraites: dict
    confiance: float
    recommandation: str
    methode: str
    source_fichier: Optional[str] = None
    preview_signal: list[float] = []
    preview_12_leads: dict[str, list[float]] = {}


class CaseSelectionRequest(BaseModel):
    case_id: str
    providers: list[str] = []


class ECGFileSelectionRequest(BaseModel):
    file_id: str


class PredictionTrainRequest(BaseModel):
    test_ratio: float = 0.2


class PredictionRequest(BaseModel):
    case_id: Optional[str] = None
    patient_id: Optional[str] = None
    ecg_files: list[str] = []


class PredictionResponse(BaseModel):
    case_id: Optional[str] = None
    patient_id: Optional[str] = None
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    distances: dict[str, float]
    n_ecg_used: int
    ecg_files: list[str]

# ─── ROUTES PARTIE 1 : DIAGNOSTIC IA ─────────────────────────────────────

@app.post("/api/diagnostic", response_model=DiagnosticResponse)
async def obtenir_diagnostic(patient: PatientData):
    """
    Reçoit les données patient de l'infirmière,
    génère un prompt structuré, appelle le LLM,
    retourne le diagnostic préliminaire.
    """
    try:
        prompt = build_prompt(patient.dict())
        resultat = await call_llm(prompt)
        return DiagnosticResponse(**resultat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cases")
async def list_cases():
    """Retourne les cas cliniques disponibles pour le frontend."""
    cases = get_available_cases()
    return {
        "total": len(cases),
        "active_provider": ACTIVE_LLM,
        "available_providers": sorted(LLM_CONFIGS.keys()),
        "cases": [
            {
                "id": case["id"],
                "description": case["description"],
                "source_document": case.get("source_document"),
            }
            for case in cases
        ],
    }


@app.get("/api/cases/{case_id}")
async def get_case(case_id: str):
    """Retourne le detail d'un cas clinique."""
    case = get_case_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Cas introuvable")
    return case


@app.post("/api/diagnostic/case")
async def obtenir_diagnostic_pour_cas(request: CaseSelectionRequest):
    """Execute un diagnostic pour un cas selectionne avec un ou plusieurs providers."""
    try:
        result = await diagnose_case_with_providers(
            case_id=request.case_id,
            providers=request.providers or [ACTIVE_LLM],
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/validation")
async def valider_diagnostic(validation: ValidationMedecin):
    """
    Le télémédecin valide ou corrige le diagnostic.
    Enregistre pour l'évaluation de fiabilité.
    """
    # En production : sauvegarder en base de données
    # Pour la démo : on log dans un fichier JSON
    log_path = "validations.json"
    validations = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            validations = json.load(f)
    validations.append(validation.dict())
    with open(log_path, "w") as f:
        json.dump(validations, f, indent=2, ensure_ascii=False)

    return {"status": "ok", "message": "Validation enregistrée"}


@app.get("/api/evaluation")
async def get_evaluation():
    """
    Calcule les métriques de fiabilité à partir des validations enregistrées.
    """
    log_path = "validations.json"
    if not os.path.exists(log_path):
        return {"message": "Aucune validation enregistrée"}

    with open(log_path) as f:
        validations = json.load(f)

    total = len(validations)
    corrects = sum(1 for v in validations if v["diagnostic_valide"])
    accuracy = corrects / total if total > 0 else 0

    return {
        "total_cas": total,
        "diagnostics_valides": corrects,
        "accuracy": round(accuracy, 3),
        "note": "Pour précision/rappel/F1, brancher sur les cas tests labellisés"
    }


# ─── ROUTES PARTIE 2 : ECG ───────────────────────────────────────────────

def _run_ecg_analysis(file_bytes: bytes, source_fichier: Optional[str] = None) -> dict:
    """Pipeline ECG commun pour upload direct et ECG charges depuis data/ecg/raw."""
    raw_signal = read_ecg_csv(file_bytes)
    preprocessed = preprocess_ecg(file_bytes)
    result = predict_pathology(preprocessed)
    result["source_fichier"] = source_fichier
    result["preview_signal"] = get_preview_signal(raw_signal)
    result["preview_12_leads"] = get_preview_12_leads(raw_signal)
    return result


@app.post("/api/ecg/analyser", response_model=ECGDiagnosticResponse)
async def analyser_ecg(file: UploadFile = File(...)):
    """
    Reçoit un fichier ECG au format CSV (12 dérivations),
    applique le pipeline de prétraitement,
    retourne le diagnostic de pathologie.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Format CSV requis")

    contenu = await file.read()
    try:
        resultat = _run_ecg_analysis(contenu, source_fichier=file.filename)
        return ECGDiagnosticResponse(**resultat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ecg/files")
async def get_ecg_files(limit: int = 200):
    """Retourne une liste d'ECG reels disponibles dans data/ecg/raw."""
    safe_limit = max(1, min(limit, 1000))
    return list_available_ecg_files(limit=safe_limit)


@app.post("/api/ecg/analyser-fichier", response_model=ECGDiagnosticResponse)
async def analyser_ecg_fichier(request: ECGFileSelectionRequest):
    """Analyse un ECG reel deja present dans data/ecg/raw."""
    try:
        contenu = load_ecg_file_bytes(request.file_id)
        resultat = _run_ecg_analysis(contenu, source_fichier=request.file_id)
        return ECGDiagnosticResponse(**resultat)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ecg/pathologies")
async def get_pathologies():
    """Liste des 7 pathologies détectables."""
    return {
        "pathologies": [
            {"id": "FA",  "nom": "Fibrillation auriculaire",      "critere_ecg": "Absence onde P, rythme irrégulier"},
            {"id": "TV",  "nom": "Tachycardie ventriculaire",     "critere_ecg": "QRS larges > 0.12s, FC > 100"},
            {"id": "BR",  "nom": "Bradycardie",                   "critere_ecg": "FC < 60 bpm"},
            {"id": "BAV", "nom": "Bloc auriculo-ventriculaire",   "critere_ecg": "PR > 0.20s ou dissociation P-QRS"},
            {"id": "HVG", "nom": "Hypertrophie ventriculaire G.", "critere_ecg": "Sokolow-Lyon > 35mm"},
            {"id": "ISC", "nom": "Ischémie myocardique",          "critere_ecg": "Sous-décalage ST > 1mm"},
            {"id": "IDM", "nom": "Infarctus du myocarde",         "critere_ecg": "Sus-décalage ST > 2mm, onde Q"},
        ]
    }


# ─── ROUTES PARTIE 3 : PREDICTION ───────────────────────────────────────

@app.get("/api/prediction/status")
async def get_prediction_status():
    mapping = load_patient_ecg_mapping()
    cases = load_case_index()
    meta_rows = 0
    meta_patients = 0
    meta_error = None
    try:
        meta_df = load_ecg_metadata()
        meta_rows = int(len(meta_df))
        meta_patients = int(meta_df["patient_id"].nunique()) if "patient_id" in meta_df.columns else 0
    except Exception as exc:
        meta_error = str(exc)
    return {
        "mapping_exists": MAPPING_PATH.exists(),
        "mapping_path": str(MAPPING_PATH),
        "metadata_exists": META_PATH.exists(),
        "metadata_path": str(META_PATH),
        "metadata_rows": meta_rows,
        "metadata_patients": meta_patients,
        "metadata_error": meta_error,
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "mapped_patients": len(mapping),
        "available_cases": len(cases),
        "feature_count": len(FEATURE_ORDER),
    }


@app.get("/api/prediction/case/{case_id}")
async def get_prediction_case_mapping(case_id: str):
    mapping = load_patient_ecg_mapping()
    matches = [item for item in mapping if item.get("case_id") == case_id]
    if not matches:
        return {
            "case_id": case_id,
            "found": False,
            "patient_ids": [],
            "ecg_files": [],
            "target_labels": [],
        }

    ecg_files: list[str] = []
    patient_ids: list[str] = []
    target_labels: list[str] = []
    for item in matches:
        patient_id = item.get("patient_id")
        target_label = item.get("target_label")
        if patient_id and patient_id not in patient_ids:
            patient_ids.append(str(patient_id))
        if target_label and target_label not in target_labels:
            target_labels.append(str(target_label))
        for ecg_file in item.get("ecg_files", []) or []:
            if ecg_file not in ecg_files:
                ecg_files.append(str(ecg_file))

    return {
        "case_id": case_id,
        "found": True,
        "patient_ids": patient_ids,
        "ecg_files": ecg_files,
        "target_labels": target_labels,
    }


@app.get("/api/prediction/patients")
async def get_prediction_patients(limit: int = 500):
    try:
        return {
            "patients": get_metadata_patient_index(limit=max(1, min(limit, 2000))),
            "source": "df_meta.pkl",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prediction/patient/{patient_id}")
async def get_prediction_patient(patient_id: str):
    try:
        result = get_metadata_patient_details(patient_id)
        if not result:
            raise HTTPException(status_code=404, detail="Patient metadata introuvable")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/prediction/train")
async def train_prediction_model(request: PredictionTrainRequest):
    try:
        result = train_prediction_pipeline(test_ratio=request.test_ratio, save_model=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/prediction/predict", response_model=PredictionResponse)
async def predict_future_patient_disease(request: PredictionRequest):
    try:
        if request.patient_id:
            patient_meta = get_metadata_patient_details(request.patient_id)
            if not patient_meta:
                raise HTTPException(status_code=404, detail="Patient metadata introuvable")
            ecg_files = request.ecg_files or patient_meta["ecg_files"]
            result = predict_future_disease(
                patient=patient_meta["patient_payload"],
                ecg_files=ecg_files,
            )
            result["patient_id"] = request.patient_id
        else:
            case = get_case_by_id(request.case_id or "")
            if not case:
                raise HTTPException(status_code=404, detail="Cas clinique introuvable")
            patient = case.get("patient", {})
            result = predict_future_disease(patient=patient, ecg_files=request.ecg_files)
            result["case_id"] = request.case_id
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── HEALTH CHECK ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "EXOFIT API opérationnelle", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
