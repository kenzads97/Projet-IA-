# EXOFIT

Assistant d'aide a la decision medicale pour teleconsultation infirmiere, analyse ECG 12 derivations et prediction temporelle a partir d'un historique ECG.

Projet MIAS M2, Centrale Lille, Mars 2026.

## Vue d'ensemble

Le projet comporte 3 briques principales :

1. `Prompt Engine`
- construit un prompt medical structure
- interroge un ou plusieurs modeles generatifs
- renvoie un diagnostic preliminaire, des diagnostics differentiels, des questions, des examens, des traitements proposes et un niveau d'urgence

2. `ECG Pipeline`
- lit les fichiers ECG bruts depuis `data/ecg/raw`
- pretraite les signaux
- extrait des features interpretable
- applique des regles heuristiques pour classer des anomalies cardiaques
- affiche une visualisation 12 derivations sur 10 secondes

3. `Prediction Pipeline`
- exploite `df_meta.pkl`
- groupe les ECG par patient et par date
- construit un dataset temporel
- entraine un modele `RandomForest`
- predit le diagnostic de la periode suivante a partir de l'historique ECG du patient

## Arborescence utile

```text
exofit/
├── backend/
│   ├── main.py
│   ├── prompt_engine.py
│   ├── ecg_pipeline.py
│   ├── prediction_pipeline.py
│   └── requirements.txt
├── data/
│   ├── cas_cliniques/
│   │   ├── raw/
│   │   └── parsed/
│   └── ecg/
│       ├── raw/
│       ├── df_meta.pkl
│       ├── ahp_ecg_usage.ipynb
│       └── patient_ecg_mapping.json
├── frontend/
│   └── index.html
└── README.md
```

## Modules backend

### `backend/main.py`

API FastAPI principale.

Roles :
- expose les routes de diagnostic clinique
- expose les routes ECG
- expose les routes d'entrainement et de prediction
- centralise les schemas `Pydantic`

Routes importantes :
- `POST /api/diagnostic`
- `GET /api/cases`
- `GET /api/cases/{case_id}`
- `POST /api/diagnostic/case`
- `POST /api/ecg/analyser`
- `GET /api/prediction/status`
- `GET /api/prediction/patients`
- `GET /api/prediction/patient/{patient_id}`
- `POST /api/prediction/train`
- `POST /api/prediction/predict`

### `backend/prompt_engine.py`

Roles :
- charge les cas parses depuis `data/cas_cliniques/parsed`
- construit le prompt medical a partir :
  - des donnees patient structurees
  - du texte brut d'origine si disponible
- interroge un provider LLM
- normalise la sortie en JSON exploitable

Fonctions importantes :
- `build_prompt(patient)`
- `load_parsed_cases()`
- `get_available_cases()`
- `get_case_by_id(case_id)`
- `_normalize_result(result)`

Sortie attendue du LLM :
- `diagnostic_preliminaire`
- `explication_diagnostic`
- `diagnostics_differentiels`
- `questions_complementaires`
- `examens_proposes`
- `traitements_proposes`
- `niveau_urgence`
- `confiance`
- `modele_utilise`

### `backend/ecg_pipeline.py`

Roles :
- lecture et preparation des ECG
- extraction de features interpretable
- classification heuristique
- generation ECG synthetique pour demo

Fonctions importantes :
- `read_ecg_csv(file_bytes)`
- `bandpass_filter(ecg, lowcut=0.5, highcut=40.0)`
- `notch_filter(ecg, freq=50.0)`
- `detect_r_peaks(lead_ii)`
- `extract_features(ecg)`
- `diagnose_by_rules(features)`
- `preprocess_ecg(file_bytes)`
- `predict_pathology(preprocessed)`

Features extraites :
- `heart_rate_bpm`
- `rr_mean_ms`
- `rr_std_ms`
- `rr_irregularity_ratio`
- `n_r_peaks`
- `qrs_width_ms`
- `r_amplitude_mean`
- `v5_r_amplitude`
- `st_max_deviation_mm`
- `st_mean_deviation_mm`
- `fibrillation_power_ratio`

Seuils principaux utilises dans les regles :
- fibrillation auriculaire :
  - `rr_irregularity_ratio > 0.18`
  - `fibrillation_power_ratio > 0.15`
  - bonus si `heart_rate_bpm > 90`
- tachycardie ventriculaire :
  - `heart_rate_bpm > 100` et `qrs_width_ms > 120`
  - bonus si `heart_rate_bpm > 130`
- bradycardie :
  - `heart_rate_bpm < 60` avec rythme regulier
  - score fort si `heart_rate_bpm < 50`
- bloc auriculo-ventriculaire :
  - `qrs_width_ms > 120`
  - renforcement si `qrs_width_ms > 140`
- ischemie myocardique :
  - `st_mean_deviation_mm < -0.1`
  - renforcement si `st_max_deviation_mm > 0.1` et `st_mean_deviation_mm < -0.05`
- infarctus du myocarde :
  - `st_mean_deviation_mm > 0.2`
  - renforcement si `st_mean_deviation_mm > 0.4`

Remarque :
- le segment ST est evalue a `80 ms` apres le pic R dans `_estimate_st_deviation(..., offset_ms=80)`

### `backend/prediction_pipeline.py`

Roles :
- charge `df_meta.pkl`
- normalise les diagnostics ECG
- regroupe les ECG par patient
- ordonne les ECG par date
- construit des echantillons temporels
- entraine un `RandomForest`
- evalue le modele
- predit la pathologie de la periode suivante

Logique temporelle :
- pour un patient avec plusieurs ECG ordonnes par date :
  - entree = moyenne des features des ECG precedents
  - cible = diagnostic normalise de l'ECG suivant

Fonctions importantes :
- `load_ecg_metadata()`
- `normalize_diagnosis_label(value)`
- `build_prediction_dataset_from_metadata()`
- `split_train_test_by_patient()`
- `train_random_forest_model(train_samples)`
- `evaluate_model(model, test_samples)`
- `train_prediction_pipeline(test_ratio=0.2, save_model=True)`
- `predict_future_disease(patient, ecg_files, model=None)`

Features utilisees par la prediction :
- toutes les features ECG du pipeline
- plus :
  - `age`
  - `sex_male`
  - `symptom_count`
  - `antecedent_count`
  - `exam_count`

Modele actuel :
- `RandomForestClassifier`
- `n_estimators=300`
- `max_depth=14`
- `min_samples_leaf=2`
- `class_weight="balanced_subsample"`
- `random_state=42`

Evaluation :
- split `train/test` par patient
- metriques :
  - `accuracy`
  - `macro_precision`
  - `macro_recall`
  - `macro_f1`
  - `per_label`

## Frontend

### `frontend/index.html`

Page unique HTML/CSS/JS.

Contenu principal :
- formulaire patient
- comparaison de modeles generatifs
- import ECG ou generation ECG synthetique
- visualisation ECG 12 derivations
- onglet prediction
- affichage des metriques d'entrainement

Le frontend communique avec le backend via `fetch()` sur `http://localhost:8000/api`.

## Donnees

### Cas cliniques

- `data/cas_cliniques/raw` : documents d'origine du professeur
- `data/cas_cliniques/parsed` : cas transformes en JSON

Chaque cas parse contient en general :
- `id`
- `description`
- `patient`
- `source_document`
- parfois du texte clinique brut

### Donnees ECG

- `data/ecg/raw` : fichiers ECG CSV
- `data/ecg/df_meta.pkl` : metadonnees associees aux ECG

Colonnes importantes observees dans `df_meta.pkl` :
- `patient_id`
- `age`
- `gender`
- `date`
- `diagnosis`
- `original_diagnosis`
- `ecg_file_path`

## Installation

### 1. Creer un environnement virtuel

```powershell
python -m venv .venv
```

### 2. Activer l'environnement

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Installer les dependances

```powershell
pip install -r backend/requirements.txt
pip install pandas scikit-learn
```

Si le backend utilise les uploads de fichiers FastAPI :

```powershell
pip install python-multipart
```

## Configuration `.env`

Le fichier est place dans `backend/.env`.

Exemple minimal :

```env
ACTIVE_LLM=google

GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-2.5-flash

QW_TOKEN=hf_...
QW_MODEL=Qwen/Qwen2.5-7B-Instruct

DS_TOKEN=hf_...
DS_MODEL=deepseek-ai/DeepSeek-V3-0324

MISTRAL_API_KEY=...
MISTRAL_MODEL=mistral-small-latest
```

Remarques :
- une seule ligne `ACTIVE_LLM=...`
- les cles API doivent rester privees
- si aucune cle n'est disponible, certains providers passent en mode demo

## Lancement

### Backend

Depuis la racine :

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

### Frontend

Depuis la racine :

```powershell
cd frontend
python -m http.server 3000
```

Puis ouvrir :

```text
http://localhost:3000
```

## Execution des modules en ligne de commande

### Prompt engine

```powershell
python backend/prompt_engine.py
```

### ECG pipeline

```powershell
python backend/ecg_pipeline.py
```

### Prediction pipeline

Depuis la racine :

```powershell
python backend/prediction_pipeline.py
```

Depuis `backend/` :

```powershell
python prediction_pipeline.py
```

Le script :
- construit le dataset temporel
- entraine le RandomForest
- evalue le modele
- sauvegarde `backend/prediction_model.json`
- affiche un resume JSON

## Etapes techniques codees

### Partie 1 : diagnostic clinique

1. chargement des cas parses
2. construction d'un prompt structure
3. appel LLM
4. normalisation de la sortie
5. affichage dans le frontend

### Partie 2 : analyse ECG

1. lecture du fichier ECG
2. filtrage passe-bande
3. filtrage notch 50 Hz
4. detection des pics R
5. extraction des features
6. classification heuristique
7. visualisation 12 derivations

### Partie 3 : prediction future

1. chargement de `df_meta.pkl`
2. normalisation des diagnostics
3. regroupement des ECG par patient
4. tri par date
5. construction des echantillons temporels
6. split train/test par patient
7. entrainement du RandomForest
8. calcul des metriques
9. prediction sur un patient a partir de son historique ECG

## Fichiers generes

- `backend/prediction_model.json`
- eventuellement `backend/prompt_engine_results.json`

## Limites actuelles

- la partie ECG est principalement basee sur des regles heuristiques interpretable
- la prediction future repose sur des features agregees et un RandomForest, pas sur un modele sequence profond
- la partie LLM fournit un diagnostic preliminaire, pas un diagnostic medical certifie
- le projet reste une preuve de concept academique

## Resume soutenance

- `prompt_engine.py` : aide au diagnostic clinique par LLM
- `ecg_pipeline.py` : analyse interpretable du signal ECG 12 derivations
- `prediction_pipeline.py` : prediction temporelle du prochain diagnostic ECG a partir de l'historique patient

