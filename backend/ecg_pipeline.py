"""
EXOFIT — Pipeline ECG (Partie 2)
Prétraitement + extraction de features + modèle CNN 1D
Fonctionne avec des données synthétiques en attendant les 30 000 ECG réels.
"""

import os
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

# ─── CONSTANTES ECG ───────────────────────────────────────────────────────

FS = 500            # Fréquence d'échantillonnage standard (Hz)
WINDOW_SEC = 10     # Fenêtre d'analyse (secondes)
WINDOW_SAMPLES = FS * WINDOW_SEC  # 5000 échantillons par fenêtre
N_LEADS = 12        # 12 dérivations standard

RAW_ECG_DIR = Path(__file__).resolve().parents[1] / "data" / "ecg" / "raw"
PREVIEW_SAMPLES = 2500

PATHOLOGIES = [
    "normal",
    "fibrillation_auriculaire",
    "tachycardie_ventriculaire",
    "bradycardie",
    "bloc_auriculo_ventriculaire",
    "hypertrophie_ventriculaire_gauche",
    "ischemie_myocardique",
    "infarctus_du_myocarde",
]

PATHOLOGIE_LABELS = {
    "normal":                          "Rythme normal (sinusal)",
    "fibrillation_auriculaire":        "Fibrillation auriculaire",
    "tachycardie_ventriculaire":       "Tachycardie ventriculaire",
    "bradycardie":                     "Bradycardie",
    "bloc_auriculo_ventriculaire":     "Bloc auriculo-ventriculaire",
    "hypertrophie_ventriculaire_gauche": "Hypertrophie ventriculaire gauche",
    "ischemie_myocardique":            "Ischémie myocardique",
    "infarctus_du_myocarde":           "Infarctus du myocarde",
}

# ─── ÉTAPE 1 : LECTURE DU CSV ─────────────────────────────────────────────

def list_available_ecg_files(limit: int = 500) -> dict:
    """Liste les ECG CSV disponibles dans data/ecg/raw."""
    if not RAW_ECG_DIR.exists():
        return {"total": 0, "returned": 0, "directory": str(RAW_ECG_DIR), "files": []}

    files = sorted(RAW_ECG_DIR.glob("*.csv"))
    selected = files[:limit]
    return {
        "total": len(files),
        "returned": len(selected),
        "directory": str(RAW_ECG_DIR),
        "files": [
            {
                "id": file.name,
                "filename": file.name,
                "size_bytes": file.stat().st_size,
            }
            for file in selected
        ],
    }


def load_ecg_file_bytes(file_id: str) -> bytes:
    """Charge un ECG brut a partir de son nom de fichier."""
    candidate = RAW_ECG_DIR / Path(file_id).name
    if not candidate.exists() or candidate.suffix.lower() != ".csv":
        raise FileNotFoundError(f"ECG introuvable: {file_id}")
    return candidate.read_bytes()


def read_ecg_csv(file_bytes: bytes) -> np.ndarray:
    """
    Lit un ECG au format CSV.
    Format attendu : 12 colonnes (une par dérivation), N lignes (échantillons).
    Retourne un array (N_samples, 12).
    """
    content = file_bytes.decode("utf-8-sig", errors="ignore")
    lines = content.strip().splitlines()

    # Ignorer les lignes d'en-tête potentielles
    data = []
    for line in lines:
        parts = [part.strip() for part in line.strip().split(",") if part.strip()]
        try:
            values = [float(p) for p in parts]
            if len(values) >= 1:
                data.append(values)
        except ValueError:
            continue  # Ligne d'en-tête ou corrompue

    arr = np.array(data)

    if arr.size == 0:
        raise ValueError("Aucune donnee ECG exploitable n'a ete trouvee dans le CSV")

    # Adapter si moins de 12 colonnes (ECG 6 dérivations)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] < 12:
        # Dupliquer les colonnes pour simuler 12 dérivations (mode dégradé)
        arr = np.tile(arr, (1, 12 // arr.shape[1] + 1))[:, :12]

    arr = _normalize_ecg_units(arr)
    return arr


def _normalize_ecg_units(ecg: np.ndarray) -> np.ndarray:
    """
    Ramene les amplitudes vers une echelle proche du mV.
    Les exports MUSE sont frequemment en microvolts.
    """
    if ecg.size == 0:
        return ecg

    if float(np.max(np.abs(ecg))) > 20:
        return ecg / 1000.0
    return ecg


def get_preview_signal(ecg: np.ndarray, lead_index: int = 1, n_samples: int = PREVIEW_SAMPLES) -> list[float]:
    """Retourne un extrait de la derivation II pour l'aperçu frontend."""
    if ecg.ndim != 2 or ecg.shape[0] == 0:
        return []
    safe_lead = min(max(lead_index, 0), ecg.shape[1] - 1)
    preview = ecg[:n_samples, safe_lead]
    return [round(float(value), 4) for value in preview]


def get_preview_12_leads(ecg: np.ndarray, n_samples: int = WINDOW_SAMPLES) -> dict[str, list[float]]:
    """Retourne 10 secondes des 12 derivations pour affichage frontend."""
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    if ecg.ndim != 2 or ecg.shape[0] == 0:
        return {lead: [] for lead in lead_names}

    previews = {}
    for index, lead in enumerate(lead_names):
        if index >= ecg.shape[1]:
            previews[lead] = []
            continue
        previews[lead] = [round(float(value), 4) for value in ecg[:n_samples, index]]
    return previews


# ─── ÉTAPE 2 : FILTRAGE ───────────────────────────────────────────────────

def bandpass_filter(ecg: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0) -> np.ndarray:
    """
    Filtre passe-bande Butterworth 4e ordre.
    Supprime : drift de ligne de base (< 0.5 Hz) et bruit EMG (> 40 Hz).
    """
    nyq = FS / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(4, [low, high], btype="band")
    filtered = np.zeros_like(ecg)
    for i in range(ecg.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, ecg[:, i])
    return filtered


def notch_filter(ecg: np.ndarray, freq: float = 50.0) -> np.ndarray:
    """Filtre coupe-bande pour éliminer le bruit secteur (50 Hz en Europe)."""
    b, a = signal.iirnotch(freq, Q=30, fs=FS)
    filtered = np.zeros_like(ecg)
    for i in range(ecg.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, ecg[:, i])
    return filtered


# ─── ÉTAPE 3 : DÉTECTION DES PICS R (Pan-Tompkins simplifié) ─────────────

def detect_r_peaks(lead_ii: np.ndarray) -> np.ndarray:
    """
    Détection des pics R sur la dérivation II.
    Utilise la dérivée + seuil adaptatif.
    Retourne les indices des pics R.
    """
    # Dérivée du signal
    diff = np.diff(lead_ii)
    # Mise au carré pour accentuer les pics
    squared = diff ** 2
    # Intégration mobile (fenêtre 150ms)
    window = int(0.15 * FS)
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")
    # Seuil adaptatif : 50% du max
    threshold = 0.5 * np.max(integrated)
    # Détection des pics avec distance minimale (200ms entre battements)
    min_dist = int(0.2 * FS)
    peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_dist)
    return peaks


# ─── ÉTAPE 4 : EXTRACTION DE FEATURES ────────────────────────────────────

def extract_features(ecg: np.ndarray) -> dict:
    """
    Extrait les features diagnostiques d'un ECG filtré.
    Ces features sont utilisées à la fois pour le modèle IA
    et pour l'approche mathématique par règles.
    """
    # Utiliser la dérivation II (index 1) pour les intervalles rythmiques
    lead_ii = ecg[:WINDOW_SAMPLES, 1] if ecg.shape[0] >= WINDOW_SAMPLES else ecg[:, 1]

    r_peaks = detect_r_peaks(lead_ii)

    # ── Intervalles RR ──
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / FS * 1000  # en ms
        rr_mean = float(np.mean(rr_intervals))
        rr_std = float(np.std(rr_intervals))
        heart_rate = int(60000 / rr_mean) if rr_mean > 0 else 0
        rr_irregularity = float(rr_std / rr_mean) if rr_mean > 0 else 0
    else:
        rr_mean = rr_std = rr_irregularity = 0.0
        heart_rate = 0

    # ── Amplitude des ondes R ──
    r_amplitudes = lead_ii[r_peaks] if len(r_peaks) > 0 else np.array([0.0])
    r_mean_amp = float(np.mean(r_amplitudes))

    # ── Analyse spectrale (Fourier) ──
    n = len(lead_ii)
    spectrum = np.abs(fft(lead_ii))[:n // 2]
    freqs = fftfreq(n, 1 / FS)[:n // 2]
    # Puissance dans la bande fibrillation (4-12 Hz)
    fib_mask = (freqs >= 4) & (freqs <= 12)
    total_power = float(np.sum(spectrum))
    fib_power = float(np.sum(spectrum[fib_mask])) / total_power if total_power > 0 else 0

    # ── Features sur dérivation V1 pour QRS (index 6) ──
    lead_v1 = ecg[:WINDOW_SAMPLES, 6] if ecg.shape[0] >= WINDOW_SAMPLES else ecg[:, 6]

    # Largeur QRS approximative (détection des croisements de zéro autour des pics R)
    qrs_width_ms = _estimate_qrs_width(lead_v1, r_peaks)

    # ── Features dérivation V5 pour amplitude onde R (HVG) ──
    lead_v5 = ecg[:WINDOW_SAMPLES, 10] if ecg.shape[0] >= WINDOW_SAMPLES else ecg[:, 10]
    v5_r_amp = float(np.max(np.abs(lead_v5)))

    # ── ST segment (approximation sur dérivations V1-V4) ──
    st_deviations = []
    for lead_idx in [6, 7, 8, 9]:  # V1-V4
        lead = ecg[:WINDOW_SAMPLES, lead_idx] if ecg.shape[0] >= WINDOW_SAMPLES else ecg[:, lead_idx]
        st_dev = _estimate_st_deviation(lead, r_peaks)
        st_deviations.append(st_dev)
    st_max_deviation = float(np.max(np.abs(st_deviations))) if st_deviations else 0.0
    st_mean_deviation = float(np.mean(st_deviations)) if st_deviations else 0.0

    return {
        # Rythme
        "heart_rate_bpm": heart_rate,
        "rr_mean_ms": round(rr_mean, 1),
        "rr_std_ms": round(rr_std, 1),
        "rr_irregularity_ratio": round(rr_irregularity, 3),
        "n_r_peaks": len(r_peaks),
        # Morphologie
        "qrs_width_ms": round(qrs_width_ms, 1),
        "r_amplitude_mean": round(r_mean_amp, 3),
        "v5_r_amplitude": round(v5_r_amp, 3),
        # Segment ST
        "st_max_deviation_mm": round(st_max_deviation, 2),
        "st_mean_deviation_mm": round(st_mean_deviation, 2),
        # Spectral
        "fibrillation_power_ratio": round(fib_power, 4),
    }


def _estimate_qrs_width(lead: np.ndarray, r_peaks: np.ndarray, margin_ms: int = 60) -> float:
    """Estime la largeur du complexe QRS en ms."""
    if len(r_peaks) == 0:
        return 0.0
    margin = int(margin_ms / 1000 * FS)
    widths = []
    for rp in r_peaks[:10]:  # Limiter aux 10 premiers pour la performance
        start = max(0, rp - margin)
        end = min(len(lead) - 1, rp + margin)
        segment = lead[start:end]
        if len(segment) < 4:
            continue
        threshold = 0.1 * np.max(np.abs(segment))
        above = np.where(np.abs(segment) > threshold)[0]
        if len(above) > 1:
            widths.append((above[-1] - above[0]) / FS * 1000)
    return float(np.mean(widths)) if widths else 80.0  # 80ms = valeur normale


def _estimate_st_deviation(lead: np.ndarray, r_peaks: np.ndarray, offset_ms: int = 80) -> float:
    """Estime la déviation du segment ST (80ms après le pic R)."""
    if len(r_peaks) == 0:
        return 0.0
    offset = int(offset_ms / 1000 * FS)
    deviations = []
    baseline = float(np.median(lead))
    for rp in r_peaks[:10]:
        st_idx = rp + offset
        if st_idx < len(lead):
            deviations.append(lead[st_idx] - baseline)
    return float(np.mean(deviations)) if deviations else 0.0


# ─── ÉTAPE 5 : APPROCHE MATHÉMATIQUE (règles diagnostiques) ───────────────

def diagnose_by_rules(features: dict) -> dict:
    """
    Diagnostic basé sur des règles cliniques établies.
    Transparent et explicable. Sert de baseline et de validation.
    """
    hr = features["heart_rate_bpm"]
    irr = features["rr_irregularity_ratio"]
    qrs = features["qrs_width_ms"]
    fib_power = features["fibrillation_power_ratio"]
    st_max = features["st_max_deviation_mm"]
    st_mean = features["st_mean_deviation_mm"]
    v5_r = features["v5_r_amplitude"]
    n_r_peaks = features["n_r_peaks"]

    scores = {p: 0.0 for p in PATHOLOGIES}

    # ── Fibrillation auriculaire ──
    # Critères : rythme irrégulier + puissance spectrale 4-12 Hz élevée
    if irr > 0.18 and fib_power > 0.25:
        scores["fibrillation_auriculaire"] += 0.45
    if irr > 0.24 and fib_power > 0.30:
        scores["fibrillation_auriculaire"] += 0.25
    if irr > 0.30 and fib_power > 0.35:
        scores["fibrillation_auriculaire"] += 0.15

    # ── Tachycardie ventriculaire ──
    # Critères : FC > 100 bpm + QRS larges (> 120ms)
    if hr > 100 and qrs > 120:
        scores["tachycardie_ventriculaire"] += 0.7
    elif hr > 130:
        scores["tachycardie_ventriculaire"] += 0.2

    # ── Bradycardie ──
    if hr < 60 and hr > 0 and irr < 0.12:
        scores["bradycardie"] += 0.8
    elif hr < 50 and irr < 0.12:
        scores["bradycardie"] = 0.95
    elif hr < 60 and irr >= 0.12:
        scores["bradycardie"] += 0.2

    # ── Bloc auriculo-ventriculaire ──
    # Critères : QRS larges sans tachycardie
    if qrs > 120 and hr <= 100:
        scores["bloc_auriculo_ventriculaire"] += 0.5
    if qrs > 160:
        scores["bloc_auriculo_ventriculaire"] += 0.3

    # ── Hypertrophie ventriculaire gauche ──
    # Critère Sokolow-Lyon approximé via amplitude V5
    if v5_r > 2.5 and irr < 0.20:  # > 2.5 mV ~ 25mm
        scores["hypertrophie_ventriculaire_gauche"] += 0.6
    if v5_r > 3.5 and irr < 0.20:
        scores["hypertrophie_ventriculaire_gauche"] += 0.3

    # ── Ischémie myocardique ──
    # Critère : sous-décalage ST
    if st_mean < -0.1 and irr < 0.20:  # Sous-décalage
        scores["ischemie_myocardique"] += 0.5
    if st_max > 0.1 and st_mean < -0.05 and irr < 0.20:
        scores["ischemie_myocardique"] += 0.3

    # ── Infarctus du myocarde ──
    # Critère : sus-décalage ST > 0.2 mV
    if st_mean > 0.2 and irr < 0.18:
        scores["infarctus_du_myocarde"] += 0.7
    if st_mean > 0.4 and irr < 0.18:
        scores["infarctus_du_myocarde"] += 0.25

    if n_r_peaks < 12 and irr > 0.18 and fib_power > 0.30:
        scores["fibrillation_auriculaire"] += 0.15

    # ── Normal ──
    if irr < 0.12 and fib_power < 0.30 and 60 <= hr <= 100:
        scores["normal"] += 0.35

    max_score = max(scores.values())
    if max_score < 0.3:
        scores["normal"] = 0.8

    # Normaliser en probabilités
    base_floor = 0.03
    smoothed_scores = {
        pathology: score + (base_floor if score == 0 else 0.0)
        for pathology, score in scores.items()
    }

    total = sum(smoothed_scores.values())
    if total > 0:
        probs = {k: round(v / total, 3) for k, v in smoothed_scores.items()}
    else:
        probs = {k: round(1.0 / len(PATHOLOGIES), 3) for k in PATHOLOGIES}

    best = max(probs, key=probs.get)
    return {
        "methode": "règles_mathematiques",
        "pathologie": best,
        "probabilites": probs,
        "confiance": probs[best]
    }


# ─── ÉTAPE 6 : MODÈLE CNN (architecture, entraînable sur les données réelles) ──

def build_cnn_model():
    """
    Architecture CNN 1D pour la classification ECG.
    À entraîner sur les 30 000 ECG réels.
    Nécessite : pip install tensorflow
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models

        model = models.Sequential([
            # Entrée : (WINDOW_SAMPLES, N_LEADS)
            layers.Input(shape=(WINDOW_SAMPLES, N_LEADS)),

            # Bloc 1 : features à grande échelle (rythme global)
            layers.Conv1D(32, kernel_size=50, strides=2, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=5),

            # Bloc 2 : features à échelle intermédiaire (complexes QRS)
            layers.Conv1D(64, kernel_size=15, strides=1, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=5),

            # Bloc 3 : features fines (ondes P, T)
            layers.Conv1D(128, kernel_size=5, strides=1, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=5),

            # Bloc 4 : features contextuelles
            layers.Conv1D(256, kernel_size=3, padding="same", activation="relu"),
            layers.GlobalAveragePooling1D(),

            # Classification
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(len(PATHOLOGIES), activation="softmax"),
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    except ImportError:
        return None


def load_model_weights(model_path: str = "ecg_model.h5"):
    """Charge les poids du modèle entraîné (à appeler après réception des données)."""
    try:
        import tensorflow as tf
        model = build_cnn_model()
        if model and os.path.exists(model_path):
            model.load_weights(model_path)
            return model
    except Exception:
        pass
    return None


# ─── PIPELINE COMPLET ─────────────────────────────────────────────────────

def preprocess_ecg(file_bytes: bytes) -> dict:
    """
    Pipeline complet de prétraitement.
    Retourne les features extraites.
    """
    raw = read_ecg_csv(file_bytes)
    filtered = bandpass_filter(raw)
    filtered = notch_filter(filtered)
    features = extract_features(filtered)
    return {"features": features, "signal": filtered}


def predict_pathology(preprocessed: dict) -> dict:
    """
    Prédit la pathologie à partir du signal prétraité.
    Utilise le CNN si disponible, sinon les règles mathématiques.
    """
    import os
    features = preprocessed["features"]
    signal_data = preprocessed["signal"]

    # Essayer le CNN d'abord
    model = load_model_weights()
    if model is not None:
        window = signal_data[:WINDOW_SAMPLES]
        if window.shape[0] < WINDOW_SAMPLES:
            pad = np.zeros((WINDOW_SAMPLES - window.shape[0], window.shape[1]))
            window = np.vstack([window, pad])
        probs_array = model.predict(window[np.newaxis])[0]
        probs = {PATHOLOGIES[i]: float(p) for i, p in enumerate(probs_array)}
        methode = "CNN 1D"
    else:
        # Fallback : règles mathématiques
        result = diagnose_by_rules(features)
        probs = result["probabilites"]
        methode = "règles mathématiques"

    best_pathologie = max(probs, key=probs.get)
    confiance = probs[best_pathologie]

    # Recommandation clinique
    recommandation = _get_recommandation(best_pathologie, confiance)

    return {
        "pathologie_detectee": PATHOLOGIE_LABELS[best_pathologie],
        "pathologie_id": best_pathologie,
        "probabilites": {PATHOLOGIE_LABELS[k]: v for k, v in probs.items()},
        "features_extraites": features,
        "confiance": round(confiance, 3),
        "recommandation": recommandation,
        "methode": methode,
    }


def _get_recommandation(pathologie: str, confiance: float) -> str:
    recommandations = {
        "infarctus_du_myocarde":         "URGENCE ABSOLUE — Appeler SAMU, aspiriné 250mg, défibrillateur en standby",
        "tachycardie_ventriculaire":      "URGENCE — Monitoring continu, préparer choc électrique externe",
        "fibrillation_auriculaire":       "Consulter médecin téléconsultant — anticoagulation à évaluer",
        "ischemie_myocardique":           "Consultation cardiologique urgente — trinitrine sublinguale si douleur",
        "bloc_auriculo_ventriculaire":    "ECG 12 dérivations complet requis — téléconsultation cardiologique",
        "hypertrophie_ventriculaire_gauche": "Consultation cardiologique programmée — bilan HTA",
        "bradycardie":                    "Surveiller FC, consulter si symptômes (syncope, dyspnée)",
        "normal":                         "ECG dans les limites normales — poursuivre bilan clinique",
    }
    base = recommandations.get(pathologie, "Téléconsultation médicale recommandée")
    if confiance < 0.5:
        base += " (confiance faible — interprétation à valider par médecin)"
    return base


# ─── DONNÉES SYNTHÉTIQUES (pour tests sans CSV réel) ──────────────────────

# Redefinition volontaire pour garder une recommandation adaptee
# au contexte d'une infirmiere deja aupres du patient.
def _get_recommandation(pathologie: str, confiance: float) -> str:
    recommandations = {
        "infarctus_du_myocarde": "URGENCE ABSOLUE - Surveillance infirmiere rapprochee, alerte medicale immediate, preparation du materiel d'urgence et application du protocole local valide medicalement",
        "tachycardie_ventriculaire": "URGENCE - Monitoring continu, alerte medicale immediate, preparation du defibrillateur et du materiel de reanimation selon protocole local",
        "fibrillation_auriculaire": "Teleconsultation medicale rapide et surveillance clinique, avec adaptation therapeutique a valider par le medecin",
        "ischemie_myocardique": "Avis medical urgent, surveillance rapprochee et mise en condition du patient selon protocole local",
        "bloc_auriculo_ventriculaire": "ECG complet requis et teleconsultation cardiologique rapide avec surveillance de la tolerance clinique",
        "hypertrophie_ventriculaire_gauche": "Orientation cardiologique programmee et bilan tensionnel a organiser",
        "bradycardie": "Surveiller la frequence cardiaque et la tolerance clinique, avec avis medical si symptomes ou aggravation",
        "normal": "ECG globalement rassurant - poursuivre l'evaluation clinique infirmiere et demander une validation medicale si necessaire",
    }
    base = recommandations.get(pathologie, "Teleconsultation medicale recommandee")
    if confiance < 0.5:
        base += " (confiance faible - interpretation a valider par un medecin)"
    return base


def generate_synthetic_ecg(pathologie: str = "normal", duration_sec: int = 10) -> np.ndarray:
    """
    Génère un ECG synthétique pour les tests du pipeline.
    Reproduit les caractéristiques diagnostiques clés de chaque pathologie.
    """
    n_samples = FS * duration_sec
    t = np.linspace(0, duration_sec, n_samples)
    ecg = np.zeros((n_samples, 12))

    if pathologie == "normal":
        hr = 70
        rr = FS * 60 / hr
        for lead in range(12):
            wave = _generate_pqrst(t, hr=hr, qrs_width=0.08, st_dev=0.0, noise=0.02)
            ecg[:, lead] = wave

    elif pathologie == "fibrillation_auriculaire":
        # Rythme irrégulier + trémulations basales
        for lead in range(12):
            wave = _generate_af_signal(t)
            ecg[:, lead] = wave

    elif pathologie == "bradycardie":
        for lead in range(12):
            wave = _generate_pqrst(t, hr=42, qrs_width=0.08, st_dev=0.0, noise=0.02)
            ecg[:, lead] = wave

    elif pathologie == "infarctus_du_myocarde":
        for lead_idx in range(12):
            st_elevation = 0.3 if lead_idx in [6, 7, 8, 9] else 0.0  # V1-V4
            wave = _generate_pqrst(t, hr=95, qrs_width=0.10, st_dev=st_elevation, noise=0.03)
            ecg[:, lead_idx] = wave

    return ecg


def _generate_pqrst(t, hr=70, qrs_width=0.08, st_dev=0.0, noise=0.02) -> np.ndarray:
    """Génère un signal PQRST synthétique."""
    rr = 60.0 / hr
    wave = np.zeros(len(t))
    beat_times = np.arange(0, t[-1], rr)

    for bt in beat_times:
        # Onde P
        p_t = t - (bt + 0.10)
        wave += 0.15 * np.exp(-p_t**2 / (2 * 0.01**2))
        # Complexe QRS
        qrs_t = t - (bt + 0.18)
        wave -= 0.05 * np.exp(-qrs_t**2 / (2 * (qrs_width * 0.3)**2))
        wave += 1.00 * np.exp(-qrs_t**2 / (2 * (qrs_width * 0.15)**2))
        wave -= 0.20 * np.exp(-(t - (bt + 0.18 + qrs_width * 0.4))**2 / (2 * (qrs_width * 0.2)**2))
        # Segment ST + onde T
        st_t = t - (bt + 0.32)
        wave += st_dev * np.exp(-st_t**2 / (2 * 0.03**2))
        t_t = t - (bt + 0.38)
        wave += 0.30 * np.exp(-t_t**2 / (2 * 0.025**2))

    wave += np.random.normal(0, noise, len(t))
    return wave


def _generate_af_signal(t) -> np.ndarray:
    """Génère un signal de fibrillation auriculaire (rythme irrégulier + trémulations)."""
    wave = np.zeros(len(t))
    # Trémulations atriales (4-12 Hz)
    wave += 0.05 * np.sin(2 * np.pi * 6 * t + np.random.uniform(0, np.pi))
    wave += 0.03 * np.sin(2 * np.pi * 9 * t)
    # Complexes QRS irréguliers
    rr_base = 60.0 / 110  # FC moyenne 110 bpm
    current_t = 0.1
    while current_t < t[-1]:
        rr = rr_base * (1 + np.random.uniform(-0.3, 0.3))
        qrs_t = t - current_t
        wave += 0.80 * np.exp(-qrs_t**2 / (2 * 0.008**2))
        wave -= 0.15 * np.exp(-(t - (current_t - 0.02))**2 / (2 * 0.005**2))
        wave += 0.25 * np.exp(-(t - (current_t + 0.25))**2 / (2 * 0.025**2))
        current_t += rr
    wave += np.random.normal(0, 0.025, len(t))
    return wave


if __name__ == "__main__":
    print("=== Test du pipeline ECG ===\n")

    for patho in ["normal", "fibrillation_auriculaire", "bradycardie", "infarctus_du_myocarde"]:
        print(f"--- Test : {patho} ---")
        synth = generate_synthetic_ecg(patho)
        filtered = bandpass_filter(synth)
        features = extract_features(filtered)
        result = diagnose_by_rules(features)
        print(f"FC détectée      : {features['heart_rate_bpm']} bpm")
        print(f"Irrégularité RR  : {features['rr_irregularity_ratio']}")
        print(f"Largeur QRS      : {features['qrs_width_ms']} ms")
        print(f"Déviation ST moy : {features['st_mean_deviation_mm']} mm")
        print(f"Diagnostic règles: {result['pathologie']} (confiance {result['confiance']:.2f})")
        print()
