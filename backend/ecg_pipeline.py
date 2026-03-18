"""
EXOFIT â€” Pipeline ECG (Partie 2)
PrÃ©traitement + extraction de features + modÃ¨le CNN 1D
Fonctionne avec des donnÃ©es synthÃ©tiques en attendant les 30 000 ECG rÃ©els.
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

# â”€â”€â”€ CONSTANTES ECG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

FS = 500            # FrÃ©quence d'Ã©chantillonnage standard (Hz)
WINDOW_SEC = 10     # FenÃªtre d'analyse (secondes)
WINDOW_SAMPLES = FS * WINDOW_SEC  # 5000 Ã©chantillons par fenÃªtre
N_LEADS = 12        # 12 dÃ©rivations standard

RAW_ECG_DIR = Path(__file__).resolve().parents[1] / "data" / "ecg" / "raw"
CNN_MODEL_PATH = Path(__file__).resolve().parent / "ecg_cnn_model.keras"
CNN_META_PATH = Path(__file__).resolve().parent / "ecg_cnn_model_meta.json"
PREVIEW_SAMPLES = 2500
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_GROUPS = {
    "inferieur": ["II", "III", "aVF"],
    "lateral": ["I", "aVL", "V5", "V6"],
    "anterieur": ["V1", "V2", "V3", "V4"],
}

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
    "bloc_auriculo_ventriculaire":     "Bloc de branche probable",
    "hypertrophie_ventriculaire_gauche": "Hypertrophie ventriculaire gauche",
    "ischemie_myocardique":            "IschÃ©mie myocardique",
    "infarctus_du_myocarde":           "Infarctus du myocarde",
}

# â”€â”€â”€ Ã‰TAPE 1 : LECTURE DU CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    Format attendu : 12 colonnes (une par dÃ©rivation), N lignes (Ã©chantillons).
    Retourne un array (N_samples, 12).
    """
    content = file_bytes.decode("utf-8-sig", errors="ignore")
    lines = content.strip().splitlines()

    # Ignorer les lignes d'en-tÃªte potentielles
    data = []
    for line in lines:
        parts = [part.strip() for part in line.strip().split(",") if part.strip()]
        try:
            values = [float(p) for p in parts]
            if len(values) >= 1:
                data.append(values)
        except ValueError:
            continue  # Ligne d'en-tÃªte ou corrompue

    arr = np.array(data)

    if arr.size == 0:
        raise ValueError("Aucune donnee ECG exploitable n'a ete trouvee dans le CSV")

    # Adapter si moins de 12 colonnes (ECG 6 dÃ©rivations)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] < 12:
        # Dupliquer les colonnes pour simuler 12 dÃ©rivations (mode dÃ©gradÃ©)
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
    """Retourne un extrait de la derivation II pour l'aperÃ§u frontend."""
    if ecg.ndim != 2 or ecg.shape[0] == 0:
        return []
    safe_lead = min(max(lead_index, 0), ecg.shape[1] - 1)
    preview = ecg[:n_samples, safe_lead]
    return [round(float(value), 4) for value in preview]


def get_preview_12_leads(ecg: np.ndarray, n_samples: int = WINDOW_SAMPLES) -> dict[str, list[float]]:
    """Retourne 10 secondes des 12 derivations pour affichage frontend."""
    if ecg.ndim != 2 or ecg.shape[0] == 0:
        return {lead: [] for lead in LEAD_NAMES}

    previews = {}
    for index, lead in enumerate(LEAD_NAMES):
        if index >= ecg.shape[1]:
            previews[lead] = []
            continue
        previews[lead] = [round(float(value), 4) for value in ecg[:n_samples, index]]
    return previews


# â”€â”€â”€ Ã‰TAPE 2 : FILTRAGE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    """Filtre coupe-bande pour Ã©liminer le bruit secteur (50 Hz en Europe)."""
    b, a = signal.iirnotch(freq, Q=30, fs=FS)
    filtered = np.zeros_like(ecg)
    for i in range(ecg.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, ecg[:, i])
    return filtered


def remove_baseline_wander(
    ecg: np.ndarray,
    short_window_ms: int = 200,
    long_window_ms: int = 600,
) -> np.ndarray:
    """
    Corrige la derive lente de ligne de base par double filtre median.
    Tres utile sur les ECG avec grandes oscillations lentes.
    """
    corrected = np.zeros_like(ecg)
    short_kernel = max(3, int(short_window_ms / 1000 * FS) | 1)
    long_kernel = max(5, int(long_window_ms / 1000 * FS) | 1)

    for i in range(ecg.shape[1]):
        baseline_short = signal.medfilt(ecg[:, i], kernel_size=short_kernel)
        baseline = signal.medfilt(baseline_short, kernel_size=long_kernel)
        corrected[:, i] = ecg[:, i] - baseline
    return corrected


def smooth_ecg(ecg: np.ndarray, window_ms: int = 18, polyorder: int = 2) -> np.ndarray:
    """
    Lissage leger pour attenuer les artefacts rapides sans ecraser le QRS.
    """
    smoothed = np.zeros_like(ecg)
    window = max(5, int(window_ms / 1000 * FS) | 1)
    for i in range(ecg.shape[1]):
        smoothed[:, i] = signal.savgol_filter(ecg[:, i], window_length=window, polyorder=polyorder)
    return smoothed


# â”€â”€â”€ Ã‰TAPE 3 : DÃ‰TECTION DES PICS R (Pan-Tompkins simplifiÃ©) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def detect_r_peaks(lead_ii: np.ndarray) -> np.ndarray:
    """
    DÃ©tection des pics R sur la dÃ©rivation II.
    Utilise la dÃ©rivÃ©e + seuil adaptatif.
    Retourne les indices des pics R.
    """
    if len(lead_ii) < FS:
        return np.array([], dtype=int)

    centered = lead_ii - np.median(lead_ii)
    energy = np.abs(centered)
    smooth_window = max(5, int(0.08 * FS) | 1)
    envelope = signal.savgol_filter(energy, window_length=smooth_window, polyorder=2)

    min_dist = int(0.35 * FS)
    prominence = max(float(np.percentile(envelope, 75)) * 0.35, 0.02)
    height = max(float(np.percentile(envelope, 80)) * 0.40, 0.03)
    peaks, _ = signal.find_peaks(envelope, distance=min_dist, prominence=prominence, height=height)

    refined = []
    search_radius = int(0.05 * FS)
    for peak in peaks:
        start = max(0, peak - search_radius)
        end = min(len(lead_ii), peak + search_radius)
        segment = lead_ii[start:end]
        if len(segment) == 0:
            continue
        local_idx = int(np.argmax(np.abs(segment)))
        refined.append(start + local_idx)

    if not refined:
        return np.array([], dtype=int)

    peaks = np.array(sorted(set(refined)), dtype=int)
    amplitudes = np.abs(centered[peaks])

    changed = True
    while changed and len(peaks) > 2:
        changed = False
        rr = np.diff(peaks)
        if len(rr) == 0:
            break
        median_rr = int(np.median(rr))
        min_rr = max(int(0.28 * FS), int(0.55 * median_rr))

        keep = np.ones(len(peaks), dtype=bool)
        for i, interval in enumerate(rr):
            if interval < min_rr:
                left_amp = amplitudes[i]
                right_amp = amplitudes[i + 1]
                if left_amp >= right_amp:
                    keep[i + 1] = False
                else:
                    keep[i] = False
                changed = True

        peaks = peaks[keep]
        amplitudes = np.abs(centered[peaks])

    return peaks


def _score_rhythm_lead(lead: np.ndarray) -> tuple[float, np.ndarray]:
    """Score une derivation pour la detection des pics R."""
    peaks = detect_r_peaks(lead)
    if len(peaks) < 2:
        return 0.0, peaks

    rr = np.diff(peaks) / FS * 1000.0
    plausible_count = 8 <= len(peaks) <= 20
    median_rr = float(np.median(rr)) if len(rr) else 0.0
    rr_cv = float(np.std(rr) / max(np.mean(rr), 1.0)) if len(rr) else 1.0
    amp = float(np.median(np.abs(lead[peaks]))) if len(peaks) else 0.0

    score = 0.0
    if plausible_count:
        score += 1.0
    if 350 <= median_rr <= 1400:
        score += 1.0
    score += min(1.0, amp / 0.3)
    score += max(0.0, 1.0 - rr_cv)
    return score, peaks


def _detect_r_peaks_multilead(ecg: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Choisit automatiquement la derivation la plus fiable pour le rythme.
    Priorise les derivations de rythme peripheriques : II, I, III, aVF.
    """
    candidate_leads = [1, 0, 2, 5]
    lead_results = {}
    for lead_idx in candidate_leads:
        if lead_idx >= ecg.shape[1]:
            continue
        lead = ecg[:WINDOW_SAMPLES, lead_idx] if ecg.shape[0] >= WINDOW_SAMPLES else ecg[:, lead_idx]
        score, peaks = _score_rhythm_lead(lead)
        lead_results[lead_idx] = (score, peaks)

    default_score, default_peaks = lead_results.get(1, (0.0, np.array([], dtype=int)))
    best_lead = 1
    best_score = default_score
    best_peaks = default_peaks

    for lead_idx, (score, peaks) in lead_results.items():
        if score > best_score:
            best_score = score
            best_peaks = peaks
            best_lead = lead_idx

    # On ne quitte la derivation II que si le gain est net.
    if best_lead != 1 and (best_score - default_score) < 0.30:
        return default_peaks, 1

    return best_peaks, best_lead


# â”€â”€â”€ Ã‰TAPE 4 : EXTRACTION DE FEATURES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def extract_features(ecg: np.ndarray) -> dict:
    """
    Extrait les features diagnostiques d'un ECG filtrÃ©.
    Ces features sont utilisÃ©es Ã  la fois pour le modÃ¨le IA
    et pour l'approche mathÃ©matique par rÃ¨gles.
    """
    # Choisir automatiquement la meilleure derivation pour le rythme
    rhythm_window = ecg[:WINDOW_SAMPLES] if ecg.shape[0] >= WINDOW_SAMPLES else ecg
    r_peaks, rhythm_lead_idx = _detect_r_peaks_multilead(rhythm_window)
    lead_ii = rhythm_window[:, rhythm_lead_idx]

    # â”€â”€ Intervalles RR â”€â”€
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / FS * 1000  # en ms
        rr_mean = float(np.mean(rr_intervals))
        rr_std = float(np.std(rr_intervals))
        heart_rate = int(60000 / rr_mean) if rr_mean > 0 else 0
        rr_irregularity = float(rr_std / rr_mean) if rr_mean > 0 else 0
        rr_median = float(np.median(rr_intervals))
        long_rr = rr_intervals[rr_intervals >= rr_median]
        slow_rr_mean = float(np.mean(long_rr)) if len(long_rr) > 0 else rr_mean
        dominant_slow_hr = int(60000 / slow_rr_mean) if slow_rr_mean > 0 else heart_rate
        alternating_rr_ratio = float(np.mean(rr_intervals < (0.75 * slow_rr_mean))) if slow_rr_mean > 0 else 0.0
    else:
        rr_mean = rr_std = rr_irregularity = 0.0
        heart_rate = 0
        dominant_slow_hr = 0
        alternating_rr_ratio = 0.0

    # â”€â”€ Amplitude des ondes R â”€â”€
    r_amplitudes = lead_ii[r_peaks] if len(r_peaks) > 0 else np.array([0.0])
    r_mean_amp = float(np.mean(r_amplitudes))

    # â”€â”€ Analyse spectrale (Fourier) â”€â”€
    n = len(lead_ii)
    spectrum = np.abs(fft(lead_ii))[:n // 2]
    freqs = fftfreq(n, 1 / FS)[:n // 2]
    # Puissance dans la bande fibrillation (4-12 Hz)
    fib_mask = (freqs >= 4) & (freqs <= 12)
    total_power = float(np.sum(spectrum))
    fib_power = float(np.sum(spectrum[fib_mask])) / total_power if total_power > 0 else 0

    # â”€â”€ Features sur dÃ©rivation V1 pour QRS (index 6) â”€â”€
    lead_v1 = ecg[:WINDOW_SAMPLES, 6] if ecg.shape[0] >= WINDOW_SAMPLES else ecg[:, 6]

    # Largeur QRS approximative (dÃ©tection des croisements de zÃ©ro autour des pics R)
    qrs_width_ms = _estimate_qrs_width(lead_v1, r_peaks)

    # â”€â”€ Features dÃ©rivation V5 pour amplitude onde R (HVG) â”€â”€
    lead_v5 = ecg[:WINDOW_SAMPLES, 10] if ecg.shape[0] >= WINDOW_SAMPLES else ecg[:, 10]
    v5_r_amp = float(np.max(np.abs(lead_v5)))

    # â”€â”€ ST segment (approximation sur dÃ©rivations V1-V4) â”€â”€
    st_by_lead = {}
    t_polarity_by_lead = {}
    q_wave_ratio_by_lead = {}
    for lead_idx, lead_name in enumerate(LEAD_NAMES):
        if lead_idx >= ecg.shape[1]:
            continue
        lead = ecg[:WINDOW_SAMPLES, lead_idx] if ecg.shape[0] >= WINDOW_SAMPLES else ecg[:, lead_idx]
        st_by_lead[lead_name] = _estimate_st_deviation(lead, r_peaks)
        t_polarity_by_lead[lead_name] = _estimate_t_wave_polarity(lead, r_peaks)
        q_wave_ratio_by_lead[lead_name] = _estimate_q_wave_ratio(lead, r_peaks)
    st_deviations = list(st_by_lead.values())
    st_max_deviation = float(np.max(np.abs(st_deviations))) if st_deviations else 0.0
    st_mean_deviation = float(np.mean(st_deviations)) if st_deviations else 0.0

    qrs_wide_leads = 0
    qrs_borderline_wide_leads = 0
    for lead_idx in [0, 1, 6, 7, 10, 11]:
        if lead_idx >= ecg.shape[1]:
            continue
        lead = ecg[:WINDOW_SAMPLES, lead_idx] if ecg.shape[0] >= WINDOW_SAMPLES else ecg[:, lead_idx]
        lead_qrs_width = _estimate_qrs_width(lead, r_peaks)
        if lead_qrs_width >= 120:
            qrs_wide_leads += 1
        if lead_qrs_width >= 100:
            qrs_borderline_wide_leads += 1

    v1_qrs_polarity = _estimate_qrs_polarity(lead_v1, r_peaks)
    lead_v6 = ecg[:WINDOW_SAMPLES, 11] if ecg.shape[0] >= WINDOW_SAMPLES else ecg[:, 11]
    v6_qrs_polarity = _estimate_qrs_polarity(lead_v6, r_peaks)

    baseline_wander = float(np.mean([np.std(signal.medfilt(ecg[:WINDOW_SAMPLES, i] if ecg.shape[0] >= WINDOW_SAMPLES else ecg[:, i], kernel_size=151)) for i in range(min(ecg.shape[1], N_LEADS))]))
    signal_energy = float(np.mean(np.std(ecg[:WINDOW_SAMPLES], axis=0))) if ecg.shape[0] >= WINDOW_SAMPLES else float(np.mean(np.std(ecg, axis=0)))
    snr_like = signal_energy / max(baseline_wander, 1e-3)
    quality_score = 1.0
    if len(r_peaks) < 8 or len(r_peaks) > 20:
        quality_score -= 0.35
    if heart_rate < 35 or heart_rate > 180:
        quality_score -= 0.35
    if baseline_wander > 0.12:
        quality_score -= 0.20
    if snr_like < 2.0:
        quality_score -= 0.20
    quality_score = float(max(0.0, min(1.0, quality_score)))

    return {
        # Rythme
        "heart_rate_bpm": heart_rate,
        "rr_mean_ms": round(rr_mean, 1),
        "rr_std_ms": round(rr_std, 1),
        "rr_irregularity_ratio": round(rr_irregularity, 3),
        "dominant_slow_heart_rate_bpm": dominant_slow_hr,
        "alternating_rr_ratio": round(alternating_rr_ratio, 3),
        "n_r_peaks": len(r_peaks),
        "rhythm_lead_index": rhythm_lead_idx,
        "rhythm_lead_name": LEAD_NAMES[rhythm_lead_idx] if rhythm_lead_idx < len(LEAD_NAMES) else f"lead_{rhythm_lead_idx}",
        # Morphologie
        "qrs_width_ms": round(qrs_width_ms, 1),
        "r_amplitude_mean": round(r_mean_amp, 3),
        "v5_r_amplitude": round(v5_r_amp, 3),
        # Segment ST
        "st_max_deviation_mm": round(st_max_deviation, 2),
        "st_mean_deviation_mm": round(st_mean_deviation, 2),
        "st_deviation_by_lead": {k: round(v, 3) for k, v in st_by_lead.items()},
        "t_polarity_by_lead": {k: round(v, 3) for k, v in t_polarity_by_lead.items()},
        "q_wave_ratio_by_lead": {k: round(v, 3) for k, v in q_wave_ratio_by_lead.items()},
        "qrs_wide_leads_count": qrs_wide_leads,
        "qrs_borderline_wide_leads_count": qrs_borderline_wide_leads,
        "v1_qrs_polarity": round(v1_qrs_polarity, 3),
        "v6_qrs_polarity": round(v6_qrs_polarity, 3),
        # Spectral
        "fibrillation_power_ratio": round(fib_power, 4),
        # Qualite
        "signal_quality_score": round(quality_score, 3),
        "baseline_wander_score": round(baseline_wander, 3),
        "snr_like_ratio": round(float(snr_like), 3),
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


def _estimate_st_deviation(
    lead: np.ndarray,
    r_peaks: np.ndarray,
    offset_ms: int = 80,
    baseline_start_ms: int = 90,
    baseline_end_ms: int = 20,
) -> float:
    """Estime la deviation du segment ST par rapport a une baseline locale pre-QRS."""
    if len(r_peaks) == 0:
        return 0.0
    offset = int(offset_ms / 1000 * FS)
    baseline_start = int(baseline_start_ms / 1000 * FS)
    baseline_end = int(baseline_end_ms / 1000 * FS)
    deviations = []
    for rp in r_peaks[:10]:
        st_idx = rp + offset
        base_start_idx = max(0, rp - baseline_start)
        base_end_idx = max(base_start_idx + 1, rp - baseline_end)
        if st_idx >= len(lead) or base_end_idx > len(lead):
            continue
        baseline_segment = lead[base_start_idx:base_end_idx]
        if len(baseline_segment) < 3:
            continue
        local_baseline = float(np.median(baseline_segment))
        deviations.append(float(lead[st_idx] - local_baseline))
    return float(np.mean(deviations)) if deviations else 0.0


# â”€â”€â”€ Ã‰TAPE 5 : APPROCHE MATHÃ‰MATIQUE (rÃ¨gles diagnostiques) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _estimate_t_wave_polarity(
    lead: np.ndarray,
    r_peaks: np.ndarray,
    offset_ms: int = 200,
    baseline_start_ms: int = 90,
    baseline_end_ms: int = 20,
) -> float:
    """Estime la polarite moyenne de l'onde T par rapport a une baseline locale."""
    if len(r_peaks) == 0:
        return 0.0
    offset = int(offset_ms / 1000 * FS)
    baseline_start = int(baseline_start_ms / 1000 * FS)
    baseline_end = int(baseline_end_ms / 1000 * FS)
    values = []
    for rp in r_peaks[:10]:
        t_idx = rp + offset
        base_start_idx = max(0, rp - baseline_start)
        base_end_idx = max(base_start_idx + 1, rp - baseline_end)
        if t_idx >= len(lead) or base_end_idx > len(lead):
            continue
        baseline_segment = lead[base_start_idx:base_end_idx]
        if len(baseline_segment) < 3:
            continue
        local_baseline = float(np.median(baseline_segment))
        values.append(float(lead[t_idx] - local_baseline))
    return float(np.mean(values)) if values else 0.0


def _estimate_q_wave_ratio(
    lead: np.ndarray,
    r_peaks: np.ndarray,
    q_window_ms: int = 40,
    baseline_start_ms: int = 90,
    baseline_end_ms: int = 20,
) -> float:
    """
    Estime un ratio de Q pathologique avant le pic R.
    Cherche une deflexion negative avant R rapportee a l'amplitude du complexe.
    """
    if len(r_peaks) == 0:
        return 0.0

    q_window = int(q_window_ms / 1000 * FS)
    baseline_start = int(baseline_start_ms / 1000 * FS)
    baseline_end = int(baseline_end_ms / 1000 * FS)
    ratios = []

    for rp in r_peaks[:10]:
        q_start = max(0, rp - q_window)
        q_end = max(q_start + 1, rp)
        base_start = max(0, rp - baseline_start)
        base_end = max(base_start + 1, rp - baseline_end)
        q_segment = lead[q_start:q_end]
        baseline_segment = lead[base_start:base_end]
        if len(q_segment) == 0 or len(baseline_segment) == 0:
            continue

        baseline = float(np.median(baseline_segment))
        q_depth = float(baseline - np.min(q_segment))
        r_height = float(np.max(np.abs(lead[max(0, rp - 5): min(len(lead), rp + 5)]) - baseline))
        if r_height <= 1e-3:
            continue
        ratio = q_depth / r_height
        ratios.append(max(0.0, ratio))

    return float(np.mean(ratios)) if ratios else 0.0


def _estimate_qrs_polarity(lead: np.ndarray, r_peaks: np.ndarray, window_ms: int = 50) -> float:
    """Mesure la polarite dominante du QRS autour des pics R."""
    if len(r_peaks) == 0:
        return 0.0
    window = int(window_ms / 1000 * FS)
    values = []
    for rp in r_peaks[:10]:
        start = max(0, rp - window)
        end = min(len(lead), rp + window)
        segment = lead[start:end]
        if len(segment) < 5:
            continue
        idx = int(np.argmax(np.abs(segment)))
        values.append(float(segment[idx]))
    return float(np.mean(values)) if values else 0.0


def _count_contiguous_abnormal_leads(values_by_lead: dict, leads: list[str], threshold: float, direction: str = "positive") -> int:
    """Compte les anomalies contigues sur un groupe de derivations."""
    count = 0
    best = 0
    for lead in leads:
        value = values_by_lead.get(lead, 0.0)
        is_abnormal = value >= threshold if direction == "positive" else value <= -threshold
        if is_abnormal:
            count += 1
            best = max(best, count)
        else:
            count = 0
    return best


def diagnose_by_rules(features: dict) -> dict:
    """
    Diagnostic base sur des regles cliniques explicables.
    Priorite aux troubles du rythme globaux, mais analyse localisee
    sur derivations contigues pour ischemie et infarctus.
    """
    hr = features["heart_rate_bpm"]
    irr = features["rr_irregularity_ratio"]
    qrs = features["qrs_width_ms"]
    fib_power = features["fibrillation_power_ratio"]
    slow_hr = features.get("dominant_slow_heart_rate_bpm", hr)
    alternating_rr_ratio = features.get("alternating_rr_ratio", 0.0)
    st_max = features["st_max_deviation_mm"]
    st_mean = features["st_mean_deviation_mm"]
    v5_r = features["v5_r_amplitude"]
    n_r_peaks = features["n_r_peaks"]
    st_by_lead = features.get("st_deviation_by_lead", {})
    t_by_lead = features.get("t_polarity_by_lead", {})
    q_by_lead = features.get("q_wave_ratio_by_lead", {})
    qrs_wide_leads = features.get("qrs_wide_leads_count", 0)
    qrs_borderline_wide_leads = features.get("qrs_borderline_wide_leads_count", 0)
    v1_qrs_polarity = features.get("v1_qrs_polarity", 0.0)
    v6_qrs_polarity = features.get("v6_qrs_polarity", 0.0)

    scores = {p: 0.0 for p in PATHOLOGIES}

    contiguous_st_elevation = (
        max(
            _count_contiguous_abnormal_leads(st_by_lead, leads, threshold=0.18, direction="positive")
            for leads in LEAD_GROUPS.values()
        )
        if st_by_lead
        else 0
    )
    contiguous_st_depression = (
        max(
            _count_contiguous_abnormal_leads(st_by_lead, leads, threshold=0.10, direction="negative")
            for leads in LEAD_GROUPS.values()
        )
        if st_by_lead
        else 0
    )
    contiguous_t_inversion = (
        max(
            _count_contiguous_abnormal_leads(t_by_lead, leads, threshold=0.05, direction="negative")
            for leads in LEAD_GROUPS.values()
        )
        if t_by_lead
        else 0
    )
    contiguous_t_inversion_mild = (
        max(
            _count_contiguous_abnormal_leads(t_by_lead, leads, threshold=0.03, direction="negative")
            for leads in LEAD_GROUPS.values()
        )
        if t_by_lead
        else 0
    )
    contiguous_q_pathologic = (
        max(
            _count_contiguous_abnormal_leads(q_by_lead, leads, threshold=0.22, direction="positive")
            for leads in LEAD_GROUPS.values()
        )
        if q_by_lead
        else 0
    )
    contiguous_st_depression_mild = (
        max(
            _count_contiguous_abnormal_leads(st_by_lead, leads, threshold=0.02, direction="negative")
            for leads in LEAD_GROUPS.values()
        )
        if st_by_lead
        else 0
    )
    localized_ischemic_pattern = contiguous_st_elevation >= 2 or contiguous_st_depression >= 2

    if irr > 0.18 and fib_power > 0.25:
        scores["fibrillation_auriculaire"] += 0.50
    if irr > 0.24 and fib_power > 0.30:
        scores["fibrillation_auriculaire"] += 0.25
    if irr > 0.30 and fib_power > 0.35:
        scores["fibrillation_auriculaire"] += 0.10
    if n_r_peaks < 12 and irr > 0.18 and fib_power > 0.30:
        scores["fibrillation_auriculaire"] += 0.10
    if irr > 0.13 and fib_power > 0.35 and max(abs(v1_qrs_polarity), abs(v6_qrs_polarity)) < 0.35:
        scores["fibrillation_auriculaire"] += 0.45
    if irr >= 0.14 and fib_power > 0.35 and 95 <= qrs <= 103 and qrs_borderline_wide_leads >= 3 and max(abs(v1_qrs_polarity), abs(v6_qrs_polarity)) < 0.35 and contiguous_st_depression < 2:
        scores["fibrillation_auriculaire"] += 0.35

    if hr > 100 and qrs > 120:
        scores["tachycardie_ventriculaire"] += 0.70
    elif hr > 130:
        scores["tachycardie_ventriculaire"] += 0.20

    if hr < 50 and irr < 0.12:
        scores["bradycardie"] = 0.95
    elif 0 < hr < 60 and irr < 0.12:
        scores["bradycardie"] += 0.85
    if hr < 50 and slow_hr < 40 and alternating_rr_ratio >= 0.45 and qrs < 116:
        scores["bradycardie"] += 0.95

    if qrs >= 120 and qrs_wide_leads >= 3 and 40 <= hr <= 110 and irr < 0.16:
        scores["bloc_auriculo_ventriculaire"] += 0.55
    if qrs >= 140 and qrs_wide_leads >= 4 and irr < 0.16:
        scores["bloc_auriculo_ventriculaire"] += 0.20
    if qrs >= 130 and not localized_ischemic_pattern:
        scores["bloc_auriculo_ventriculaire"] += 0.15
    if qrs >= 100 and qrs_borderline_wide_leads >= 3 and 40 <= hr <= 115:
        scores["bloc_auriculo_ventriculaire"] += 0.35
    if qrs >= 108 and qrs_borderline_wide_leads >= 4:
        scores["bloc_auriculo_ventriculaire"] += 0.20
    if qrs >= 100 and qrs_borderline_wide_leads >= 3 and contiguous_st_elevation == 0 and contiguous_st_depression < 2:
        scores["bloc_auriculo_ventriculaire"] += 0.10
    if qrs_borderline_wide_leads >= 4 and 40 <= hr <= 115 and fib_power < 0.45 and (irr < 0.14 or max(abs(v1_qrs_polarity), abs(v6_qrs_polarity)) > 0.35):
        scores["bloc_auriculo_ventriculaire"] += 0.35
    if qrs >= 94 and contiguous_q_pathologic >= 3 and contiguous_t_inversion < 2 and irr < 0.15 and (max(abs(v1_qrs_polarity), abs(v6_qrs_polarity)) > 0.45 or v1_qrs_polarity * v6_qrs_polarity < 0):
        scores["bloc_auriculo_ventriculaire"] += 0.30
    if qrs >= 94 and contiguous_st_elevation >= 3 and contiguous_t_inversion < 2 and irr < 0.15:
        scores["bloc_auriculo_ventriculaire"] += 0.45
    if qrs >= 92 and v1_qrs_polarity < -0.15 and v6_qrs_polarity < -0.10 and irr < 0.15:
        scores["bloc_auriculo_ventriculaire"] += 0.50
    if qrs >= 92 and v1_qrs_polarity < -0.25 and v6_qrs_polarity < -0.25 and irr < 0.12:
        scores["bloc_auriculo_ventriculaire"] += 0.35
    if qrs_borderline_wide_leads >= 4 and abs(v1_qrs_polarity) > 0.20 and abs(v6_qrs_polarity) > 0.20:
        scores["bloc_auriculo_ventriculaire"] += 0.20

    if v5_r > 2.5 and irr < 0.20:
        scores["hypertrophie_ventriculaire_gauche"] += 0.6
    if v5_r > 3.5 and irr < 0.20:
        scores["hypertrophie_ventriculaire_gauche"] += 0.3

    if contiguous_st_depression >= 2 and irr < 0.20 and st_mean <= -0.08:
        scores["ischemie_myocardique"] += 0.55
    if contiguous_st_depression >= 2 and contiguous_t_inversion >= 2 and st_mean <= -0.05:
        scores["ischemie_myocardique"] += 0.20
    if contiguous_st_depression >= 3 and st_max >= 0.12:
        scores["ischemie_myocardique"] += 0.10
    if contiguous_st_depression_mild >= 2 and contiguous_t_inversion >= 2:
        scores["ischemie_myocardique"] += 0.45
    if contiguous_st_depression >= 3 and contiguous_t_inversion >= 3:
        scores["ischemie_myocardique"] += 0.20
    if contiguous_st_depression_mild >= 3 and contiguous_t_inversion_mild >= 3 and qrs_borderline_wide_leads <= 3:
        scores["ischemie_myocardique"] += 0.35

    if contiguous_st_elevation >= 2 and irr < 0.16:
        scores["infarctus_du_myocarde"] += 0.65
    if contiguous_st_elevation >= 3 and st_max >= 0.25 and irr < 0.16:
        scores["infarctus_du_myocarde"] += 0.20
    if contiguous_st_elevation >= 2 and contiguous_t_inversion >= 2:
        scores["infarctus_du_myocarde"] += 0.10
    if hr >= 90 and contiguous_st_elevation >= 2 and st_mean >= 0.12:
        scores["infarctus_du_myocarde"] += 0.05
    if contiguous_q_pathologic >= 2 and contiguous_t_inversion >= 2:
        scores["infarctus_du_myocarde"] += 0.45
    if contiguous_q_pathologic >= 2 and contiguous_st_depression >= 2:
        scores["infarctus_du_myocarde"] += 0.20
    if contiguous_q_pathologic >= 2 and irr < 0.15 and fib_power < 0.35 and qrs < 110:
        if not (v1_qrs_polarity < -0.25 and v6_qrs_polarity < -0.25):
            scores["infarctus_du_myocarde"] += 0.35
    if contiguous_q_pathologic >= 3 and irr < 0.15 and fib_power < 0.35:
        scores["infarctus_du_myocarde"] += 0.10

    if scores["fibrillation_auriculaire"] >= 0.50:
        scores["normal"] = 0.0
        if hr >= 55:
            scores["bradycardie"] = 0.0
        if scores["ischemie_myocardique"] >= 0.45:
            scores["fibrillation_auriculaire"] = max(0.0, scores["fibrillation_auriculaire"] - 0.35)
        if scores["bloc_auriculo_ventriculaire"] >= 0.35 and (qrs >= 100 or qrs_borderline_wide_leads >= 4):
            scores["fibrillation_auriculaire"] = max(0.0, scores["fibrillation_auriculaire"] - 0.45)
        if max(abs(v1_qrs_polarity), abs(v6_qrs_polarity)) < 0.35 and fib_power > 0.35:
            scores["bloc_auriculo_ventriculaire"] = max(0.0, scores["bloc_auriculo_ventriculaire"] - 0.35)

    if scores["bradycardie"] >= 0.80:
        scores["normal"] = 0.0
        scores["fibrillation_auriculaire"] = min(scores["fibrillation_auriculaire"], 0.20)
        if contiguous_t_inversion < 2:
            scores["ischemie_myocardique"] = min(scores["ischemie_myocardique"], 0.15)
        if qrs < 116:
            scores["bloc_auriculo_ventriculaire"] = max(0.0, scores["bloc_auriculo_ventriculaire"] - 0.25)

    if scores["bloc_auriculo_ventriculaire"] >= 0.55:
        scores["normal"] = 0.0
        if not localized_ischemic_pattern:
            scores["infarctus_du_myocarde"] = max(0.0, scores["infarctus_du_myocarde"] - 0.20)
        scores["fibrillation_auriculaire"] = min(scores["fibrillation_auriculaire"], 0.25)
    elif scores["bloc_auriculo_ventriculaire"] >= 0.35 and contiguous_q_pathologic >= 3 and contiguous_t_inversion < 2:
        scores["infarctus_du_myocarde"] = max(0.0, scores["infarctus_du_myocarde"] - 0.25)
    elif scores["bloc_auriculo_ventriculaire"] >= 0.45 and contiguous_st_elevation >= 3 and contiguous_t_inversion < 2:
        scores["infarctus_du_myocarde"] = max(0.0, scores["infarctus_du_myocarde"] - 0.40)
    if scores["bloc_auriculo_ventriculaire"] >= 0.50 and v1_qrs_polarity < -0.15 and v6_qrs_polarity < -0.10:
        scores["infarctus_du_myocarde"] = max(0.0, scores["infarctus_du_myocarde"] - 0.30)
    if scores["bloc_auriculo_ventriculaire"] >= 0.70 and v1_qrs_polarity < -0.25 and v6_qrs_polarity < -0.25 and irr < 0.12:
        scores["infarctus_du_myocarde"] = max(0.0, scores["infarctus_du_myocarde"] - 0.45)
        scores["ischemie_myocardique"] = max(0.0, scores["ischemie_myocardique"] - 0.10)
    if scores["bloc_auriculo_ventriculaire"] >= 0.70 and abs(scores["bloc_auriculo_ventriculaire"] - scores["infarctus_du_myocarde"]) <= 0.10:
        scores["bloc_auriculo_ventriculaire"] += 0.08

    if scores["ischemie_myocardique"] >= 0.50:
        scores["normal"] = 0.0
    if scores["ischemie_myocardique"] >= 0.40 and contiguous_st_depression >= 2 and contiguous_t_inversion >= 2:
        scores["bloc_auriculo_ventriculaire"] = max(0.0, scores["bloc_auriculo_ventriculaire"] - 0.25)
    if scores["ischemie_myocardique"] >= 0.35 and contiguous_st_depression_mild >= 3 and contiguous_t_inversion_mild >= 3:
        scores["bloc_auriculo_ventriculaire"] = max(0.0, scores["bloc_auriculo_ventriculaire"] - 0.25)

    if scores["infarctus_du_myocarde"] >= 0.60:
        scores["normal"] = 0.0
        scores["ischemie_myocardique"] = min(scores["ischemie_myocardique"], 0.30)
    elif scores["infarctus_du_myocarde"] >= 0.40 and contiguous_q_pathologic >= 2:
        scores["ischemie_myocardique"] = max(0.0, scores["ischemie_myocardique"] - 0.10)

    if irr < 0.12 and fib_power < 0.30 and 60 <= hr <= 100 and qrs < 120 and not localized_ischemic_pattern:
        scores["normal"] += 0.45

    max_score = max(scores.values())
    if max_score < 0.30:
        scores["normal"] = 0.80

    base_floor = 0.005
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
        "methode": "r?gles_mathematiques",
        "pathologie": best,
        "probabilites": probs,
        "confiance": probs[best]
    }


def build_cnn_model(num_classes: int | None = None):
    """
    Architecture CNN 1D pour la classification ECG.
    Ã€ entraÃ®ner sur les 30 000 ECG rÃ©els.
    NÃ©cessite : pip install tensorflow
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models

        n_classes = num_classes or len(PATHOLOGIES)

        model = models.Sequential([
            # EntrÃ©e : (WINDOW_SAMPLES, N_LEADS)
            layers.Input(shape=(WINDOW_SAMPLES, N_LEADS)),

            # Bloc 1 : features Ã  grande Ã©chelle (rythme global)
            layers.Conv1D(32, kernel_size=50, strides=2, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=5),

            # Bloc 2 : features Ã  Ã©chelle intermÃ©diaire (complexes QRS)
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
            layers.Dense(n_classes, activation="softmax"),
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    except ImportError:
        return None


def load_model_weights(model_path: str | Path = CNN_MODEL_PATH):
    """Charge le modÃ¨le CNN 1D entraÃ®nÃ© si disponible."""
    try:
        import tensorflow as tf
        resolved_path = Path(model_path)
        if resolved_path.exists():
            return tf.keras.models.load_model(resolved_path)
    except Exception:
        pass
    return None


# â”€â”€â”€ PIPELINE COMPLET â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def preprocess_ecg(file_bytes: bytes) -> dict:
    """
    Pipeline complet de prÃ©traitement.
    Retourne les features extraites.
    """
    raw = read_ecg_csv(file_bytes)
    filtered = remove_baseline_wander(raw)
    filtered = bandpass_filter(filtered)
    filtered = notch_filter(filtered)
    filtered = smooth_ecg(filtered)
    features = extract_features(filtered)
    return {"features": features, "signal": filtered}


def _fuse_probability_sources(
    rule_probs: dict[str, float],
    learned_probs: dict[str, float],
    features: dict[str, float],
) -> dict[str, float]:
    fused: dict[str, float] = {}
    critical_pathologies = {"infarctus_du_myocarde", "ischemie_myocardique", "bloc_auriculo_ventriculaire"}

    rule_sorted = sorted(rule_probs.items(), key=lambda item: item[1], reverse=True)
    learned_sorted = sorted(learned_probs.items(), key=lambda item: item[1], reverse=True)

    rule_best, rule_best_prob = rule_sorted[0]
    learned_best, learned_best_prob = learned_sorted[0]
    rule_second_prob = rule_sorted[1][1] if len(rule_sorted) > 1 else 0.0
    learned_second_prob = learned_sorted[1][1] if len(learned_sorted) > 1 else 0.0

    rule_margin = max(0.0, rule_best_prob - rule_second_prob)
    learned_margin = max(0.0, learned_best_prob - learned_second_prob)
    quality_score = float(features.get("signal_quality_score", 1.0))

    rule_weight = 0.35
    learned_weight = 0.65

    if rule_best == learned_best and rule_margin >= 0.25:
        rule_weight = 0.50
        learned_weight = 0.50
    elif rule_best in critical_pathologies and rule_best_prob >= 0.50 and rule_margin >= 0.12:
        rule_weight = 0.70
        learned_weight = 0.30
    elif rule_best != learned_best and learned_margin >= 0.12:
        rule_weight = 0.20
        learned_weight = 0.80
    elif rule_margin < 0.15:
        rule_weight = 0.15
        learned_weight = 0.85

    if rule_best == "normal" and learned_best != "normal" and learned_best_prob >= 0.45:
        rule_weight = min(rule_weight, 0.15)
        learned_weight = max(learned_weight, 0.85)

    if quality_score < 0.70:
        rule_weight = max(0.10, rule_weight - 0.10)
        learned_weight = min(0.90, learned_weight + 0.10)

    for pathology in PATHOLOGIES:
        fused[pathology] = (
            rule_weight * float(rule_probs.get(pathology, 0.0))
            + learned_weight * float(learned_probs.get(pathology, 0.0))
        )

        if pathology == rule_best and pathology == learned_best:
            fused[pathology] += 0.05
        if pathology in critical_pathologies and rule_probs.get(pathology, 0.0) >= 0.70:
            fused[pathology] += 0.03
        if pathology == learned_best and learned_best_prob >= 0.55 and learned_margin >= 0.10:
            fused[pathology] += 0.04

    total = sum(fused.values()) or 1.0
    return {pathology: value / total for pathology, value in fused.items()}


def predict_pathology(preprocessed: dict) -> dict:
    """
    PrÃ©dit la pathologie Ã  partir du signal prÃ©traitÃ©.
    Utilise le CNN si disponible, sinon les rÃ¨gles mathÃ©matiques.
    """
    import os
    features = preprocessed["features"]
    signal_data = preprocessed["signal"]

    # Règles explicables toujours calculées, puis fusion éventuelle avec apprentissage.
    rule_result = diagnose_by_rules(features)
    rule_probs = rule_result["probabilites"]
    probs = rule_probs
    methode = "rÃ¨gles mathÃ©matiques"

    try:
        try:
            from ecg_hybrid_model import predict_hybrid_probabilities
        except ModuleNotFoundError:
            from backend.ecg_hybrid_model import predict_hybrid_probabilities
        learned_probs = predict_hybrid_probabilities(features)
    except Exception:
        learned_probs = None

    cnn_probs = None
    try:
        try:
            from ecg_cnn_model import predict_cnn_probabilities
        except ModuleNotFoundError:
            from backend.ecg_cnn_model import predict_cnn_probabilities

        cnn_classes = PATHOLOGIES
        if CNN_META_PATH.exists():
            try:
                cnn_classes = json.loads(CNN_META_PATH.read_text(encoding="utf-8")).get("classes", PATHOLOGIES)
            except Exception:
                cnn_classes = PATHOLOGIES
        cnn_probs = predict_cnn_probabilities(signal_data, classes=cnn_classes)
    except Exception:
        cnn_probs = None

    if cnn_probs:
        probs = _fuse_probability_sources(rule_probs, cnn_probs, features)
        methode = "CNN 1D + règles"
    elif learned_probs:
        probs = _fuse_probability_sources(rule_probs, learned_probs, features)
        methode = "Hybride rÃ¨gles + apprentissage"
    else:
        model = load_model_weights()
        if model is not None:
            window = signal_data[:WINDOW_SAMPLES]
            if window.shape[0] < WINDOW_SAMPLES:
                pad = np.zeros((WINDOW_SAMPLES - window.shape[0], window.shape[1]))
                window = np.vstack([window, pad])
            probs_array = model.predict(window[np.newaxis])[0]
            probs = {PATHOLOGIES[i]: float(p) for i, p in enumerate(probs_array)}
            methode = "CNN 1D"

    quality_score = float(features.get("signal_quality_score", 1.0))
    sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    best_pathologie, best_prob = sorted_probs[0]
    second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    margin = max(0.0, best_prob - second_prob)

    confidence_cap = 0.88
    if best_pathologie == "normal":
        confidence_cap = 0.82
    if quality_score < 0.75:
        confidence_cap = min(confidence_cap, 0.70)
    if quality_score < 0.50:
        confidence_cap = min(confidence_cap, 0.55)
    if margin < 0.15:
        confidence_cap = min(confidence_cap, 0.68)

    confiance = min(best_prob, confidence_cap)

    if quality_score < 0.40:
        confiance = min(confiance, 0.45)
        if best_pathologie in {"infarctus_du_myocarde", "ischemie_myocardique", "fibrillation_auriculaire"} and second_prob > 0:
            best_pathologie = sorted_probs[1][0]
            confiance = min(sorted_probs[1][1], 0.45)

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
        "infarctus_du_myocarde":         "URGENCE ABSOLUE â€” Appeler SAMU, aspirinÃ© 250mg, dÃ©fibrillateur en standby",
        "tachycardie_ventriculaire":      "URGENCE â€” Monitoring continu, prÃ©parer choc Ã©lectrique externe",
        "fibrillation_auriculaire":       "Consulter mÃ©decin tÃ©lÃ©consultant â€” anticoagulation Ã  Ã©valuer",
        "ischemie_myocardique":           "Consultation cardiologique urgente â€” trinitrine sublinguale si douleur",
        "bloc_auriculo_ventriculaire":    "ECG 12 dÃ©rivations complet requis â€” tÃ©lÃ©consultation cardiologique",
        "hypertrophie_ventriculaire_gauche": "Consultation cardiologique programmÃ©e â€” bilan HTA",
        "bradycardie":                    "Surveiller FC, consulter si symptÃ´mes (syncope, dyspnÃ©e)",
        "normal":                         "ECG dans les limites normales â€” poursuivre bilan clinique",
    }
    base = recommandations.get(pathologie, "TÃ©lÃ©consultation mÃ©dicale recommandÃ©e")
    if confiance < 0.5:
        base += " (confiance faible â€” interprÃ©tation Ã  valider par mÃ©decin)"
    return base


# â”€â”€â”€ DONNÃ‰ES SYNTHÃ‰TIQUES (pour tests sans CSV rÃ©el) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Redefinition volontaire pour garder une recommandation adaptee
# au contexte d'une infirmiere deja aupres du patient.
def _get_recommandation(pathologie: str, confiance: float) -> str:
    recommandations = {
        "infarctus_du_myocarde": "URGENCE ABSOLUE - Surveillance infirmiere rapprochee, alerte medicale immediate, preparation du materiel d'urgence et application du protocole local valide medicalement",
        "tachycardie_ventriculaire": "URGENCE - Monitoring continu, alerte medicale immediate, preparation du defibrillateur et du materiel de reanimation selon protocole local",
        "fibrillation_auriculaire": "Teleconsultation medicale rapide et surveillance clinique, avec adaptation therapeutique a valider par le medecin",
        "ischemie_myocardique": "Avis medical urgent, surveillance rapprochee et mise en condition du patient selon protocole local",
        "bloc_auriculo_ventriculaire": "Suspicion de bloc de branche - ECG complet, avis cardiologique rapide et surveillance de la tolerance clinique",
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
    GÃ©nÃ¨re un ECG synthÃ©tique pour les tests du pipeline.
    Reproduit les caractÃ©ristiques diagnostiques clÃ©s de chaque pathologie.
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
        # Rythme irrÃ©gulier + trÃ©mulations basales
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
    """GÃ©nÃ¨re un signal PQRST synthÃ©tique."""
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
    """GÃ©nÃ¨re un signal de fibrillation auriculaire (rythme irrÃ©gulier + trÃ©mulations)."""
    wave = np.zeros(len(t))
    # TrÃ©mulations atriales (4-12 Hz)
    wave += 0.05 * np.sin(2 * np.pi * 6 * t + np.random.uniform(0, np.pi))
    wave += 0.03 * np.sin(2 * np.pi * 9 * t)
    # Complexes QRS irrÃ©guliers
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
        print(f"FC dÃ©tectÃ©e      : {features['heart_rate_bpm']} bpm")
        print(f"IrrÃ©gularitÃ© RR  : {features['rr_irregularity_ratio']}")
        print(f"Largeur QRS      : {features['qrs_width_ms']} ms")
        print(f"DÃ©viation ST moy : {features['st_mean_deviation_mm']} mm")
        print(f"Diagnostic rÃ¨gles: {result['pathologie']} (confiance {result['confiance']:.2f})")
        print()
