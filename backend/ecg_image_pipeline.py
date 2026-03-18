"""
Extraction locale d'un pseudo-signal ECG 12 derivations a partir d'une image.

Objectif :
- convertir une image ECG en matrice (5000, 12)
- reutiliser ensuite le pipeline ECG existant

Hypothese principale :
- l'image contient 12 traces empilees verticalement
- le trace ECG est majoritairement dessine en bleu sur fond clair
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import signal

FS = 500
WINDOW_SEC = 10
WINDOW_SAMPLES = FS * WINDOW_SEC
N_LEADS = 12


@dataclass
class ECGImageExtractionResult:
    signal: np.ndarray
    debug: dict


def extract_ecg_signal_from_image(image_bytes: bytes) -> ECGImageExtractionResult:
    image = _decode_image(image_bytes)
    blue_mask = _extract_trace_mask(image)

    if int(np.count_nonzero(blue_mask)) < 500:
        raise ValueError("Trace ECG insuffisamment visible sur l'image")

    cropped_mask, crop_debug = _crop_to_signal_region(blue_mask)
    lead_bounds = _detect_lead_bounds(cropped_mask, expected=N_LEADS)
    signal_12 = _extract_leads_from_mask(cropped_mask, lead_bounds)
    signal_12 = _postprocess_signal(signal_12)

    debug = {
        "source_type": "image_ecg_signal",
        "image_height_px": int(image.shape[0]),
        "image_width_px": int(image.shape[1]),
        "cropped_height_px": int(cropped_mask.shape[0]),
        "cropped_width_px": int(cropped_mask.shape[1]),
        "lead_count_detected": int(signal_12.shape[1]),
        "trace_pixels": int(np.count_nonzero(blue_mask)),
        **crop_debug,
    }
    return ECGImageExtractionResult(signal=signal_12, debug=debug)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image ECG non lisible")
    return image


def _extract_trace_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Bleu classique des tracés matplotlib / ECG exportés.
    hsv_mask = cv2.inRange(hsv, (85, 35, 40), (140, 255, 255))

    b, g, r = cv2.split(image)
    blue_dominant = (
        (b.astype(np.int16) > g.astype(np.int16) + 10)
        & (b.astype(np.int16) > r.astype(np.int16) + 10)
        & (b > 50)
    )

    mask = np.where((hsv_mask > 0) | blue_dominant, 255, 0).astype(np.uint8)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _crop_to_signal_region(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    row_density = np.count_nonzero(mask, axis=1)
    col_density = np.count_nonzero(mask, axis=0)

    row_idx = np.where(row_density > max(8, 0.01 * mask.shape[1]))[0]
    col_idx = np.where(col_density > max(4, 0.004 * mask.shape[0]))[0]

    if row_idx.size == 0 or col_idx.size == 0:
        return mask, {"crop_top": 0, "crop_bottom": int(mask.shape[0]), "crop_left": 0, "crop_right": int(mask.shape[1])}

    top = max(0, int(row_idx[0] - 5))
    bottom = min(mask.shape[0], int(row_idx[-1] + 6))
    left = max(0, int(col_idx[0] - 5))
    right = min(mask.shape[1], int(col_idx[-1] + 6))

    cropped = mask[top:bottom, left:right]

    # Les images de type matplotlib ont souvent une grande marge vide.
    # On garde une petite marge mais on retire les bords inutiles.
    h, w = cropped.shape[:2]
    left_trim = int(0.02 * w)
    right_trim = int(0.01 * w)
    top_trim = int(0.005 * h)
    bottom_trim = int(0.005 * h)
    cropped = cropped[top_trim : max(top_trim + 1, h - bottom_trim), left_trim : max(left_trim + 1, w - right_trim)]
    return cropped, {
        "crop_top": top,
        "crop_bottom": bottom,
        "crop_left": left,
        "crop_right": right,
    }


def _detect_lead_bounds(mask: np.ndarray, expected: int = 12) -> list[tuple[int, int]]:
    row_density = np.count_nonzero(mask, axis=1).astype(float)
    smoothed = signal.savgol_filter(row_density, 31 if mask.shape[0] >= 31 else max(5, (mask.shape[0] // 2) * 2 + 1), 2, mode="interp")
    min_distance = max(20, mask.shape[0] // (expected + 3))
    prominence = max(3.0, np.max(smoothed) * 0.08)
    peaks, _ = signal.find_peaks(smoothed, distance=min_distance, prominence=prominence)

    if peaks.size >= expected:
        order = np.argsort(smoothed[peaks])[::-1][:expected]
        centers = np.sort(peaks[order])
    else:
        centers = np.linspace(0, mask.shape[0] - 1, expected + 2, dtype=int)[1:-1]

    # Si les centres detectes sont trop irreguliers, on repasse a une
    # segmentation uniforme, plus robuste pour les images en 12 bandes fixes.
    if len(centers) == expected:
        diffs = np.diff(centers.astype(float))
        if np.mean(diffs) > 0 and (np.std(diffs) / np.mean(diffs)) > 0.25:
            centers = np.linspace(0, mask.shape[0] - 1, expected + 2, dtype=int)[1:-1]

    centers = np.asarray(centers, dtype=int)
    boundaries = [0]
    for idx in range(len(centers) - 1):
        boundaries.append(int((centers[idx] + centers[idx + 1]) / 2))
    boundaries.append(mask.shape[0] - 1)

    bounds: list[tuple[int, int]] = []
    for i in range(expected):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        if end <= start:
            end = min(mask.shape[0] - 1, start + max(8, mask.shape[0] // expected))
        bounds.append((start, end))
    return bounds


def _extract_leads_from_mask(mask: np.ndarray, lead_bounds: list[tuple[int, int]]) -> np.ndarray:
    extracted: list[np.ndarray] = []
    width = mask.shape[1]
    x_source = np.arange(width)

    for start, end in lead_bounds:
        band = mask[start:end, :]
        band_height = max(1, end - start)
        center = band_height / 2.0
        y_values = _extract_continuous_trace(band, center)

        amplitude = (center - y_values) / max(4.0, band_height / 2.2)
        amplitude = _clip_signal_outliers(amplitude, band_height)
        amplitude = signal.savgol_filter(amplitude, 21 if width >= 21 else max(5, (width // 2) * 2 + 1), 2, mode="interp")

        x_target = np.linspace(0, width - 1, WINDOW_SAMPLES)
        lead_resampled = np.interp(x_target, x_source, amplitude)
        extracted.append(lead_resampled.astype(np.float32))

    return np.stack(extracted, axis=1)


def _postprocess_signal(signal_12: np.ndarray) -> np.ndarray:
    centered = signal_12 - np.median(signal_12, axis=0, keepdims=True)
    centered = signal.savgol_filter(centered, 9, 2, axis=0, mode="interp")
    scale = np.std(centered, axis=0, keepdims=True)
    scale[scale < 1e-3] = 1.0
    normalized = centered / scale

    # Calibrage prudent pour retrouver une amplitude de type mV compatible avec le pipeline.
    normalized *= 0.25
    return normalized.astype(np.float32)


def _extract_continuous_trace(band: np.ndarray, center: float) -> np.ndarray:
    width = band.shape[1]
    x_source = np.arange(width)
    y_values = np.full(width, np.nan, dtype=float)
    prev_y = None

    for x in range(width):
        ys = np.where(band[:, x] > 0)[0]
        if ys.size == 0:
            continue

        if prev_y is None:
            candidate = float(np.median(ys))
        else:
            candidate = float(ys[np.argmin(np.abs(ys - prev_y))])
            if abs(candidate - prev_y) > max(6.0, 0.18 * band.shape[0]):
                candidate = float(np.median(ys))

        y_values[x] = candidate
        prev_y = candidate

    if np.all(np.isnan(y_values)):
        return np.full(width, center, dtype=float)

    valid = ~np.isnan(y_values)
    y_values = np.interp(x_source, x_source[valid], y_values[valid])
    y_values = signal.medfilt(y_values, kernel_size=7 if width >= 7 else 3)
    return y_values


def _clip_signal_outliers(amplitude: np.ndarray, band_height: int) -> np.ndarray:
    amp = amplitude.copy()
    if amp.size < 5:
        return amp
    diff = np.diff(amp, prepend=amp[0])
    jump_threshold = max(0.18, 1.4 / max(1.0, band_height / 20.0))
    for idx in range(1, len(amp)):
        if abs(diff[idx]) > jump_threshold:
            amp[idx] = amp[idx - 1]
    return amp
