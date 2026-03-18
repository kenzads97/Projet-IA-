"""
EXOFIT — Module d'évaluation
Calcule toutes les métriques de performance des deux parties.
Fonctionne sans données réelles : utilise les cas synthétiques pour tester le pipeline.
"""

import json
import numpy as np
from collections import defaultdict
from ecg_pipeline import (
    generate_synthetic_ecg, bandpass_filter, extract_features,
    diagnose_by_rules, PATHOLOGIES, PATHOLOGIE_LABELS
)
from prompt_engine import CAS_TESTS


# ─── MÉTRIQUES GÉNÉRIQUES ─────────────────────────────────────────────────

def compute_metrics(y_true: list, y_pred: list, labels: list) -> dict:
    """
    Calcule précision, rappel, F1 et accuracy pour chaque classe.
    Implémentation from scratch (sans sklearn) pour transparence.
    """
    results = {}
    total = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    overall_accuracy = correct / total if total > 0 else 0

    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn,
            "support": tp + fn
        }

    # Macro-average
    macro_p  = np.mean([results[l]["precision"] for l in labels])
    macro_r  = np.mean([results[l]["recall"]    for l in labels])
    macro_f1 = np.mean([results[l]["f1"]        for l in labels])

    return {
        "par_classe": results,
        "accuracy": round(overall_accuracy, 3),
        "macro_precision": round(macro_p, 3),
        "macro_recall": round(macro_r, 3),
        "macro_f1": round(macro_f1, 3),
        "total_cas": total
    }


def confusion_matrix(y_true: list, y_pred: list, labels: list) -> dict:
    """Matrice de confusion."""
    matrix = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return {t: dict(row) for t, row in matrix.items()}


# ─── ÉVALUATION PARTIE 2 : ECG ────────────────────────────────────────────

def evaluate_ecg_pipeline(n_per_class: int = 20) -> dict:
    """
    Évalue le pipeline ECG sur des données synthétiques.
    Remplacer generate_synthetic_ecg par les vraies données dès réception.
    
    n_per_class : nombre d'ECG synthétiques par pathologie
    """
    print(f"Génération de {n_per_class} ECG synthétiques par pathologie...")
    y_true = []
    y_pred = []
    confidences = []
    errors = []

    pathologies_test = [p for p in PATHOLOGIES if p != "normal"]
    pathologies_test.append("normal")

    for patho in pathologies_test:
        for i in range(n_per_class):
            # Générer ECG synthétique
            ecg = generate_synthetic_ecg(patho)
            filtered = bandpass_filter(ecg)
            features = extract_features(filtered)
            result = diagnose_by_rules(features)

            pred = result["pathologie"]
            conf = result["confiance"]

            y_true.append(patho)
            y_pred.append(pred)
            confidences.append(conf)

            if pred != patho:
                errors.append({
                    "vrai_label": patho,
                    "predit": pred,
                    "confiance": conf,
                    "fc_bpm": features["heart_rate_bpm"],
                    "irr_rr": features["rr_irregularity_ratio"],
                    "qrs_ms": features["qrs_width_ms"],
                })

    metrics = compute_metrics(y_true, y_pred, PATHOLOGIES)
    cm = confusion_matrix(y_true, y_pred, PATHOLOGIES)

    return {
        "methode": "règles mathématiques (données synthétiques)",
        "metriques": metrics,
        "matrice_confusion": cm,
        "confiance_moyenne": round(np.mean(confidences), 3),
        "n_erreurs": len(errors),
        "exemples_erreurs": errors[:5],
        "note": "Relancer avec les 30 000 ECG réels et le CNN pour des métriques de production"
    }


# ─── ÉVALUATION PARTIE 1 : PROMPT ENGINE ──────────────────────────────────

def evaluate_prompt_engine_offline() -> dict:
    """
    Évaluation offline du prompt engine sur les cas tests synthétiques.
    Vérifie la structure du prompt et la cohérence des demandes.
    (L'évaluation réelle nécessite les réponses du LLM.)
    """
    from prompt_engine import build_prompt

    results = []
    for cas in CAS_TESTS:
        prompt = build_prompt(cas["patient"])

        # Vérifications structurelles
        checks = {
            "bloc_contexte_present": "INFORMATIONS PATIENT" in prompt["user"],
            "bloc_patient_present": "Symptômes rapportés" in prompt["user"],
            "bloc_demandes_present": "DEMANDES" in prompt["user"],
            "format_json_demande": "diagnostic_preliminaire" in prompt["user"],
            "niveau_urgence_inclus": "niveau_urgence" in prompt["user"],
            "gestion_examens_manquants": "examens_proposes" in prompt["user"],
            "contraintes_specifiees": "Contraintes" in prompt["user"],
        }

        all_pass = all(checks.values())
        results.append({
            "cas_id": cas["id"],
            "description": cas["description"],
            "urgence_attendue": cas["urgence_attendue"],
            "diagnostic_attendu": cas["diagnostic_attendu"],
            "checks": checks,
            "prompt_valide": all_pass,
            "longueur_prompt_chars": len(prompt["user"])
        })

    n_valides = sum(1 for r in results if r["prompt_valide"])
    return {
        "total_cas_testes": len(results),
        "prompts_valides": n_valides,
        "taux_validation": round(n_valides / len(results), 3),
        "detail_par_cas": results,
        "note": "Pour l'évaluation LLM réelle, comparer diagnostic IA vs diagnostic_attendu"
    }


# ─── RAPPORT COMPLET ──────────────────────────────────────────────────────

def generate_full_report() -> dict:
    """Génère le rapport d'évaluation complet des deux parties."""
    print("\n=== EXOFIT — Rapport d'évaluation ===\n")

    print("[1/2] Évaluation du pipeline ECG (données synthétiques)...")
    ecg_eval = evaluate_ecg_pipeline(n_per_class=15)

    print("[2/2] Évaluation du prompt engine (vérification structurelle)...")
    prompt_eval = evaluate_prompt_engine_offline()

    report = {
        "partie_1_prompt_engine": prompt_eval,
        "partie_2_ecg_pipeline": ecg_eval,
        "resume": {
            "ecg_accuracy": ecg_eval["metriques"]["accuracy"],
            "ecg_macro_recall": ecg_eval["metriques"]["macro_recall"],
            "ecg_macro_f1": ecg_eval["metriques"]["macro_f1"],
            "prompt_validation_rate": prompt_eval["taux_validation"],
        }
    }

    # Sauvegarder le rapport
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nRapport sauvegardé : evaluation_report.json")

    return report


def print_report(report: dict):
    """Affiche un résumé lisible du rapport."""
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES PERFORMANCES — EXOFIT")
    print("=" * 60)

    r = report["resume"]
    print(f"\nPartie 1 — Prompt Engine")
    print(f"  Validation structurelle : {r['prompt_validation_rate'] * 100:.0f}%")

    print(f"\nPartie 2 — Pipeline ECG (méthode : règles mathématiques, données synthétiques)")
    print(f"  Accuracy   : {r['ecg_accuracy'] * 100:.1f}%")
    print(f"  Rappel (macro) : {r['ecg_macro_recall'] * 100:.1f}%")
    print(f"  F1 (macro) : {r['ecg_macro_f1'] * 100:.1f}%")

    print(f"\nDétail par pathologie ECG :")
    for label, m in report["partie_2_ecg_pipeline"]["metriques"]["par_classe"].items():
        if m["support"] > 0:
            nom = PATHOLOGIE_LABELS.get(label, label)
            print(f"  {nom[:40]:40s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}")

    print("\n" + "=" * 60)
    print("Note : métriques sur données SYNTHÉTIQUES.")
    print("Relancer avec les 30 000 ECG réels pour les métriques de production.")
    print("=" * 60)


if __name__ == "__main__":
    report = generate_full_report()
    print_report(report)
