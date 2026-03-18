"""
EXOFIT — Prompt Engine
Construit le prompt structuré à 3 blocs et appelle l'API LLM.
Testable sans données réelles grâce aux cas synthétiques.
"""

import json
import httpx
import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Any
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))

# ─── CONFIGURATION ────────────────────────────────────────────────────────

LLM_CONFIGS = {
    "google": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "model": os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
        "header_key": "x-goog-api-key",
        "header_prefix": "",
        "env_key": "GOOGLE_API_KEY",
    },
    "qwen": {
        "url": "https://router.huggingface.co/v1/chat/completions",
        "model": os.getenv("QW_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "header_key": "Authorization",
        "header_prefix": "Bearer ",
        "env_key": "QW_TOKEN",
    },
    "deepseek_hf": {
        "url": "https://router.huggingface.co/v1/chat/completions",
        "model": os.getenv("DS_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        "header_key": "Authorization",
        "header_prefix": "Bearer ",
        "env_key": "DS_TOKEN",
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        "header_key": "Authorization",
        "header_prefix": "Bearer ",
        "env_key": "MISTRAL_API_KEY",
    },
}

ACTIVE_LLM = os.getenv("ACTIVE_LLM", "google").lower()
PARSED_CASES_DIR = Path(__file__).resolve().parent.parent / "data" / "cas_cliniques" / "parsed"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "prompt_engine_results.json"

# ─── LISTE DES SYMPTÔMES STANDARDISÉS ─────────────────────────────────────

def format_symptom_label(symptom: str, custom_labels: dict[str, str] | None = None) -> str:
    """
    Formate un symptome de maniere dynamique.
    Accepte des labels fournis par les donnees si disponibles, sinon humanise la cle.
    """
    if custom_labels and symptom in custom_labels:
        return custom_labels[symptom]

    cleaned = symptom.replace("_", " ").strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else symptom

# ─── CONSTRUCTION DU PROMPT ───────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant médical expert utilisé dans un contexte de télémédecine.
Tu assistes une infirmière qui ne dispose pas de médecin sur place.
Tu as accès au matériel suivant : tensiomètre, thermomètre, stéthoscope, ECG 6 dérivations, analyseur sanguin portable.
Ton rôle est de proposer un diagnostic différentiel structuré, des questions complémentaires à poser au patient, et des examens à envisager.
Tu ne poses PAS de diagnostic définitif — ce rôle revient au médecin téléconsultant.
Si tu détectes un signe de gravité immédiate (douleur thoracique + dyspnée + ECG anormal, trouble de conscience, etc.), indique-le clairement avec le niveau d'urgence CRITIQUE.
Réponds UNIQUEMENT en JSON valide selon le format demandé. Pas de texte avant ou après le JSON."""

def build_prompt(patient: dict) -> dict:
    """
    Construit le prompt structuré en 3 blocs :
    1. Contexte (rôle IA, matériel, objectif)
    2. Informations patient (données structurées)
    3. Demandes précises à l'IA
    """
    symptom_labels = patient.get("symptomes_labels")
    symptomes_str = "\n".join([
        f"  - {format_symptom_label(s, symptom_labels)}"
        for s in patient.get("symptomes", [])
    ])
    if patient.get("symptome_libre"):
        symptomes_str += f"\n  - (Libre) {patient['symptome_libre']}"

    antecedents_str = "\n".join([f"  - {a}" for a in patient.get("antecedents", [])]) or "  - Aucun renseigné"
    traitements_str = "\n".join([f"  - {t}" for t in patient.get("traitements_en_cours", [])]) or "  - Aucun"
    examens_str = "\n".join([f"  - {e}" for e in patient.get("examens_realises", [])]) or "  - Aucun réalisé"

    resultats_str = ""
    if patient.get("resultats_examens"):
        resultats_str = "\n".join([f"  - {k}: {v}" for k, v in patient["resultats_examens"].items()])
    else:
        resultats_str = "  - Non disponibles"

    source_document = patient.get("_source_document")
    raw_case_text = patient.get("_raw_case_text")
    source_doc_block = ""
    if source_document or raw_case_text:
        source_doc_block = f"""

### Document source enseignant
- Source : {source_document or 'Document clinique fourni'}
- Texte brut du cas :
{raw_case_text or '  - Non disponible'}"""

    constantes = []
    if patient.get("tension_systolique") and patient.get("tension_diastolique"):
        constantes.append(f"TA : {patient['tension_systolique']}/{patient['tension_diastolique']} mmHg")
    if patient.get("frequence_cardiaque"):
        constantes.append(f"FC : {patient['frequence_cardiaque']} bpm")
    if patient.get("temperature"):
        constantes.append(f"T° : {patient['temperature']} °C")
    if patient.get("spo2"):
        constantes.append(f"SpO2 : {patient['spo2']} %")
    constantes_str = "\n".join([f"  - {c}" for c in constantes]) or "  - Non mesurées"

    bmi = None
    if patient.get("taille_cm") and patient.get("poids_kg"):
        bmi = round(patient["poids_kg"] / (patient["taille_cm"] / 100) ** 2, 1)

    user_content = f"""## INFORMATIONS PATIENT

- Âge : {patient['age']} ans
- Sexe : {patient['sexe']}
- Taille / Poids : {patient.get('taille_cm', 'NC')} cm / {patient.get('poids_kg', 'NC')} kg{f' (IMC {bmi})' if bmi else ''}

### Symptômes rapportés
{symptomes_str if symptomes_str else '  - Aucun symptôme renseigné'}

### Constantes vitales
{constantes_str}

### Antécédents médicaux
{antecedents_str}

### Traitements en cours
{traitements_str}

### Examens réalisés
{examens_str}

### Résultats d'examens disponibles
{resultats_str}
{source_doc_block}

## DEMANDES

Fournis une réponse JSON STRICTEMENT dans ce format :
{{
  "diagnostic_preliminaire": "string — hypothèse diagnostique principale",
  "explication_diagnostic": "string — justification clinique courte du choix diagnostique",
  "diagnostics_differentiels": [
    {{
      "nom": "string",
      "explication": "string",
      "credibilite": 0
    }}
  ],
  "questions_complementaires": ["string", "string", "string"],
  "examens_proposes": ["string", "string"],
  "traitements_proposes": ["string", "string"],
  "niveau_urgence": "faible|modere|eleve|critique",
  "confiance": 0,
  "modele_utilise": "string"
}}

Contraintes :
- Si des examens sont manquants (ex: ECG non réalisé), mentionne-le dans les examens_proposes
- Propose uniquement des traitements ou mesures initiales compatibles avec une prise en charge infirmiere / telemedecine, sans remplacer la validation medicale
- Le niveau_urgence doit être "critique" si tu détectes un signe de gravité immédiate
- La confiance est un pourcentage entier entre 0 et 100
- La somme des credibilites des diagnostics_differentiels ne doit pas depasser le pourcentage restant apres la confiance du diagnostic principal
- Reste concis et cliniquement pertinent"""

    return {
        "system": SYSTEM_PROMPT,
        "user": user_content
    }


# ─── APPEL LLM ────────────────────────────────────────────────────────────

async def call_llm(prompt: dict, provider: str = ACTIVE_LLM) -> dict:
    """
    Appelle l'API LLM configurée et parse la réponse JSON.
    Supporte OpenAI, Anthropic et Google.
    """
    provider = provider.lower()
    if provider not in LLM_CONFIGS:
        supported = ", ".join(sorted(LLM_CONFIGS))
        raise ValueError(f"Provider LLM inconnu: {provider}. Providers supportes: {supported}")

    config = LLM_CONFIGS[provider]
    api_key = os.getenv(config["env_key"])

    if not api_key:
        # MODE DÉMO : retourne un diagnostic synthétique sans API
        return _demo_response(provider)

    url = config["url"].format(model=config["model"])

    headers = {
        "Content-Type": "application/json",
        config["header_key"]: f"{config['header_prefix']}{api_key}"
    }
    # Formater le body selon le provider
    if provider in {"qwen", "deepseek_hf", "mistral"}:
        body = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ],
            "temperature": 0.3,
        }
    else:  # google
        body = {
            "contents": [{"parts": [{"text": prompt["system"] + "\n\n" + prompt["user"]}]}]
        }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=body)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            details = exc.response.text
            raise ValueError(
                f"Erreur API {provider} ({exc.response.status_code}): {details}"
            ) from exc
        data = response.json()

    # Extraire le texte de la réponse selon le provider
    if provider in {"qwen", "deepseek_hf", "mistral"}:
        text = data["choices"][0]["message"]["content"]
    else:
        text = data["candidates"][0]["content"]["parts"][0]["text"]

    # Parser le JSON
    result = _normalize_result(_parse_json_response(text))
    result["modele_utilise"] = f"{provider}/{config['model']}"
    return result


def _parse_json_response(text: Any) -> dict:
    """
    Parse de facon robuste une reponse JSON provenant d'un LLM.
    Gere les cas ou le modele renvoie du JSON dans un bloc ```json ... ```.
    """
    if isinstance(text, list):
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in text
        )

    if not isinstance(text, str):
        raise ValueError("La reponse du modele n'est pas du texte JSON exploitable.")

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Reponse non JSON du modele: {cleaned[:300]}")
        return json.loads(cleaned[start:end + 1])


def _normalize_percentage(value: Any) -> int:
    """Normalise une valeur de confiance/credibilite en pourcentage 0-100."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0

    if 0 <= numeric <= 1:
        numeric *= 100
    return max(0, min(100, int(round(numeric))))


def _normalize_differentials(items: Any, remaining_budget: int) -> list[dict]:
    """Normalise les diagnostics differentiels et les cale dans le budget restant."""
    normalized = []
    if not isinstance(items, list):
        return normalized

    for item in items:
        if isinstance(item, str):
            normalized.append({
                "nom": item,
                "explication": "",
                "credibilite": 0,
            })
        elif isinstance(item, dict):
            normalized.append({
                "nom": str(item.get("nom", "")),
                "explication": str(item.get("explication", "")),
                "credibilite": _normalize_percentage(item.get("credibilite", 0)),
            })

    total = sum(item["credibilite"] for item in normalized)
    if total <= remaining_budget or total == 0:
        return normalized

    ratio = remaining_budget / total if total > 0 else 0
    scaled = []
    for item in normalized:
        scaled.append({
            **item,
            "credibilite": int(item["credibilite"] * ratio),
        })

    current_total = sum(item["credibilite"] for item in scaled)
    idx = 0
    while current_total < remaining_budget and scaled:
        scaled[idx % len(scaled)]["credibilite"] += 1
        current_total += 1
        idx += 1

    return scaled


def _normalize_result(result: dict) -> dict:
    """Met les scores sur 100 et borne les differenciels au reliquat disponible."""
    confidence = _normalize_percentage(result.get("confiance", 0))
    remaining = max(0, 100 - confidence)
    result["confiance"] = confidence
    result["diagnostics_differentiels"] = _normalize_differentials(
        result.get("diagnostics_differentiels", []),
        remaining_budget=remaining,
    )
    return result


def _demo_response(provider: str) -> dict:
    """Réponse de démonstration quand aucune API key n'est configurée."""
    return {
        "diagnostic_preliminaire": "MODE DÉMO — Configurez une clé API dans le fichier .env",
        "explication_diagnostic": "Aucune API n'est configuree. Le systeme renvoie une reponse de demonstration sans analyse clinique reelle.",
        "diagnostics_differentiels": [
            {
                "nom": "Douleur thoracique d'origine coronarienne",
                "explication": "Compatible avec une douleur thoracique possiblement ischemique a explorer rapidement.",
                "credibilite": 76
            },
            {
                "nom": "Syndrome coronarien aigu",
                "explication": "Hypothese importante si la douleur est oppressante ou associee a des signes de gravite.",
                "credibilite": 16
            },
            {
                "nom": "Pericardite aigue",
                "explication": "Differentiel possible selon le caractere de la douleur et les donnees ECG ou inflammatoires.",
                "credibilite": 8
            }
        ],
        "questions_complementaires": [
            "La douleur irradie-t-elle dans le bras gauche ou la mâchoire ?",
            "Avez-vous déjà eu ce type de douleur ?",
            "La douleur s'aggrave-t-elle à l'effort ?"
        ],
        "examens_proposes": [
            "ECG 12 dérivations en urgence",
            "Troponine (si analyseur disponible)",
            "Saturation en oxygène"
        ],
        "traitements_proposes": [
            "Mettre le patient au repos et assurer une surveillance rapprochee",
            "Oxygene si hypoxemie et selon protocole local",
            "Teleconsultation medicale urgente pour validation therapeutique"
        ],
        "niveau_urgence": "eleve",
        "confiance": 0,
        "modele_utilise": f"{provider}/demo"
    }


# ─── CAS TESTS SYNTHÉTIQUES ───────────────────────────────────────────────

CAS_TESTS = [
    {
        "id": "CAS_001",
        "description": "Infarctus du myocarde typique",
        "patient": {
            "age": 58, "sexe": "M", "taille_cm": 175, "poids_kg": 85,
            "symptomes": ["douleur_thoracique", "dyspnee", "nausees_vomissements"],
            "symptome_libre": "Douleur oppressive irradiant dans le bras gauche depuis 45 min",
            "tension_systolique": 145, "tension_diastolique": 92,
            "frequence_cardiaque": 98, "temperature": 37.1, "spo2": 94,
            "antecedents": ["HTA", "tabagisme 20 PA", "diabète type 2"],
            "traitements_en_cours": ["metformine 1g", "amlodipine 5mg"],
            "examens_realises": ["ecg"],
            "resultats_examens": {"ecg": "Sus-décalage ST en V1-V4, onde Q en V1-V2"}
        },
        "diagnostic_attendu": "Infarctus du myocarde antérieur",
        "urgence_attendue": "critique"
    },
    {
        "id": "CAS_002",
        "description": "Fibrillation auriculaire",
        "patient": {
            "age": 72, "sexe": "F", "taille_cm": 162, "poids_kg": 68,
            "symptomes": ["palpitations", "asthenie", "dyspnee"],
            "symptome_libre": "Palpitations irrégulières depuis hier soir",
            "tension_systolique": 132, "tension_diastolique": 78,
            "frequence_cardiaque": 118, "temperature": 36.8, "spo2": 97,
            "antecedents": ["HTA", "hypothyroïdie"],
            "traitements_en_cours": ["levothyroxine 75µg", "ramipril 5mg"],
            "examens_realises": ["ecg"],
            "resultats_examens": {"ecg": "Rythme irrégulier, absence d'onde P, trémulations de la ligne isoélectrique"}
        },
        "diagnostic_attendu": "Fibrillation auriculaire",
        "urgence_attendue": "eleve"
    },
    {
        "id": "CAS_003",
        "description": "Angine bactérienne simple",
        "patient": {
            "age": 24, "sexe": "F", "taille_cm": 168, "poids_kg": 60,
            "symptomes": ["fievre", "cephalees", "asthenie"],
            "symptome_libre": "Gorge très douloureuse depuis 2 jours, difficulté à avaler",
            "tension_systolique": 118, "tension_diastolique": 72,
            "frequence_cardiaque": 88, "temperature": 38.9, "spo2": 99,
            "antecedents": [],
            "traitements_en_cours": [],
            "examens_realises": [],
            "resultats_examens": {}
        },
        "diagnostic_attendu": "Angine bactérienne (streptococcique)",
        "urgence_attendue": "faible"
    }
]


def load_parsed_cases(parsed_dir: Path = PARSED_CASES_DIR) -> list[dict]:
    """
    Charge les cas cliniques reellement parses depuis data/cas_cliniques/parsed.
    Ignore index.json, qui ne contient qu'un sommaire.
    """
    if not parsed_dir.exists():
        return []

    cases = []
    for path in sorted(parsed_dir.glob("*.json")):
        if path.name == "index.json":
            continue

        with path.open(encoding="utf-8") as f:
            case_data = json.load(f)

        patient = case_data.get("patient")
        if not patient:
            continue

        patient = dict(patient)
        patient["_source_document"] = case_data.get("source_document")
        patient["_raw_case_text"] = case_data.get("raw_text")

        cases.append({
            "id": case_data.get("id", path.stem),
            "description": (
                case_data.get("source_document", "Cas clinique")
                + f" - cas {case_data.get('case_number', '?')}"
            ),
            "patient": patient,
            "diagnostic_attendu": case_data.get("diagnostic_reference"),
            "urgence_attendue": case_data.get("urgence_reference"),
            "source_document": case_data.get("source_document"),
            "raw_text": case_data.get("raw_text"),
        })

    return cases


def get_available_cases() -> list[dict]:
    """
    Retourne les vrais cas cliniques parses si disponibles, sinon les cas de demo.
    """
    parsed_cases = load_parsed_cases()
    return parsed_cases if parsed_cases else CAS_TESTS


def get_case_by_id(case_id: str) -> dict | None:
    """Retourne un cas par son identifiant."""
    for case in get_available_cases():
        if case["id"] == case_id:
            return case
    return None


async def diagnose_case(case_id: str, provider: str = ACTIVE_LLM) -> dict:
    """Execute un diagnostic pour un cas clinique donne."""
    case = get_case_by_id(case_id)
    if not case:
        raise ValueError(f"Cas introuvable: {case_id}")

    prompt = build_prompt(case["patient"])
    result = await call_llm(prompt, provider=provider)
    return {
        "case_id": case["id"],
        "description": case["description"],
        "source_document": case.get("source_document"),
        "patient": case["patient"],
        "result": result,
    }


async def diagnose_case_with_providers(case_id: str, providers: list[str]) -> dict:
    """Lance plusieurs modeles generatifs sur le meme cas."""
    case = get_case_by_id(case_id)
    if not case:
        raise ValueError(f"Cas introuvable: {case_id}")

    normalized = []
    for provider in providers or [ACTIVE_LLM]:
        provider = provider.lower()
        if provider not in normalized:
            normalized.append(provider)

    prompt = build_prompt(case["patient"])

    async def run_provider(provider: str) -> dict:
        try:
            result = await call_llm(prompt, provider=provider)
            return {
                "provider": provider,
                "status": "ok",
                "result": result,
            }
        except Exception as exc:
            return {
                "provider": provider,
                "status": "error",
                "error": str(exc),
            }

    comparisons = await asyncio.gather(*(run_provider(provider) for provider in normalized))

    return {
        "case_id": case["id"],
        "description": case["description"],
        "source_document": case.get("source_document"),
        "patient": case["patient"],
        "comparisons": comparisons,
    }


async def run_cases_with_chatgpt(
    provider: str = ACTIVE_LLM,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = None,
) -> dict:
    """
    Execute tous les cas disponibles avec le LLM configure et sauvegarde les resultats.
    """
    cases = get_available_cases()
    if limit is not None:
        cases = cases[:limit]

    results = []
    for index, cas in enumerate(cases, start=1):
        prompt = build_prompt(cas["patient"])

        try:
            llm_result = await call_llm(prompt, provider=provider)
            results.append({
                "case_id": cas["id"],
                "description": cas["description"],
                "source_document": cas.get("source_document"),
                "diagnostic_attendu": cas.get("diagnostic_attendu"),
                "urgence_attendue": cas.get("urgence_attendue"),
                "llm_response": llm_result,
                "status": "ok",
            })
        except Exception as exc:
            results.append({
                "case_id": cas["id"],
                "description": cas["description"],
                "source_document": cas.get("source_document"),
                "diagnostic_attendu": cas.get("diagnostic_attendu"),
                "urgence_attendue": cas.get("urgence_attendue"),
                "status": "error",
                "error": str(exc),
            })

        print(f"[{index}/{len(cases)}] {cas['id']} traite")

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "provider": provider,
        "total_cases": len(cases),
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


if __name__ == "__main__":
    import asyncio

    async def test():
        run_all_cases = os.getenv("RUN_ALL_CASES", "0") == "1"
        limit_env = os.getenv("CASE_LIMIT")
        limit = int(limit_env) if limit_env else None

        if run_all_cases:
            print("=== Execution des cas cliniques du professeur ===\n")
            result = await run_cases_with_chatgpt(limit=limit)
            print(f"\nResultats sauvegardes dans : {DEFAULT_OUTPUT_PATH}")
            print(f"Cas traites : {result['total_cases']}")
            return

        print("=== Apercu du prompt engine ===\n")
        cases = get_available_cases()
        print(f"{len(cases)} cas charges\n")

        for cas in cases[:limit or len(cases)]:
            print(f"--- {cas['id']} : {cas['description']} ---")
            prompt = build_prompt(cas["patient"])
            print("PROMPT USER (extrait) :")
            print(prompt["user"][:300], "...\n")
            print(f"Diagnostic attendu : {cas.get('diagnostic_attendu')}")
            print(f"Urgence attendue   : {cas.get('urgence_attendue')}\n")

    asyncio.run(test())
