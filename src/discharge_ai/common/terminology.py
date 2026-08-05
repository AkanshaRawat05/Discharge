"""
common/terminology.py
=====================

Clinical terminology utilities shared by the Normalizer Agent, the Validation
Agent and the MCP `resource://medical-abbreviations` resource:

* medical abbreviation expansion  (rules.yaml map + a wider clinical set)
* multilingual drug-name canonicalisation  (Metformina/Amoxicilline/… → INN)
* allergy ↔ drug-class conflict detection  (Penicillin ↔ Amoxicillin)
* lightweight language detection (script + keyword heuristics, no model)

The drug tables matter for correctness: the Dutch note prescribing
"Amoxicilline" must still collide with a "Penicillin" allergy on file, and the
Spanish "Metformina" must reconcile against the EHR order "Metformin".
"""

from __future__ import annotations

import re
import unicodedata

from .rules import abbreviation_map

# --------------------------------------------------------------------------- #
#  1. Medical abbreviations
# --------------------------------------------------------------------------- #
#  rules.yaml is authoritative; these extend it with the frequency/route codes
#  the specification calls out explicitly (BID = twice daily, PO = by mouth).
EXTRA_ABBREVIATIONS: dict[str, str] = {
    "PO": "by mouth",
    "IV": "intravenous",
    "IM": "intramuscular",
    "SC": "subcutaneous",
    "SL": "sublingual",
    "PR": "per rectum",
    "NPO": "nothing by mouth",
    "q4h": "every 4 hours",
    "q6h": "every 6 hours",
    "q8h": "every 8 hours",
    "q12h": "every 12 hours",
    "QOD": "every other day",
    "STAT": "immediately",
    "AC": "before meals",
    "PC": "after meals",
    "OD": "once daily",
    "NKDA": "No Known Drug Allergies",
    "ADR": "Adverse Drug Reaction",
    "ED": "Emergency Department",
    "ER": "Emergency Room",
    "ICU": "Intensive Care Unit",
    "LOS": "Length of Stay",
    "PCP": "Primary Care Physician",
    "F/U": "Follow-up",
    "WBC": "White Blood Cell count",
    "CRP": "C-Reactive Protein",
    "BNP": "B-type Natriuretic Peptide",
    "LDL": "Low-Density Lipoprotein cholesterol",
    "HDL": "High-Density Lipoprotein cholesterol",
    "HbA1c": "Glycated Haemoglobin",
    "INR": "International Normalised Ratio",
    "eGFR": "estimated Glomerular Filtration Rate",
    "SpO2": "Peripheral Oxygen Saturation",
    "BG": "Blood Glucose",
    "CXR": "Chest X-Ray",
    "SABA": "Short-Acting Beta Agonist",
    "ICS": "Inhaled Corticosteroid",
    "LAMA": "Long-Acting Muscarinic Antagonist",
    "HFA": "Hydrofluoroalkane inhaler",
    "T2DM": "Type 2 Diabetes Mellitus",
    "CAP": "Community-Acquired Pneumonia",
}


def full_abbreviation_map() -> dict[str, str]:
    """rules.yaml abbreviations merged with the extended clinical set."""
    merged = dict(EXTRA_ABBREVIATIONS)
    merged.update(abbreviation_map())          # rules.yaml wins on conflicts
    return merged


def expand_abbreviations(text: str) -> tuple[str, list[dict[str, str]]]:
    """Expand abbreviations in `text` as "expansion (ABBR)".

    Returns the rewritten text and the list of expansions applied, which the
    Normalizer Agent reports and LangFuse records.
    """
    if not text:
        return text, []

    applied: list[dict[str, str]] = []
    result = text

    # Longest first so "T2DM" is not shadowed by "DM".
    for abbr, expansion in sorted(
        full_abbreviation_map().items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(abbr)}(?![A-Za-z0-9])")
        if not pattern.search(result):
            continue
        # Skip when the text already spells the expansion out next to it.
        if f"{expansion} ({abbr})" in result:
            continue
        result = pattern.sub(f"{expansion} ({abbr})", result)
        applied.append({"abbreviation": abbr, "expansion": expansion})

    return result, applied


# --------------------------------------------------------------------------- #
#  2. Drug-name canonicalisation
# --------------------------------------------------------------------------- #
#  Localised spellings and common synonyms → canonical INN name (lower case).
DRUG_SYNONYMS: dict[str, str] = {
    # --- Spanish -----------------------------------------------------------
    "metformina": "metformin",
    "atorvastatina": "atorvastatin",
    "aspirina": "aspirin",
    "lisinoprilo": "lisinopril",
    "amoxicilina": "amoxicillin",
    "paracetamol": "acetaminophen",
    "ondansetrón": "ondansetron",
    "loperamida": "loperamide",
    "hioscina": "hyoscine",
    "azitromicina": "azithromycin",
    "ciprofloxacino": "ciprofloxacin",
    "nitrofurantoína": "nitrofurantoin",
    "furosemida": "furosemide",
    "prednisona": "prednisone",
    "insulina": "insulin",
    "amlodipino": "amlodipine",
    "warfarina": "warfarin",
    "heparina": "heparin",
    "digoxina": "digoxin",
    "metotrexato": "methotrexate",
    # --- Dutch / German ----------------------------------------------------
    "amoxicilline": "amoxicillin",
    "ciprofloxacine": "ciprofloxacin",
    "azitromycine": "azithromycin",
    "nitrofurantoine": "nitrofurantoin",
    "metoprololsuccinaat": "metoprolol",
    "acetylsalicylzuur": "aspirin",
    "acetylsalicylsäure": "aspirin",
    "ibuprofeen": "ibuprofen",
    "amoxicillin-clavulansäure": "amoxicillin-clavulanate",
    "prednisolon": "prednisolone",
    # --- French ------------------------------------------------------------
    "amoxicilline-acide clavulanique": "amoxicillin-clavulanate",
    "ciprofloxacine 500": "ciprofloxacin",
    "nitrofurantoïne": "nitrofurantoin",
    "acétaminophène": "acetaminophen",
    # --- Hindi (Devanagari) ------------------------------------------------
    "मेटफॉर्मिन": "metformin",
    "एटोरवास्टेटिन": "atorvastatin",
    "एम्लोडिपिन": "amlodipine",
    "एजिथ्रोमाइसिन": "azithromycin",
    "पैरासिटामोल": "acetaminophen",
    "एस्पिरिन": "aspirin",
    "लिसिनोप्रिल": "lisinopril",
    # --- English synonyms / brand-ish forms --------------------------------
    "amox-clav": "amoxicillin-clavulanate",
    "amoxicillin clavulanate": "amoxicillin-clavulanate",
    "amoxicillin-clavulanate": "amoxicillin-clavulanate",
    "augmentin": "amoxicillin-clavulanate",
    "co-amoxiclav": "amoxicillin-clavulanate",
    "tylenol": "acetaminophen",
    "asa": "aspirin",
    "albuterol": "salbutamol",
    "albuterol hfa": "salbutamol",
    "salbutamol hfa": "salbutamol",
    "ventolin": "salbutamol",
    "fluticasone hfa": "fluticasone",
    "coumadin": "warfarin",
    "lasix": "furosemide",
    "glucophage": "metformin",
}

#  Dose/route/form noise that must be stripped before comparison.
_DOSE_NOISE = re.compile(
    r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?|%|puffs?|tab(let)?s?|caps?(ule)?s?|"
    r"gotas|comprimidos|tabletten|capsules|goliyan|गोली|गोलियाँ)\b",
    re.IGNORECASE,
)
_FORM_NOISE = re.compile(
    r"\b(hfa|mdi|inhaler|nebuliser|nebulizer|oral|po|iv|im|sc|er|xr|sr|cr|"
    r"tablet|tablets|capsule|capsules|syrup|solution|injection|susp(ension)?)\b",
    re.IGNORECASE,
)


def canonical_drug(name: str | None) -> str:
    """Canonical, comparable drug key: lower-case INN with noise stripped."""
    if not name:
        return ""

    text = unicodedata.normalize("NFC", str(name)).strip().lower()
    text = text.replace("–", "-").replace("—", "-")

    # Exact synonym hit before any stripping (handles Devanagari + accents).
    if text in DRUG_SYNONYMS:
        return DRUG_SYNONYMS[text]

    text = _DOSE_NOISE.sub(" ", text)
    text = _FORM_NOISE.sub(" ", text)
    text = re.sub(r"[()\[\].,;:/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -")

    if text in DRUG_SYNONYMS:
        return DRUG_SYNONYMS[text]

    # Accent-folded retry (Nitrofurantoína → nitrofurantoina).
    folded = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    if folded in DRUG_SYNONYMS:
        return DRUG_SYNONYMS[folded]

    # First token only, so "metformin hydrochloride" == "metformin".
    first = folded.split(" ")[0] if folded else ""
    if first in DRUG_SYNONYMS:
        return DRUG_SYNONYMS[first]

    return folded or text


# --------------------------------------------------------------------------- #
#  3. Allergy ↔ drug conflict detection
# --------------------------------------------------------------------------- #
#  Allergy label (lower case) → canonical drugs that are contraindicated.
ALLERGY_DRUG_CONFLICTS: dict[str, set[str]] = {
    "penicillin": {
        "penicillin", "penicillin v", "penicillin g", "amoxicillin",
        "amoxicillin-clavulanate", "ampicillin", "ampicillin-sulbactam",
        "piperacillin", "piperacillin-tazobactam", "flucloxacillin",
        "dicloxacillin", "nafcillin", "oxacillin", "benzylpenicillin",
    },
    "cephalosporin": {"cefazolin", "ceftriaxone", "cefuroxime", "cefepime", "cephalexin"},
    "sulfa": {
        "sulfamethoxazole", "trimethoprim-sulfamethoxazole", "co-trimoxazole",
        "sulfasalazine", "furosemide",   # sulfonamide cross-reactivity
    },
    "nsaid": {"ibuprofen", "naproxen", "diclofenac", "ketorolac", "aspirin"},
    "aspirin": {"aspirin"},
    "latex": set(),                      # device allergy — no oral drug conflict
    "iodine": {"iohexol", "iopamidol", "povidone-iodine"},
    "macrolide": {"azithromycin", "clarithromycin", "erythromycin"},
    "statin": {"atorvastatin", "simvastatin", "rosuvastatin", "pravastatin"},
    "metformin": {"metformin"},
    "quinolone": {"ciprofloxacin", "levofloxacin", "moxifloxacin", "ofloxacin"},
}

#  Multilingual allergy spellings → the canonical allergy key above.
ALLERGY_ALIASES: dict[str, str] = {
    "penicilline": "penicillin",
    "penicilina": "penicillin",
    "penizillin": "penicillin",
    "पेनिसिलिन": "penicillin",
    "sulfa drugs": "sulfa",
    "sulfonamide": "sulfa",
    "sulfonamides": "sulfa",
    "nsaids": "nsaid",
    "cephalosporins": "cephalosporin",
    "macrolides": "macrolide",
    "quinolones": "quinolone",
    "fluoroquinolone": "quinolone",
    "jodium": "iodine",
    "yodo": "iodine",
    "latexallergie": "latex",
}

_NO_ALLERGY_MARKERS = {
    "nkda", "none", "no known drug allergies", "no known allergies", "nil",
    "sin alergias conocidas", "keine bekannten allergien", "geen bekende allergieën",
    "कोई ज्ञात एलर्जी नहीं", "aucune allergie connue", "n/a", "-",
}


def canonical_allergy(label: str | None) -> str:
    if not label:
        return ""
    text = unicodedata.normalize("NFC", str(label)).strip().lower()
    text = re.sub(r"\((.*?)\)", " ", text)                 # drop "(documented rash)"
    text = re.sub(r"[.,;:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in _NO_ALLERGY_MARKERS:
        return ""
    if text in ALLERGY_ALIASES:
        return ALLERGY_ALIASES[text]
    if text in ALLERGY_DRUG_CONFLICTS:
        return text

    # Substring match: "penicilline (gedocumenteerde huiduitslag)" → penicillin
    for alias, canonical in ALLERGY_ALIASES.items():
        if alias in text:
            return canonical
    for key in ALLERGY_DRUG_CONFLICTS:
        if key in text:
            return key

    folded = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    return folded.split(" ")[0] if folded else text


def is_no_known_allergy(label: str | None) -> bool:
    if not label:
        return True
    text = str(label).strip().lower()
    return any(marker in text for marker in _NO_ALLERGY_MARKERS)


def allergy_conflicts(allergies: list[str], medications: list[str]) -> list[dict[str, str]]:
    """Every (allergy, medication) pair that is clinically contraindicated."""
    conflicts: list[dict[str, str]] = []

    for raw_allergy in allergies or []:
        allergy_key = canonical_allergy(raw_allergy)
        if not allergy_key:
            continue
        contraindicated = ALLERGY_DRUG_CONFLICTS.get(allergy_key, {allergy_key})

        for raw_med in medications or []:
            med_key = canonical_drug(raw_med)
            if not med_key:
                continue
            if med_key in contraindicated or med_key == allergy_key:
                conflicts.append(
                    {
                        "allergy": str(raw_allergy).strip(),
                        "allergy_class": allergy_key,
                        "medication": str(raw_med).strip(),
                        "medication_canonical": med_key,
                    }
                )
    return conflicts


# --------------------------------------------------------------------------- #
#  4. Language detection + demographic value normalisation
# --------------------------------------------------------------------------- #
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "hi": "Hindi", "de": "German",
    "fr": "French", "nl": "Dutch",
}

_SCRIPT_RANGES = {"hi": (0x0900, 0x097F)}          # Devanagari

_LANGUAGE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "es": ("resumen de alta", "paciente", "medicamento", "alergias", "seguimiento",
           "fecha de ingreso", "recetas", "instrucciones", "médico"),
    "de": ("entlassungsbericht", "patient", "medikamente", "allergien",
           "nachsorge", "aufnahmedatum", "entlassung", "arzt", "krankenhaus"),
    "nl": ("ontslagbrief", "patiëntnummer", "geneesmiddel", "allergieën",
           "vervolgafspraak", "opnamedatum", "ontslagdatum", "ziekenhuis",
           "afdeling", "toedieningsweg"),
    "fr": ("compte rendu", "sortie", "patient", "médicament", "allergies",
           "suivi", "hôpital", "ordonnance"),
    "en": ("discharge summary", "patient", "medication", "allergies",
           "follow-up", "admission date", "prescriptions", "hospital"),
}


def detect_language(text: str, declared: str | None = None) -> str:
    """ISO-639-1 code for `text`; a declared code always wins."""
    if declared:
        code = str(declared).strip().lower()
        if code in LANGUAGE_NAMES:
            return code
        for iso, name in LANGUAGE_NAMES.items():
            if name.lower() == code:
                return iso

    if not text:
        return "en"

    for code, (low, high) in _SCRIPT_RANGES.items():
        if sum(1 for ch in text[:4000] if low <= ord(ch) <= high) > 12:
            return code

    lowered = text.lower()
    scores = {
        code: sum(1 for kw in keywords if kw in lowered)
        for code, keywords in _LANGUAGE_KEYWORDS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "en"


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get((code or "en").lower(), code or "English")


_GENDER_MAP = {
    "m": "Male", "male": "Male", "man": "Male", "masculino": "Male",
    "männlich": "Male", "mannelijk": "Male", "homme": "Male", "पुरुष": "Male",
    "f": "Female", "female": "Female", "vrouw": "Female", "femenino": "Female",
    "weiblich": "Female", "vrouwelijk": "Female", "femme": "Female",
    "महिला": "Female", "स्त्री": "Female",
}


def normalize_gender(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    return _GENDER_MAP.get(key, str(value).strip() or None)


_YES_VALUES = {"yes", "y", "true", "ja", "sí", "si", "oui", "हाँ", "हां", "approved", "ok"}
_NO_VALUES = {"no", "n", "false", "nee", "nein", "non", "नहीं", "pending"}


def parse_bool(value: object) -> bool | None:
    """Parse multilingual yes/no markers into a bool (None when unknown)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _YES_VALUES:
        return True
    if text in _NO_VALUES:
        return False
    return None


PAYMENT_STATUS_MAP = {
    "paid": "PAID", "settled": "PAID", "betaald": "PAID", "pagado": "PAID",
    "bezahlt": "PAID", "payé": "PAID", "भुगतान किया": "PAID",
    "unpaid": "UNPAID", "outstanding": "UNPAID", "pending": "UNPAID",
    "onbetaald": "UNPAID", "no pagado": "UNPAID", "pendiente": "UNPAID",
    "unbezahlt": "UNPAID", "offen": "UNPAID", "impayé": "UNPAID",
    "भुगतान नहीं किया": "UNPAID", "बकाया": "UNPAID",
    "partially paid": "PARTIAL", "partial": "PARTIAL",
    "insurance guarantee": "GUARANTEED",
}


def normalize_payment_status(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower()
    if text in PAYMENT_STATUS_MAP:
        return PAYMENT_STATUS_MAP[text]
    for key, canonical in PAYMENT_STATUS_MAP.items():
        if key in text:
            return canonical
    return str(value).strip().upper()
