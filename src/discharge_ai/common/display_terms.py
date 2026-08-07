"""
common/display_terms.py
=======================

English display layer for structured clinical table fields.

The Clinical Normalizer Agent translates the *narrative* blocks of a record
(discharge diagnosis, instructions, follow-up, and the report narrative) through
the MCP Sampling `medical_lang_bridge` tool, and expands clinical abbreviations
(``TID`` → three times daily) everywhere.  What it does **not** translate are
the short, structured cells of the medication and lab tables — route, duration,
remarks, lab test names and result flags — because those are enumerable
vocabulary rather than prose, and round-tripping every cell through an LLM
would be slow, costly and non-deterministic for values a lookup handles exactly.

This module is that lookup.  It is a **display layer only**: nothing here
mutates an `ExtractedCase`, and the validation rules continue to compare on the
original values via `common.terminology`.  Unknown values fall through
unchanged, so an unrecognised term is shown as written rather than blanked.

It lives in the package rather than in `dashboard/` because both surfaces need
it: the Streamlit tables *and* the exported summary HTML/PDF must show the same
English wording, or a reviewer signs off on one thing and the patient goes home
with another.

Covered source languages: Dutch (nl), Spanish (es), Hindi (hi) — the languages
present in `Data/incoming`.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
#  Administration route
# --------------------------------------------------------------------------- #
ROUTE_TERMS: dict[str, str] = {
    "oral": "Oral",
    "oraal": "Oral",            # nl
    "por via oral": "Oral",     # es
    "via oral": "Oral",         # es
    "mundlich": "Oral",         # de
    "मौखिक": "Oral",             # hi
    "iv": "Intravenous",
    "intraveneus": "Intravenous",       # nl
    "intravenoso": "Intravenous",       # es
    "im": "Intramuscular",
    "intramusculair": "Intramuscular",  # nl
    "intramuscular": "Intramuscular",   # es
    "sc": "Subcutaneous",
    "subcutaan": "Subcutaneous",        # nl
    "subcutaneo": "Subcutaneous",       # es
    "topical": "Topical",
    "topisch": "Topical",               # nl
    "topico": "Topical",                # es
    "inhalation": "Inhalation",
    "inhalatie": "Inhalation",          # nl
    "inhalacion": "Inhalation",         # es
}

# --------------------------------------------------------------------------- #
#  Duration units — the number is preserved, only the unit word is translated.
# --------------------------------------------------------------------------- #
DURATION_UNITS: dict[str, tuple[str, str]] = {
    # source unit (lower, unaccented) -> (singular, plural)
    "day": ("day", "days"),
    "days": ("day", "days"),
    "dag": ("day", "days"),       # nl
    "dagen": ("day", "days"),     # nl
    "dia": ("day", "days"),       # es
    "dias": ("day", "days"),      # es
    "din": ("day", "days"),       # hi (romanised)
    "दिन": ("day", "days"),        # hi
    "week": ("week", "weeks"),
    "weeks": ("week", "weeks"),
    "weken": ("week", "weeks"),   # nl
    "semana": ("week", "weeks"),  # es
    "semanas": ("week", "weeks"),
    "सप्ताह": ("week", "weeks"),
    "month": ("month", "months"),
    "months": ("month", "months"),
    "maand": ("month", "months"),   # nl
    "maanden": ("month", "months"),
    "mes": ("month", "months"),     # es
    "meses": ("month", "months"),
    "महीना": ("month", "months"),
}

# --------------------------------------------------------------------------- #
#  Laboratory test names
# --------------------------------------------------------------------------- #
LAB_TEST_TERMS: dict[str, str] = {
    # Dutch
    "leukocyten": "White Blood Cells",
    "hemoglobine": "Haemoglobin",
    "procalcitonine": "Procalcitonin",
    "creatinine": "Creatinine",
    "trombocyten": "Platelets",
    "natrium": "Sodium",
    "kalium": "Potassium",
    # Spanish
    "glucosa en ayuno": "Fasting Glucose",
    "glucosa en ayunas": "Fasting Glucose",
    "creatinina": "Creatinine",
    "potasio": "Potassium",
    "colesterol ldl": "LDL Cholesterol",
    "colesterol hdl": "HDL Cholesterol",
    "sodio": "Sodium",
    "hemoglobina": "Haemoglobin",
    # Hindi
    "उपवास ग्लूकोज": "Fasting Glucose",
    "क्रिएटिनिन": "Creatinine",
    "पोटेशियम": "Potassium",
    "ldl कोलेस्ट्रॉल": "LDL Cholesterol",
    "एचडीएल कोलेस्ट्रॉल": "HDL Cholesterol",
    "हीमोग्लोबिन": "Haemoglobin",
    "सोडियम": "Sodium",
    # Already-English canonical spellings (normalise capitalisation)
    "ldl cholesterol": "LDL Cholesterol",
    "hdl cholesterol": "HDL Cholesterol",
    "fasting glucose": "Fasting Glucose",
    "potassium": "Potassium",
    "sodium": "Sodium",
    "hba1c": "HbA1c",
    "crp": "CRP",
}

# --------------------------------------------------------------------------- #
#  Result flags
# --------------------------------------------------------------------------- #
FLAG_TERMS: dict[str, str] = {
    "normal": "NORMAL",
    "normaal": "NORMAL",     # nl
    "सामान्य": "NORMAL",      # hi
    "high": "HIGH",
    "hoog": "HIGH",          # nl
    "alto": "HIGH",          # es
    "उच्च": "HIGH",           # hi
    "low": "LOW",
    "laag": "LOW",           # nl
    "bajo": "LOW",           # es
    "कम": "LOW",              # hi
    "critical": "CRITICAL",
    "kritiek": "CRITICAL",   # nl
    "critico": "CRITICAL",   # es
    "abnormal": "ABNORMAL",
    "afwijkend": "ABNORMAL",  # nl
    "anormal": "ABNORMAL",    # es
}

# --------------------------------------------------------------------------- #
#  Short prescription remarks
# --------------------------------------------------------------------------- #
REMARK_TERMS: dict[str, str] = {
    # Dutch
    "volledige kuur afmaken": "Complete the full course",
    "bij koorts": "If fever",
    "met voedsel": "With food",
    "na het eten": "After food",
    "voor het slapen": "At bedtime",
    "s ochtends": "In the morning",
    # Spanish
    "con las comidas": "With meals",
    "controlar la presion": "Monitor blood pressure",
    "por la noche": "At night",
    "cardioprotector": "Cardioprotective",
    "en ayunas": "On an empty stomach",
    "si hay fiebre": "If fever",
    # Hindi
    "भोजन के साथ": "With meals",
    "रात को सोते समय": "At bedtime",
    "सुबह": "In the morning",
    "बुखार होने पर": "If fever",
    "खाली पेट": "On an empty stomach",
}


# --------------------------------------------------------------------------- #
#  Lookup helpers
# --------------------------------------------------------------------------- #
_ACCENTS = str.maketrans("áàâäãéèêëíìîïóòôöõúùûüñç", "aaaaaeeeeiiiiooooouuuunc")


def _key(value: str) -> str:
    """Lookup key: lower-cased, accent-folded, punctuation-trimmed."""
    text = str(value).strip().lower().translate(_ACCENTS)
    return re.sub(r"[.,;:!¡¿?]+$", "", text).strip()


def _lookup(value: str | None, table: dict[str, str]) -> str:
    if not value:
        return ""
    return table.get(_key(value), str(value).strip())


def english_route(value: str | None) -> str:
    """`ORAAL` / `ORAL` → `Oral`. Unknown routes pass through title-cased."""
    if not value:
        return ""
    hit = ROUTE_TERMS.get(_key(value))
    return hit if hit else str(value).strip().title()


def english_duration(value: str | None) -> str:
    """`7 dagen` → `7 days`, `30 días` → `30 days`. Number is preserved."""
    if not value:
        return ""
    raw = str(value).strip()
    match = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(.+?)\s*$", raw)
    if not match:
        return _lookup(raw, {k: v[1] for k, v in DURATION_UNITS.items()}) or raw

    number, unit = match.group(1), match.group(2)
    entry = DURATION_UNITS.get(_key(unit))
    if not entry:
        return raw
    singular, plural = entry
    try:
        is_one = float(number.replace(",", ".")) == 1
    except ValueError:
        is_one = False
    return f"{number} {singular if is_one else plural}"


def english_lab_test(value: str | None) -> str:
    """`Leukocyten` → `White Blood Cells`, `क्रिएटिनिन` → `Creatinine`."""
    return _lookup(value, LAB_TEST_TERMS)


def english_flag(value: str | None) -> str:
    """`HOOG` → `HIGH`. Unknown flags pass through upper-cased."""
    if not value:
        return ""
    hit = FLAG_TERMS.get(_key(value))
    return hit if hit else str(value).strip().upper()


def english_remark(value: str | None) -> str:
    """`Volledige kuur afmaken` → `Complete the full course`."""
    return _lookup(value, REMARK_TERMS)
