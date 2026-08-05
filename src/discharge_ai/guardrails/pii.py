"""
guardrails/pii.py
=================

PII / PHI redaction.

Trigger  : any text containing a patient name, phone number, Aadhaar or PAN.
Action   : mask **before** logging or sending to an external API (LangFuse, LLM).

Redaction is deliberately conservative: it masks identifiers while leaving
clinical content intact, because a discharge summary stripped of its medicines
and doses is useless. Patient names are only masked when the caller supplies
them (`known_names`), which avoids mangling clinician names in the body text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#  Indian national identifiers called out by the specification, plus the
#  contact/­financial identifiers that appear in hospital documents.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    # Aadhaar: 12 digits, usually grouped 4-4-4. Checked before generic phones.
    "aadhaar": re.compile(r"(?<!\d)(\d{4}[ -]?\d{4}[ -]?\d{4})(?!\d)"),
    # PAN: 5 letters, 4 digits, 1 letter.
    "pan": re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b"),
    # E.164 / Indian / US phone shapes.
    "phone": re.compile(
        r"(?<!\d)(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)[ -]?|\d{3}[ -])\d{3}[ -]?\d{4}(?!\d)"
    ),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),
    "mrn": re.compile(r"\bMRN[:\s#]*([A-Z0-9-]{4,15})\b", re.IGNORECASE),
}

MASKS = {
    "aadhaar": "[AADHAAR_REDACTED]",
    "pan": "[PAN_REDACTED]",
    "phone": "[PHONE_REDACTED]",
    "email": "[EMAIL_REDACTED]",
    "ssn": "[SSN_REDACTED]",
    "credit_card": "[CARD_REDACTED]",
    "mrn": "[MRN_REDACTED]",
    "name": "[PATIENT_NAME_REDACTED]",
    "address": "[ADDRESS_REDACTED]",
}

#  Keys whose *values* are always PII regardless of content.
SENSITIVE_KEYS = {
    "address", "phone", "phone_number", "mobile", "contact", "email",
    "aadhaar", "aadhar", "pan", "ssn", "national_id", "insurance_id",
}

#  Street-address shapes: "28 Sycamore Street, Springfield, IL 62704".
ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,4}\s+"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Court|Ct|Drive|Dr|Boulevard|Blvd|"
    r"Straat|straat|Kerkstraat|Weg|Calle|Marg)\b[^\n,]*(?:,[^\n,]{2,40}){0,2}",
    re.IGNORECASE,
)


@dataclass
class RedactionResult:
    text: str
    triggered: bool = False
    categories: list[str] = field(default_factory=list)
    count: int = 0

    @property
    def detail(self) -> str:
        if not self.triggered:
            return "no PII/PHI detected"
        return f"masked {self.count} identifier(s): {', '.join(sorted(set(self.categories)))}"


class PIIRedactor:
    """Mask PII/PHI in strings and nested structures."""

    name = "PIIRedactor"

    def __init__(self, known_names: list[str] | None = None,
                 redact_addresses: bool = True) -> None:
        self.known_names = [n for n in (known_names or []) if n and len(n) > 2]
        self.redact_addresses = redact_addresses

    # ------------------------------------------------------------------ #
    def redact(self, text: str) -> RedactionResult:
        if not text or not isinstance(text, str):
            return RedactionResult(text=text or "")

        result = RedactionResult(text=text)

        for category, pattern in PII_PATTERNS.items():
            def _mask(_match: re.Match[str], _category: str = category) -> str:
                result.categories.append(_category)
                result.count += 1
                return MASKS[_category]

            result.text = pattern.sub(_mask, result.text)

        if self.redact_addresses:
            def _mask_address(_match: re.Match[str]) -> str:
                result.categories.append("address")
                result.count += 1
                return MASKS["address"]

            result.text = ADDRESS_RE.sub(_mask_address, result.text)

        for full_name in self.known_names:
            #  Mask the full name and each substantial name part.
            variants = [full_name] + [
                part for part in re.split(r"\s+", full_name.strip()) if len(part) > 2
            ]
            for variant in sorted(set(variants), key=len, reverse=True):
                pattern = re.compile(rf"(?<!\w){re.escape(variant)}(?!\w)", re.IGNORECASE)
                if pattern.search(result.text):
                    result.text = pattern.sub(MASKS["name"], result.text)
                    result.categories.append("name")
                    result.count += 1

        result.triggered = result.count > 0
        return result

    # ------------------------------------------------------------------ #
    def redact_text(self, text: str) -> str:
        return self.redact(text).text

    def redact_any(self, value: Any) -> Any:
        """Recursively redact strings inside dicts / lists / models."""
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            out: dict[Any, Any] = {}
            for key, item in value.items():
                if str(key).lower() in SENSITIVE_KEYS and isinstance(item, str) and item:
                    out[key] = MASKS.get(str(key).lower(), "[REDACTED]")
                else:
                    out[key] = self.redact_any(item)
            return out
        if isinstance(value, (list, tuple)):
            return [self.redact_any(item) for item in value]
        if hasattr(value, "model_dump"):
            try:
                return self.redact_any(value.model_dump())
            except Exception:  # noqa: BLE001
                return value
        return value

    def contains_pii(self, text: str) -> bool:
        return self.redact(text).triggered
