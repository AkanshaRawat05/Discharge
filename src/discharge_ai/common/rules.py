"""
common/rules.py
===============

Loader for `configs/rules.yaml` plus the completeness / cross-validation rule
catalogues from the specification (Tables 3 and 4).

`rules.yaml` is the single source of truth for mandatory fields, risk weights
and thresholds, and every audit report is stamped with the SHA-256 of the file
(`rules_version`) for compliance reproducibility.

The Primary MCP Server serves the two catalogues below through the MCP
**Resources** primitive:

    resource://clinical-rules/completeness
    resource://clinical-rules/cross-validation
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import yaml

from ..settings import settings


# --------------------------------------------------------------------------- #
#  rules.yaml
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_rules() -> dict[str, Any]:
    with settings.rules_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def rules_version() -> str:
    """SHA-256 of rules.yaml — stamped onto every audit report."""
    digest = hashlib.sha256(settings.rules_path.read_bytes()).hexdigest()
    return f"sha256:{digest[:16]}"


def risk_weights() -> dict[str, int]:
    return load_rules().get("risk_scoring_matrix", {}).get("weights", {})


def risk_thresholds() -> dict[str, int]:
    matrix = load_rules().get("risk_scoring_matrix", {})
    thresholds = matrix.get("thresholds", {})
    return {
        "low_max": int(thresholds.get("low_max", 2)),
        "medium_max": int(thresholds.get("medium_max", 8)),
    }


def hard_guardrails() -> list[str]:
    return load_rules().get("risk_scoring_matrix", {}).get("hitl_hard_guardrails", [])


def quality_thresholds() -> dict[str, Any]:
    return load_rules().get("quality_thresholds", {})


def recommendation_text(level: str) -> str:
    mapping = load_rules().get("reporting", {}).get("recommendations", {})
    return mapping.get(level.lower(), "")


def high_risk_medications() -> list[str]:
    policies = load_rules().get("clinical_validation_policies", {})
    return policies.get("high_risk_meds_need_counseling", [])


def always_hitl_service_lines() -> dict[str, str]:
    """Service lines that force HITL regardless of score (rules.yaml §3)."""
    policies = load_rules().get("clinical_validation_policies", {})
    mapping = {}
    if policies.get("pediatric_always_hitl"):
        mapping["pediatrics"] = "service_line_pediatric"
    if policies.get("obstetric_always_hitl"):
        mapping["obstetric"] = "service_line_obstetric"
    if policies.get("oncology_always_hitl"):
        mapping["oncology"] = "service_line_oncology"
    return mapping


def abbreviation_map() -> dict[str, str]:
    return load_rules().get("normalization_standards", {}).get("abbreviation_map", {})


def icd10_map() -> dict[str, str]:
    return load_rules().get("normalization_standards", {}).get("icd10_map", {})


def supported_languages() -> list[str]:
    return load_rules().get("normalization_standards", {}).get(
        "language_codes_supported", ["en"]
    )


def business_rules() -> dict[str, Any]:
    return load_rules().get("business_rules", {})


# --------------------------------------------------------------------------- #
#  Completeness catalogue  (specification Table 3)
# --------------------------------------------------------------------------- #
#  `blocking=True` fields halt auto-generation of the discharge summary and
#  force HITL.  Non-blocking gaps are what the Rules Engine tool tries to fill
#  through MCP Elicitation before falling back to HITL.
COMPLETENESS_RULES: dict[str, dict[str, Any]] = {
    "discharge_report": {
        "label": "Discharge Report",
        "fields": [
            {"field": "patient_id",                "blocking": True,  "type": "str"},
            {"field": "patient_name",              "blocking": True,  "type": "str"},
            {"field": "age",                       "blocking": False, "type": "int"},
            {"field": "gender",                    "blocking": False, "type": "str"},
            {"field": "address",                   "blocking": False, "type": "str"},
            {"field": "admission_date",            "blocking": False, "type": "str"},
            {"field": "discharge_date",            "blocking": False, "type": "str"},
            {"field": "ward",                      "blocking": False, "type": "str"},
            {"field": "bed_no",                    "blocking": False, "type": "str"},
            {"field": "doctors",                   "blocking": False, "type": "str"},
            {"field": "discharge_diagnosis",       "blocking": True,  "type": "list"},
            {"field": "medications",               "blocking": True,  "type": "list"},
            {"field": "adr_allergy_info",          "blocking": False, "type": "list"},
            {"field": "follow_up_appointments",    "blocking": False, "type": "str"},
            {"field": "discharge_instructions",    "blocking": False, "type": "str"},
            {"field": "discharge_approved_by",     "blocking": False, "type": "str"},
            {"field": "discharge_approved",        "blocking": True,  "type": "bool"},
        ],
    },
    "lab_report": {
        "label": "Lab Report",
        "fields": [
            {"field": "patient_id",   "blocking": True,  "type": "str"},
            {"field": "vendor_name",  "blocking": False, "type": "str"},
            {"field": "lab_name",     "blocking": False, "type": "str"},
            {"field": "report_date",  "blocking": False, "type": "str"},
            {"field": "tests",        "blocking": True,  "type": "list"},
        ],
    },
    "bill": {
        "label": "Hospital Bill",
        "fields": [
            {"field": "patient_id",      "blocking": True,  "type": "str"},
            {"field": "hospital_name",   "blocking": False, "type": "str"},
            {"field": "billing_date",    "blocking": False, "type": "str"},
            {"field": "line_items",      "blocking": False, "type": "list"},
            {"field": "total_amount",    "blocking": True,  "type": "float"},
            {"field": "payment_status",  "blocking": True,  "type": "str"},
        ],
    },
    "prescription": {
        "label": "Prescription (per medication row)",
        "fields": [
            {"field": "sl_no",           "blocking": False, "type": "int"},
            {"field": "medicine_name",   "blocking": True,  "type": "str"},
            {"field": "strength",        "blocking": True,  "type": "str"},
            {"field": "dosage",          "blocking": False, "type": "str"},
            {"field": "frequency",       "blocking": True,  "type": "str"},
            {"field": "route",           "blocking": True,  "type": "str"},
            {"field": "period",          "blocking": False, "type": "str"},
            {"field": "remarks",         "blocking": False, "type": "str"},
            {"field": "total_quantity",  "blocking": False, "type": "str"},
        ],
    },
}

#  Human-readable prompts used when the Rules Engine tool builds an MCP
#  Elicitation schema for a missing field.
FIELD_PROMPTS: dict[str, str] = {
    "age": "Patient age in years",
    "gender": "Patient gender as recorded (Male / Female / Other)",
    "address": "Patient home address for the discharge letter",
    "admission_date": "Admission date (YYYY-MM-DD)",
    "discharge_date": "Discharge date (YYYY-MM-DD)",
    "ward": "Ward identifier (e.g. 3B)",
    "bed_no": "Bed number (e.g. 14)",
    "doctors": "Attending physician and consulting doctors",
    "adr_allergy_info": "Known allergies / adverse drug reactions ('NKDA' if none)",
    "follow_up_appointments": "Follow-up appointment (speciality, date, clinician)",
    "discharge_instructions": "Discharge instructions given to the patient",
    "discharge_approved_by": "Treating physician who approved the discharge",
    "vendor_name": "Laboratory vendor name",
    "lab_name": "Performing laboratory name",
    "report_date": "Lab report date (YYYY-MM-DD)",
    "hospital_name": "Billing hospital name",
    "billing_date": "Bill issue date (YYYY-MM-DD)",
    "line_items": "Billed line items",
    "dosage": "Dosage per administration (e.g. 1 tablet)",
    "period": "Duration of therapy (e.g. 30 days)",
    "remarks": "Prescription remarks / counselling notes",
    "total_quantity": "Total quantity dispensed",
    "sl_no": "Prescription row number",
}


# --------------------------------------------------------------------------- #
#  Cross-validation catalogue  (specification Table 4)
# --------------------------------------------------------------------------- #
CROSS_VALIDATION_RULES: list[dict[str, Any]] = [
    {
        "rule_id": "med_omission_check",
        "severity": "Warning",
        "description": "Discharge meds differ from EHR medication history",
        "action": "Flag for review",
        "blocks_discharge": False,
        "risk_key": "medication_omission",
    },
    {
        "rule_id": "allergy_contradiction_check",
        "severity": "Critical",
        "description": "Prescribed med conflicts with known allergy",
        "action": "Block discharge",
        "blocks_discharge": True,
        "risk_key": "allergy_contradiction",
    },
    {
        "rule_id": "diagnosis_mismatch_check",
        "severity": "Warning",
        "description": "Discharge diagnosis differs from EHR care plan",
        "action": "Flag for review",
        "blocks_discharge": False,
        "risk_key": "diagnosis_mismatch",
    },
    {
        "rule_id": "follow_up_missing_check",
        "severity": "Critical",
        "description": "Follow-up not documented despite care plan requirement",
        "action": "Block discharge",
        "blocks_discharge": True,
        "risk_key": "followup_missing",
    },
    {
        "rule_id": "lab_follow_up_mismatch_check",
        "severity": "Warning",
        "description": "Abnormal lab values have no documented action",
        "action": "Flag for review",
        "blocks_discharge": False,
        "risk_key": "abnormal_lab_unresolved",
    },
    {
        "rule_id": "discharge_approval_check",
        "severity": "Critical",
        "description": "Discharge not approved by treating physician",
        "action": "Block discharge",
        "blocks_discharge": True,
        "risk_key": "missing_mandatory_field",
    },
    {
        "rule_id": "bill_settlement_check",
        "severity": "Critical",
        "description": "Bill not PAID or lacking insurance guarantee letter",
        "action": "Block discharge",
        "blocks_discharge": True,
        "risk_key": "bill_unpaid_with_discharge_ok",
    },
]

CROSS_VALIDATION_BY_ID: dict[str, dict[str, Any]] = {
    rule["rule_id"]: rule for rule in CROSS_VALIDATION_RULES
}

#  Extra rules that are not in Table 4 but are demanded by rules.yaml §3/§4.
SUPPLEMENTARY_RULES: dict[str, dict[str, Any]] = {
    "medication_added_check": {
        "rule_id": "medication_added_check",
        "severity": "Warning",
        "description": "Discharge prescribes a medication with no EHR order",
        "action": "Flag for review",
        "blocks_discharge": False,
        "risk_key": "medication_added",
    },
    "high_risk_med_check": {
        "rule_id": "high_risk_med_check",
        "severity": "Critical",
        "description": "High-risk medication added without an EHR order",
        "action": "Block discharge — page pharmacist",
        "blocks_discharge": True,
        "risk_key": "high_risk_med_missing_in_ehr",
    },
    "high_risk_med_counseling_check": {
        "rule_id": "high_risk_med_counseling_check",
        "severity": "Warning",
        "description": "High-risk medication dispensed without documented counselling",
        "action": "Flag for review",
        "blocks_discharge": False,
        "risk_key": "high_risk_med_no_counseling",
    },
    "translation_confidence_check": {
        "rule_id": "translation_confidence_check",
        "severity": "Warning",
        "description": "Translation confidence below the configured minimum",
        "action": "Flag for review — verify translated content",
        "blocks_discharge": False,
        "risk_key": "low_translation_confidence",
    },
    "service_line_hitl_check": {
        "rule_id": "service_line_hitl_check",
        "severity": "Warning",
        "description": "Service line always requires human review",
        "action": "Mandatory HITL review",
        "blocks_discharge": False,
        "risk_key": None,
    },
    "prescription_completeness_check": {
        "rule_id": "prescription_completeness_check",
        "severity": "Critical",
        "description": "Prescription row is missing a mandatory column",
        "action": "Block discharge — correct the prescription",
        "blocks_discharge": True,
        "risk_key": "incomplete_prescription_fields",
    },
}

ALL_RULES_BY_ID: dict[str, dict[str, Any]] = {
    **CROSS_VALIDATION_BY_ID,
    **SUPPLEMENTARY_RULES,
}


# --------------------------------------------------------------------------- #
#  MCP Resource payload builders
# --------------------------------------------------------------------------- #
def completeness_resource_text() -> str:
    """Body of `resource://clinical-rules/completeness`."""
    rules = load_rules()
    payload = {
        "rules_version": rules_version(),
        "source": "configs/rules.yaml + specification Table 3",
        "documents": COMPLETENESS_RULES,
        "mandatory_clinical_fields": rules.get("mandatory_clinical_fields", []),
        "mandatory_prescription_fields": rules.get("mandatory_prescription_fields", []),
        "field_prompts": FIELD_PROMPTS,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def cross_validation_resource_text() -> str:
    """Body of `resource://clinical-rules/cross-validation`."""
    rules = load_rules()
    payload = {
        "rules_version": rules_version(),
        "source": "configs/rules.yaml + specification Table 4",
        "rules": CROSS_VALIDATION_RULES,
        "supplementary_rules": list(SUPPLEMENTARY_RULES.values()),
        "clinical_validation_policies": rules.get("clinical_validation_policies", {}),
        "risk_scoring_matrix": rules.get("risk_scoring_matrix", {}),
        "quality_thresholds": rules.get("quality_thresholds", {}),
        "business_rules": rules.get("business_rules", {}),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
