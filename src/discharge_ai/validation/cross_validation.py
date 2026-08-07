"""
validation/cross_validation.py
==============================

Cross-validation against the Mock EHR, care plan and labs
(specification §2.4.2, Table 4).

    med_omission_check            Warning   discharge meds differ from EHR history
    allergy_contradiction_check   Critical  prescribed med conflicts with allergy
    diagnosis_mismatch_check      Warning   dx differs from EHR care plan
    follow_up_missing_check       Critical  follow-up absent despite care plan
    lab_follow_up_mismatch_check  Warning   abnormal labs with no documented action
    discharge_approval_check      Critical  not approved by treating physician
    bill_settlement_check         Critical  bill not PAID / no guarantee letter

Plus the supplementary rules rules.yaml demands (medication_added,
high_risk_med_missing_in_ehr, counselling, always-HITL service lines,
incomplete prescription rows).

Translation confidence is metadata from the Normalizer Agent — not a validation
rule. It is displayed on the report but never creates a Finding or a rule_id.

Drug comparison runs on canonicalised INN names, so the Spanish "Metformina"
reconciles against the EHR's "Metformin" and the Dutch "Amoxicilline" still
collides with a "Penicillin" allergy.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..common.rules import (
    ALL_RULES_BY_ID,
    always_hitl_service_lines,
    high_risk_medications,
)
from ..common.schemas import ExtractedCase, Severity, ValidationFinding
from ..common.terminology import (
    allergy_conflicts,
    canonical_drug,
    is_no_known_allergy,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Finding factory
# --------------------------------------------------------------------------- #
def _finding(rule_id: str, message: str, **details: Any) -> ValidationFinding:
    rule = ALL_RULES_BY_ID.get(rule_id, {})
    return ValidationFinding(
        rule_id=rule_id,
        severity=Severity(rule.get("severity", "Warning")),
        message=message,
        action=rule.get("action", "Flag for review"),
        blocks_discharge=bool(rule.get("blocks_discharge")),
        risk_key=rule.get("risk_key"),
        details=details,
    )


# --------------------------------------------------------------------------- #
#  Individual checks
# --------------------------------------------------------------------------- #
def _check_medication_reconciliation(
    case: ExtractedCase, ehr: dict[str, Any]
) -> list[ValidationFinding]:
    """Omissions, additions and high-risk additions vs. the EHR order set."""
    findings: list[ValidationFinding] = []

    ehr_orders = ehr.get("medications", []) or []
    ehr_map = {
        canonical_drug(order.get("name")): order
        for order in ehr_orders
        if canonical_drug(order.get("name"))
    }
    discharge_map = {
        canonical_drug(med.medicine_name): med
        for med in case.medications
        if canonical_drug(med.medicine_name)
    }

    high_risk = {canonical_drug(name) for name in high_risk_medications()}

    # ---- omissions: ordered in the EHR but absent at discharge -------------
    omitted = [key for key in ehr_map if key not in discharge_map]
    for key in omitted:
        order = ehr_map[key]
        findings.append(
            _finding(
                "med_omission_check",
                f"EHR medication '{order.get('name')}' "
                f"({order.get('dose', '')} {order.get('frequency', '')})".strip()
                + " is not on the discharge prescription",
                ehr_medication=order.get("name"),
                ehr_dose=order.get("dose"),
                ehr_frequency=order.get("frequency"),
                canonical=key,
            )
        )

    # ---- additions: prescribed at discharge with no EHR order --------------
    for key, medication in discharge_map.items():
        if key in ehr_map:
            continue
        if key in high_risk:
            findings.append(
                _finding(
                    "high_risk_med_check",
                    f"High-risk medication '{medication.medicine_name}' "
                    f"{medication.strength or ''}".strip()
                    + " is prescribed at discharge with no corresponding EHR order",
                    medication=medication.medicine_name,
                    strength=medication.strength,
                    canonical=key,
                )
            )
        else:
            findings.append(
                _finding(
                    "medication_added_check",
                    f"Discharge medication '{medication.medicine_name}' "
                    "has no corresponding order in the EHR medication history",
                    medication=medication.medicine_name,
                    strength=medication.strength,
                    canonical=key,
                )
            )

    # ---- high-risk counselling documentation -------------------------------
    counselling_text = " ".join(
        filter(
            None,
            [case.discharge_instructions or ""]
            + [med.remarks or "" for med in case.medications],
        )
    ).lower()
    counselling_documented = any(
        marker in counselling_text
        for marker in ("counsel", "counsell", "education", "instructed on", "teach-back")
    )
    for key, medication in discharge_map.items():
        if key in high_risk and not counselling_documented:
            findings.append(
                _finding(
                    "high_risk_med_counseling_check",
                    f"High-risk medication '{medication.medicine_name}' dispensed "
                    "without documented pharmacist counselling",
                    medication=medication.medicine_name,
                )
            )

    return findings


def _check_allergies(case: ExtractedCase, ehr: dict[str, Any]) -> list[ValidationFinding]:
    """Allergy registry (EHR is authoritative) vs. prescribed medications."""
    findings: list[ValidationFinding] = []

    registry = list(ehr.get("allergies", []) or [])
    documented = [a for a in case.allergies if not is_no_known_allergy(a)]
    #  The EHR registry is the immutable source of truth; the discharge note's
    #  own allergy list is additional evidence, never a replacement.
    all_allergies = list(dict.fromkeys(registry + documented))

    medications = [m.medicine_name or "" for m in case.medications]

    #  The registry and the note usually document the same allergy with
    #  different spellings ("Penicillin" vs "Penicilline (documented rash)"),
    #  so collapse on (allergy class, canonical drug) — one clinical conflict
    #  must produce exactly one finding, or it would be risk-scored twice.
    seen_conflicts: set[tuple[str, str]] = set()
    for conflict in allergy_conflicts(all_allergies, medications):
        key = (conflict["allergy_class"], conflict["medication_canonical"])
        if key in seen_conflicts:
            continue
        seen_conflicts.add(key)

        findings.append(
            _finding(
                "allergy_contradiction_check",
                f"Prescribed medication '{conflict['medication']}' conflicts with "
                f"documented allergy '{conflict['allergy']}' "
                f"({conflict['allergy_class']} class)",
                **conflict,
                ehr_registry=registry,
            )
        )

    #  A documented allergy that the EHR registry does not know about is a
    #  reconciliation gap worth surfacing (informational, non-blocking).
    from ..common.terminology import canonical_allergy

    registry_keys = {canonical_allergy(a) for a in registry if canonical_allergy(a)}
    for allergy in documented:
        key = canonical_allergy(allergy)
        if key and key not in registry_keys:
            findings.append(
                ValidationFinding(
                    rule_id="allergy_registry_gap",
                    severity=Severity.INFO,
                    message=(
                        f"Discharge note documents allergy '{allergy}' which is "
                        "not present in the EHR allergy registry"
                    ),
                    action="Update the EHR allergy registry",
                    blocks_discharge=False,
                    details={"allergy": allergy, "ehr_registry": registry},
                )
            )

    return findings


_ICD10_RE = re.compile(r"\b([A-TV-Z]\d{2}(?:\.\d{1,3})?)\b")


def _check_diagnosis(case: ExtractedCase, ehr: dict[str, Any]) -> list[ValidationFinding]:
    """Discharge diagnosis vs. the EHR care plan / primary diagnoses."""
    demographics = ehr.get("demographics", {}) or {}
    ehr_codes = {str(code).upper() for code in demographics.get("primary_dx", [])}
    if not ehr_codes:
        return []

    diagnosis_text = " ".join(case.discharge_diagnosis)
    documented_codes = {code.upper() for code in _ICD10_RE.findall(diagnosis_text)}

    #  Also accept a textual match against the guideline diagnosis name, since
    #  translated notes may spell the diagnosis without its ICD-10 code.
    lowered = diagnosis_text.lower()
    for guideline in ehr.get("guidelines", []) or []:
        name = str(guideline.get("diagnosis", "")).lower()
        if name and any(word in lowered for word in name.split() if len(word) > 4):
            documented_codes.add(str(guideline.get("icd10", "")).upper())

    missing = sorted(ehr_codes - documented_codes)
    if not missing:
        return []

    return [
        _finding(
            "diagnosis_mismatch_check",
            "Discharge diagnosis does not document EHR care-plan diagnosis "
            f"code(s): {', '.join(missing)}",
            ehr_primary_dx=sorted(ehr_codes),
            documented_in_discharge=sorted(documented_codes),
            missing_codes=missing,
        )
    ]


def _check_follow_up(case: ExtractedCase, ehr: dict[str, Any]) -> list[ValidationFinding]:
    """Care plan requires follow-up → the discharge note must document it."""
    care_plan = ehr.get("care_plan", {}) or {}
    if not care_plan.get("followup_required"):
        return []

    documented = (case.follow_up_appointment or "").strip()
    if documented:
        return []

    return [
        _finding(
            "follow_up_missing_check",
            "Care plan requires "
            f"{care_plan.get('speciality', 'specialist')} follow-up within "
            f"{care_plan.get('window_days', '?')} days, but no follow-up "
            "appointment is documented on the discharge note",
            required_speciality=care_plan.get("speciality"),
            window_days=care_plan.get("window_days"),
        )
    ]


def _check_abnormal_labs(case: ExtractedCase, ehr: dict[str, Any]) -> list[ValidationFinding]:
    """Every abnormal lab needs a documented action somewhere in the discharge."""
    findings: list[ValidationFinding] = []

    #  The EHR lab registry decides what counts as abnormal (its `abnormal` flag
    #  is clinically adjudicated); the incoming lab report is corroborating.
    abnormal = ehr.get("abnormal_labs", []) or []
    if not abnormal:
        return findings

    #  Anywhere an action could reasonably be documented.
    documented_blob = " ".join(
        filter(
            None,
            [
                case.discharge_instructions or "",
                case.follow_up_appointment or "",
                " ".join(f"{a.test or ''} {a.value or ''} {a.action or ''}"
                         for a in case.abnormal_labs),
                " ".join(med.remarks or "" for med in case.medications),
                " ".join(med.medicine_name or "" for med in case.medications),
            ],
        )
    ).lower()

    for lab in abnormal:
        test = str(lab.get("test", "")).strip()
        if not test:
            continue

        #  Resolved if the discharge documents an action for this test, either
        #  explicitly (our parsed abnormal_labs carry an action) or by naming
        #  the test in the instructions/follow-up.
        explicit_action = any(
            (a.action or "").strip()
            and test.lower() in f"{(a.test or '').lower()}"
            for a in case.abnormal_labs
        )
        mentioned = test.lower() in documented_blob
        if explicit_action or mentioned:
            continue

        findings.append(
            _finding(
                "lab_follow_up_mismatch_check",
                f"Abnormal lab '{test}' ({lab.get('value')}) has no documented "
                "action in the discharge report",
                test=test,
                value=lab.get("value"),
                ehr_action=lab.get("action_in_ehr"),
            )
        )

    return findings


def _check_discharge_approval(case: ExtractedCase) -> list[ValidationFinding]:
    """Discharge must be approved by the treating physician."""
    if case.discharge_approved is True:
        return []

    reason = (
        "Discharge is explicitly NOT approved"
        if case.discharge_approved is False
        else "Discharge approval status is missing from the record"
    )
    return [
        _finding(
            "discharge_approval_check",
            f"{reason} — a treating physician sign-off is mandatory before release",
            discharge_approved=case.discharge_approved,
            discharge_approved_by=case.discharge_approved_by,
        )
    ]


def _check_bill(case: ExtractedCase) -> list[ValidationFinding]:
    """Bill must be PAID or carry an insurance guarantee letter."""
    bill = case.bill
    if bill.is_settled:
        return []

    return [
        _finding(
            "bill_settlement_check",
            f"Hospital bill is {bill.payment_status or 'UNKNOWN'} "
            f"({bill.total_amount if bill.total_amount is not None else '?'} "
            f"{bill.currency or ''}".strip()
            + ") with no insurance guarantee letter on file",
            payment_status=bill.payment_status,
            total_amount=bill.total_amount,
            currency=bill.currency,
            insurance_guarantee_letter=bill.insurance_guarantee_letter,
        )
    ]


def _check_service_line(case: ExtractedCase, ehr: dict[str, Any]) -> list[ValidationFinding]:
    """rules.yaml forces HITL for paediatric / obstetric / oncology cases."""
    demographics = ehr.get("demographics", {}) or {}
    service_line = (
        case.service_line or demographics.get("service_line") or ""
    ).lower()
    if not service_line:
        return []

    findings: list[ValidationFinding] = []
    for keyword, guardrail in always_hitl_service_lines().items():
        if keyword in service_line:
            finding = _finding(
                "service_line_hitl_check",
                f"Service line '{case.service_line or demographics.get('service_line')}' "
                "always requires human review under hospital policy",
                service_line=case.service_line or demographics.get("service_line"),
                guardrail=guardrail,
            )
            finding.details["hard_guardrail"] = guardrail
            findings.append(finding)
    return findings


def _check_prescription_rows(case: ExtractedCase) -> list[ValidationFinding]:
    """A prescription row missing a mandatory column blocks discharge."""
    from ..common.rules import COMPLETENESS_RULES

    blocking_columns = [
        rule["field"]
        for rule in COMPLETENESS_RULES["prescription"]["fields"]
        if rule.get("blocking")
    ]

    findings: list[ValidationFinding] = []
    for index, medication in enumerate(case.medications, start=1):
        gaps = [
            column for column in blocking_columns
            if not str(getattr(medication, column, "") or "").strip()
        ]
        if not gaps:
            continue
        findings.append(
            _finding(
                "prescription_completeness_check",
                f"Prescription row {medication.sl_no or index} "
                f"('{medication.medicine_name or 'unnamed'}') is missing "
                f"mandatory column(s): {', '.join(gaps)}",
                row=medication.sl_no or index,
                medicine_name=medication.medicine_name,
                missing_columns=gaps,
            )
        )
    return findings


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def cross_validate(
    case: ExtractedCase, ehr_bundle: dict[str, Any]
) -> list[ValidationFinding]:
    """Run every cross-validation rule and return the findings."""
    if not ehr_bundle or ehr_bundle.get("found") is False:
        return [
            ValidationFinding(
                rule_id="ehr_record_missing",
                severity=Severity.CRITICAL,
                message=(
                    f"No EHR record found for patient {case.patient_id} — "
                    "cross-validation cannot be performed"
                ),
                action="Block discharge — verify patient identity",
                blocks_discharge=True,
                risk_key="missing_mandatory_field",
                details={"ehr_source": ehr_bundle.get("source")},
            )
        ]

    findings: list[ValidationFinding] = []
    findings += _check_medication_reconciliation(case, ehr_bundle)
    findings += _check_allergies(case, ehr_bundle)
    findings += _check_diagnosis(case, ehr_bundle)
    findings += _check_follow_up(case, ehr_bundle)
    findings += _check_abnormal_labs(case, ehr_bundle)
    findings += _check_discharge_approval(case)
    findings += _check_bill(case)
    findings += _check_service_line(case, ehr_bundle)
    findings += _check_prescription_rows(case)

    #  Critical first, then Warning, then Info — the order the dashboard shows.
    order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    findings.sort(key=lambda f: (order.get(f.severity, 3), f.rule_id))
    return findings
