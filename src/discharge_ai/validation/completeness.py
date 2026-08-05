"""
validation/completeness.py
==========================

Clinical completeness validation (specification §2.4.1, Table 3).

Every document type has a mandatory-field list; a subset are **blocking**.  If a
blocking field is missing, auto-generation of the discharge summary is stopped
and HITL must intervene.  Non-blocking gaps are exactly what the MCP
**Elicitation** flow tries to collect from the human reviewer first.

The rule catalogue itself is fetched from
`resource://clinical-rules/completeness` at runtime — this module owns the
mapping from those rule names onto `ExtractedCase` attributes, since the spec's
field names ("doctors", "adr_allergy_info", "tests") are document-level concepts
rather than model attributes.
"""

from __future__ import annotations

from typing import Any, Callable

from ..common.rules import COMPLETENESS_RULES, FIELD_PROMPTS
from ..common.schemas import (
    CompletenessResult,
    ExtractedCase,
    MissingField,
)


# --------------------------------------------------------------------------- #
#  Rule field  ->  how to read it off an ExtractedCase
# --------------------------------------------------------------------------- #
def _has_text(value: Any) -> bool:
    return bool(value is not None and str(value).strip())


#  Each accessor returns the value that the rule field refers to.
_ACCESSORS: dict[str, dict[str, Callable[[ExtractedCase], Any]]] = {
    "discharge_report": {
        "patient_id": lambda c: c.patient_id,
        "patient_name": lambda c: c.patient_name,
        "age": lambda c: c.age,
        "gender": lambda c: c.gender or c.sex,
        "address": lambda c: c.address,
        "admission_date": lambda c: c.admission_date,
        "discharge_date": lambda c: c.discharge_date,
        "ward": lambda c: c.ward,
        "bed_no": lambda c: c.bed_no,
        #  "doctors" = the treating team: attending and/or consulting clinicians.
        "doctors": lambda c: c.attending_physician or (
            ", ".join(c.consulting_doctors) if c.consulting_doctors else None
        ),
        "discharge_diagnosis": lambda c: c.discharge_diagnosis,
        "medications": lambda c: c.medications,
        #  "adr_allergy_info" is present when allergies are *documented*, and an
        #  explicit "NKDA" counts as documentation.
        "adr_allergy_info": lambda c: c.allergies,
        "follow_up_appointments": lambda c: c.follow_up_appointment,
        "discharge_instructions": lambda c: c.discharge_instructions,
        "discharge_approved_by": lambda c: c.discharge_approved_by,
        "discharge_approved": lambda c: c.discharge_approved,
    },
    "lab_report": {
        "patient_id": lambda c: c.patient_id,
        "vendor_name": lambda c: c.lab_vendor_name or c.lab_name,
        "lab_name": lambda c: c.lab_name,
        "report_date": lambda c: c.lab_report_date,
        "tests": lambda c: c.lab_tests,
    },
    "bill": {
        "patient_id": lambda c: c.bill.patient_id or c.patient_id,
        "hospital_name": lambda c: c.bill.hospital_name,
        "billing_date": lambda c: c.bill.billing_date,
        "line_items": lambda c: c.bill.line_items,
        "total_amount": lambda c: c.bill.total_amount,
        "payment_status": lambda c: c.bill.payment_status,
    },
}


def _is_present(value: Any, field_type: str) -> bool:
    if value is None:
        return False
    if field_type == "bool":
        return isinstance(value, bool)      # False is a *present* value, not a gap
    if field_type == "list":
        return bool(value)
    if field_type in {"int", "float"}:
        return not isinstance(value, str) or _has_text(value)
    return _has_text(value)


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def check_completeness(
    case: ExtractedCase, *, documents_present: list[str] | None = None
) -> CompletenessResult:
    """Validate `case` against the Table 3 mandatory-field catalogue."""
    present_docs = set(documents_present or case.doc_types_present or [])
    result = CompletenessResult()

    for doc_type, spec in COMPLETENESS_RULES.items():
        if doc_type == "prescription":
            continue
        #  Only score documents the hospital actually sent us; a patient with no
        #  lab report should not be penalised for missing lab fields, but the
        #  absence is recorded as a note by the extractor.
        if present_docs and doc_type not in present_docs:
            continue

        accessors = _ACCESSORS.get(doc_type, {})
        for rule in spec["fields"]:
            field_name = rule["field"]
            field_type = rule.get("type", "str")
            blocking = bool(rule.get("blocking"))

            result.total_required += 1
            accessor = accessors.get(field_name)
            value = accessor(case) if accessor else None

            if _is_present(value, field_type):
                result.present += 1
                continue

            missing = MissingField(
                field=field_name,
                document=doc_type,
                blocking=blocking,
                field_type=field_type,     # type: ignore[arg-type]
                prompt=FIELD_PROMPTS.get(field_name, f"Value for {field_name}"),
            )
            result.missing_fields.append(missing)
            if blocking:
                result.blocking_missing.append(f"{doc_type}.{field_name}")
            else:
                result.non_blocking_missing.append(f"{doc_type}.{field_name}")

    # ---- prescription rows -------------------------------------------------
    prescription_fields = COMPLETENESS_RULES["prescription"]["fields"]
    for index, medication in enumerate(case.medications, start=1):
        row_gaps: list[str] = []
        for rule in prescription_fields:
            field_name = rule["field"]
            result.total_required += 1
            value = getattr(medication, field_name, None)

            if _is_present(value, rule.get("type", "str")):
                result.present += 1
                continue

            row_gaps.append(field_name)
            result.missing_fields.append(
                MissingField(
                    field=field_name,
                    document="prescription",
                    blocking=bool(rule.get("blocking")),
                    field_type=rule.get("type", "str"),   # type: ignore[arg-type]
                    prompt=FIELD_PROMPTS.get(field_name, f"Value for {field_name}"),
                    row=medication.sl_no or index,
                )
            )
            label = f"prescription[{medication.sl_no or index}].{field_name}"
            if rule.get("blocking"):
                result.blocking_missing.append(label)
            else:
                result.non_blocking_missing.append(label)

        if row_gaps:
            result.prescription_gaps.append(
                {
                    "row": medication.sl_no or index,
                    "medicine_name": medication.medicine_name,
                    "missing": row_gaps,
                }
            )

    result.score = (
        round(100.0 * result.present / result.total_required, 1)
        if result.total_required else 100.0
    )
    return result


# --------------------------------------------------------------------------- #
#  MCP Elicitation schema
# --------------------------------------------------------------------------- #
_PY_TYPES: dict[str, type] = {
    "str": str, "int": int, "float": float, "bool": bool, "list": str,
}


def elicitation_schema_for(missing: list[MissingField]) -> tuple[type, dict[str, Any]]:
    """Build the Pydantic schema handed to `ctx.elicit()`.

    MCP elicitation schemas must be flat objects of primitive types, so list
    fields are collected as comma-separated strings and split by the caller.

    Returns `(model_class, json_schema)` — the JSON Schema is what the HITL
    dashboard renders as a dynamic form, and it is recorded in the audit trail.
    """
    from pydantic import Field, create_model

    definitions: dict[str, Any] = {}
    seen: set[str] = set()

    for field in missing:
        #  Prescription gaps are per-row, so the key carries the row number.
        key = (
            f"{field.field}_row{field.row}"
            if field.document == "prescription" and field.row
            else field.field
        )
        if key in seen:
            continue
        seen.add(key)

        python_type = _PY_TYPES.get(field.field_type, str)
        description = field.prompt or f"Value for {field.field}"
        if field.document == "prescription" and field.row:
            description = f"{description} (prescription row {field.row})"
        if field.field_type == "list":
            description += " — separate multiple entries with commas"

        #  Optional with a None default: a reviewer may legitimately not know a
        #  value, and MCP `accept` with a blank field must not fail validation.
        definitions[key] = (
            python_type | None,
            Field(default=None, description=description),
        )

    model = create_model("MissingClinicalFields", **definitions)  # type: ignore[call-overload]
    return model, model.model_json_schema()
