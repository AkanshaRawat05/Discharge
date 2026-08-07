"""
dashboard/resolutions.py
========================

Reviewer resolution controls for blocking validation findings.

The Table 4 cross-validation rules are surfaced on **2 · Validation Report** as
read-only findings.  This module turns each *blocking* finding into an
actionable control on **3 · HITL Corrections**, so a reviewer can fix the
underlying record rather than only overriding the block and signing off.

Nothing here is keyed on a rule id.  A finding is matched to a resolver by the
shape of its own `details` payload — the same payload `cross_validation._finding()`
already builds — so a new rule that carries a `medication` or a `payment_status`
detail gets a working control without touching this file.

The resolutions a reviewer chooses are collected into session state and applied
to the working case by `apply_resolutions()`, which the caller runs immediately
before the **existing** "Re-run validation" pipeline call.  There is no second
pipeline and no new backend path: a resolution is just a case mutation, exactly
like a medication-table edit.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from discharge_ai.common.schemas import ExtractedCase

#  Payment states the Bill schema treats as settled (see `Bill.is_settled`).
SETTLED_STATUS = "PAID"


# --------------------------------------------------------------------------- #
#  Resolver registry
# --------------------------------------------------------------------------- #
#  Each resolver declares:
#    kind    — stable id used as the session-state key
#    detect  — does this finding's `details` payload look like my case?
#    render  — draw the control, return the reviewer's choice (or None)
#    apply   — mutate the ExtractedCase from that choice
#
#  `detect` inspects details only, never rule_id, so the mapping stays generic.
Resolver = dict[str, Any]


def _detect_medication(details: dict[str, Any]) -> bool:
    return bool(details.get("medication") or details.get("medication_canonical"))


def _detect_bill(details: dict[str, Any]) -> bool:
    return "payment_status" in details or "total_amount" in details


def _detect_approval(details: dict[str, Any]) -> bool:
    return "discharge_approved" in details or "discharge_approved_by" in details


def _detect_follow_up(details: dict[str, Any]) -> bool:
    return "required_speciality" in details or "window_days" in details


# --------------------------------------------------------------------------- #
#  Render helpers — one per resolution kind
# --------------------------------------------------------------------------- #
def _render_medication(finding: Any, case: ExtractedCase, key: str) -> dict | None:
    """Remove the offending medicine, swap it for an alternative, or keep it."""
    details = finding.details or {}
    offending = details.get("medication") or details.get("medication_canonical") or ""
    allergy = details.get("allergy") or details.get("allergy_class")

    st.markdown(
        f"**{offending}**"
        + (f" conflicts with the documented allergy **{allergy}**." if allergy else "")
    )

    choice = st.radio(
        "Resolution",
        ["Take no action", "Remove this medicine", "Replace with an alternative"],
        key=f"{key}-choice",
        horizontal=True,
    )

    if choice == "Remove this medicine":
        return {"action": "remove", "medication": offending}

    if choice == "Replace with an alternative":
        replacement = st.text_input(
            "Replacement medicine",
            key=f"{key}-replacement",
            placeholder="e.g. Azithromycin",
            help="The prescription row keeps its dose and duration; only the "
                 "medicine name changes.",
        )
        if replacement.strip():
            return {
                "action": "replace",
                "medication": offending,
                "replacement": replacement.strip(),
            }
        st.caption("Enter a replacement medicine to enable this resolution.")

    return None


def _render_bill(finding: Any, case: ExtractedCase, key: str) -> dict | None:
    """Mark the bill settled, or record an insurance guarantee letter."""
    bill = case.bill
    amount = bill.total_amount
    st.markdown(
        f"Outstanding balance: **{amount if amount is not None else '?'} "
        f"{bill.currency or ''}** · current status **{bill.payment_status or 'UNKNOWN'}**"
    )

    choice = st.radio(
        "Resolution",
        ["Take no action", "Mark bill as paid", "Insurance guarantee letter received"],
        key=f"{key}-choice",
        horizontal=True,
    )

    if choice == "Mark bill as paid":
        return {"action": "mark_paid"}
    if choice == "Insurance guarantee letter received":
        return {"action": "guarantee_letter"}
    return None


def _render_approval(finding: Any, case: ExtractedCase, key: str) -> dict | None:
    """Record the treating physician's discharge approval."""
    st.markdown("Discharge has not been approved by the treating physician.")

    confirmed = st.checkbox(
        "Confirm the treating physician has approved this discharge",
        key=f"{key}-confirm",
    )
    if not confirmed:
        return None

    approver = st.text_input(
        "Approving physician",
        key=f"{key}-approver",
        placeholder="e.g. Dr. O. Hart, MD",
    )
    if approver.strip():
        return {"action": "approve", "approved_by": approver.strip()}

    st.caption("Enter the approving physician to enable this resolution.")
    return None


def _render_follow_up(finding: Any, case: ExtractedCase, key: str) -> dict | None:
    """Schedule the follow-up the care plan requires."""
    details = finding.details or {}
    speciality = details.get("required_speciality") or "specialist"
    window = details.get("window_days")

    st.markdown(
        f"The care plan requires **{speciality}** follow-up"
        + (f" within **{window} days**." if window else ".")
    )

    appointment_date = st.date_input(
        "Follow-up date", value=None, key=f"{key}-date", format="YYYY-MM-DD"
    )
    detail = st.text_input(
        "Appointment detail",
        key=f"{key}-detail",
        placeholder=f"e.g. {speciality} clinic, Dr. Patel",
    )

    if appointment_date:
        return {
            "action": "schedule_follow_up",
            "date": appointment_date.isoformat(),
            "detail": detail.strip(),
            "speciality": speciality,
        }
    return None


# --------------------------------------------------------------------------- #
#  Apply helpers — mutate the case from a resolution choice
# --------------------------------------------------------------------------- #
def _apply_medication(case: ExtractedCase, choice: dict) -> str:
    from discharge_ai.common.terminology import canonical_drug

    target = canonical_drug(choice.get("medication"))
    if not target:
        return ""

    if choice["action"] == "remove":
        before = len(case.medications)
        case.medications = [
            medication for medication in case.medications
            if canonical_drug(medication.medicine_name) != target
        ]
        removed = before - len(case.medications)
        return f"removed {removed} prescription row(s) for {choice['medication']}"

    if choice["action"] == "replace":
        replacement = choice.get("replacement") or ""
        changed = 0
        for medication in case.medications:
            if canonical_drug(medication.medicine_name) == target:
                medication.medicine_name = replacement
                changed += 1
        return f"replaced {choice['medication']} with {replacement} on {changed} row(s)"

    return ""


def _apply_bill(case: ExtractedCase, choice: dict) -> str:
    if choice["action"] == "mark_paid":
        case.bill.payment_status = SETTLED_STATUS
        return "bill marked as PAID"
    if choice["action"] == "guarantee_letter":
        case.bill.insurance_guarantee_letter = True
        return "insurance guarantee letter recorded"
    return ""


def _apply_approval(case: ExtractedCase, choice: dict) -> str:
    case.discharge_approved = True
    case.discharge_approved_by = choice.get("approved_by")
    return f"discharge approved by {case.discharge_approved_by}"


def _apply_follow_up(case: ExtractedCase, choice: dict) -> str:
    detail = choice.get("detail") or f"{choice.get('speciality', 'Specialist')} follow-up"
    case.follow_up_appointment = f"{detail} on {choice['date']}"
    return f"follow-up scheduled: {case.follow_up_appointment}"


RESOLVERS: list[Resolver] = [
    {
        "kind": "medication",
        "label": "Medication / allergy conflict",
        "detect": _detect_medication,
        "render": _render_medication,
        "apply": _apply_medication,
    },
    {
        "kind": "bill",
        "label": "Bill settlement",
        "detect": _detect_bill,
        "render": _render_bill,
        "apply": _apply_bill,
    },
    {
        "kind": "approval",
        "label": "Discharge approval",
        "detect": _detect_approval,
        "render": _render_approval,
        "apply": _apply_approval,
    },
    {
        "kind": "follow_up",
        "label": "Follow-up appointment",
        "detect": _detect_follow_up,
        "render": _render_follow_up,
        "apply": _apply_follow_up,
    },
]


def resolver_for(finding: Any) -> Resolver | None:
    """First resolver whose `detect` matches this finding's details payload."""
    details = finding.details or {}
    for resolver in RESOLVERS:
        try:
            if resolver["detect"](details):
                return resolver
        except Exception:  # noqa: BLE001 — a bad detail payload must never crash the page
            continue
    return None


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def _store(patient_id: str) -> dict[str, Any]:
    store: dict[str, Any] = st.session_state.setdefault("resolutions", {})
    return store.setdefault(patient_id, {})


def render_resolution_panel(report: Any, case: ExtractedCase, patient_id: str) -> int:
    """Draw a resolution control for every blocking finding.

    Returns the number of resolutions the reviewer has filled in.  The choices
    are held in session state until `apply_resolutions()` folds them into the
    case, so nothing changes until the reviewer re-runs validation.
    """
    blocking = [finding for finding in report.findings if finding.blocks_discharge]
    store = _store(patient_id)
    store.clear()

    if not blocking:
        st.success(
            "No blocking findings on this case — nothing to resolve here."
        )
        return 0

    st.caption(
        f"{len(blocking)} finding(s) are blocking this discharge. Resolve them "
        "here and re-run validation: the corrections are applied to the case "
        "and the rules are evaluated again, exactly like the elicitation form."
    )

    resolved = 0
    for index, finding in enumerate(blocking):
        resolver = resolver_for(finding)
        title = finding.rule_id.replace("_", " ").replace(" check", "").title()

        with st.container(border=True):
            st.markdown(f"**{title}** · `{finding.rule_id}`")
            st.caption(finding.message)

            if resolver is None:
                #  No generic control fits this finding's payload. Say so rather
                #  than pretending it is resolvable — the reviewer can still fix
                #  it through the medication table or the elicitation form.
                st.info(
                    "No in-place correction is available for this finding. Use "
                    "the medication table or the elicitation form above, or "
                    "record a reviewer override below."
                )
                continue

            key = f"resolve-{patient_id}-{finding.rule_id}-{index}"
            choice = resolver["render"](finding, case, key)
            if choice:
                store[key] = {"resolver": resolver["kind"], "choice": choice}
                resolved += 1
                st.success("Will be applied when you re-run validation.")

    return resolved


def apply_resolutions(case: ExtractedCase, patient_id: str) -> list[str]:
    """Fold every pending resolution into the working case.

    Returns a human-readable list of what was applied, for the audit note and
    the on-screen confirmation.  Called immediately before the existing re-run.
    """
    store = _store(patient_id)
    if not store:
        return []

    by_kind: dict[str, Callable[[ExtractedCase, dict], str]] = {
        resolver["kind"]: resolver["apply"] for resolver in RESOLVERS
    }

    applied: list[str] = []
    for entry in store.values():
        handler = by_kind.get(entry["resolver"])
        if handler is None:
            continue
        note = handler(case, entry["choice"])
        if note:
            applied.append(note)

    return applied
