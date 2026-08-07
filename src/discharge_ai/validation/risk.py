"""
validation/risk.py
==================

Composite discharge risk scoring, driven entirely by the
`risk_scoring_matrix` section of `configs/rules.yaml`.

    score      = Σ weight(finding)  +  Σ weight(missing mandatory field)
    level      = Low   if score <= thresholds.low_max      (auto-approve)
                 Medium if score <= thresholds.medium_max  (standard HITL)
                 High   otherwise                           (escalate / block)
    recommend  = reporting.recommendations[level]

Two overrides sit on top of the arithmetic:

* a **hard guardrail** in `hitl_hard_guardrails` (allergy contradiction,
  high-risk med not in the EHR, incomplete prescription rows, low translation
  confidence, always-HITL service lines) forces High and HITL regardless of the
  total, and
* any finding whose `blocks_discharge` flag is set blocks release.

Demographic gaps get their own softer weights (`missing_address`,
`missing_gender` = 1) so a purely cosmetic data-entry omission does not push an
otherwise clean discharge out of the auto-approve band.
"""

from __future__ import annotations

from typing import Any

from ..common.rules import (
    hard_guardrails,
    recommendation_text,
    risk_thresholds,
    risk_weights,
)
from ..common.schemas import (
    CompletenessResult,
    Recommendation,
    RiskAssessment,
    RiskLevel,
    Severity,
    ValidationFinding,
)

#  Missing-field name → dedicated weight key in rules.yaml (default is
#  `missing_mandatory_field`).
_SOFT_FIELD_WEIGHTS = {
    "address": "missing_address",
    "gender": "missing_gender",
}

#  Findings whose risk_key is not a rules.yaml weight are mapped here.
_RULE_TO_GUARDRAIL = {
    "allergy_contradiction_check": "allergy_contradiction",
    "high_risk_med_check": "high_risk_med_missing_in_ehr",
    "prescription_completeness_check": "incomplete_prescription_fields",
}


def score_case(
    findings: list[ValidationFinding],
    completeness: CompletenessResult,
    *,
    rag_unsafe: bool = False,
) -> RiskAssessment:
    """Compute score, tier, recommendation and blocking status."""
    weights = risk_weights()
    thresholds = risk_thresholds()
    guardrail_catalogue = set(hard_guardrails())

    assessment = RiskAssessment()
    total = 0

    # ---- 1. cross-validation findings --------------------------------------
    for finding in findings:
        if finding.severity == Severity.INFO:
            continue

        key = finding.risk_key
        weight = int(weights.get(key, 0)) if key else 0
        if weight:
            total += weight
            assessment.contributions.append(
                {
                    "source": "finding",
                    "rule_id": finding.rule_id,
                    "risk_key": key,
                    "weight": weight,
                    "severity": finding.severity.value,
                    "message": finding.message[:180],
                }
            )

        if finding.blocks_discharge:
            assessment.discharge_blocked = True

        guardrail = _RULE_TO_GUARDRAIL.get(finding.rule_id) or finding.details.get(
            "hard_guardrail"
        )
        if guardrail and guardrail in guardrail_catalogue:
            if guardrail not in assessment.hard_guardrails_hit:
                assessment.hard_guardrails_hit.append(guardrail)

    # ---- 2. missing mandatory fields ---------------------------------------
    for missing in completeness.missing_fields:
        weight_key = (
            "incomplete_prescription_fields"
            if missing.document == "prescription"
            else _SOFT_FIELD_WEIGHTS.get(missing.field, "missing_mandatory_field")
        )
        #  Blocking prescription columns already scored via the
        #  prescription_completeness_check finding — don't double-count them.
        if missing.document == "prescription" and missing.blocking:
            continue

        weight = int(weights.get(weight_key, 0))
        if not weight:
            continue
        #  Non-blocking prescription columns are cosmetic; charge them softly.
        if missing.document == "prescription":
            weight = 1

        total += weight
        assessment.contributions.append(
            {
                "source": "missing_field",
                "field": f"{missing.document}.{missing.field}"
                + (f"[row {missing.row}]" if missing.row else ""),
                "risk_key": weight_key,
                "weight": weight,
                "blocking": missing.blocking,
            }
        )

    if completeness.has_blocking:
        assessment.discharge_blocked = True

    # ---- 3. RAG safety guardrail -------------------------------------------
    if rag_unsafe and "rag_unsafe_response" in guardrail_catalogue:
        assessment.hard_guardrails_hit.append("rag_unsafe_response")

    # ---- 4. tier + recommendation ------------------------------------------
    assessment.score = total

    if total <= thresholds["low_max"]:
        assessment.level = RiskLevel.LOW
    elif total <= thresholds["medium_max"]:
        assessment.level = RiskLevel.MEDIUM
    else:
        assessment.level = RiskLevel.HIGH

    #  Hard guardrails always escalate to the High tier.
    if assessment.hard_guardrails_hit:
        assessment.level = RiskLevel.HIGH
    #  A blocked discharge can never be Low risk.
    elif assessment.discharge_blocked and assessment.level == RiskLevel.LOW:
        assessment.level = RiskLevel.MEDIUM

    assessment.recommendation = {
        RiskLevel.LOW: Recommendation.APPROVE,
        RiskLevel.MEDIUM: Recommendation.EDIT,
        RiskLevel.HIGH: Recommendation.REJECT,
    }[assessment.level]
    assessment.recommendation_text = recommendation_text(assessment.level.value)

    assessment.hitl_required = (
        assessment.level != RiskLevel.LOW
        or assessment.discharge_blocked
        or bool(assessment.hard_guardrails_hit)
    )
    return assessment


def explain(assessment: RiskAssessment) -> dict[str, Any]:
    """Human-readable breakdown for the dashboard and the audit report."""
    thresholds = risk_thresholds()
    return {
        "score": assessment.score,
        "level": assessment.level.value,
        "thresholds": thresholds,
        "recommendation": assessment.recommendation.value,
        "recommendation_text": assessment.recommendation_text,
        "discharge_blocked": assessment.discharge_blocked,
        "hitl_required": assessment.hitl_required,
        "hard_guardrails_hit": assessment.hard_guardrails_hit,
        "top_contributors": sorted(
            assessment.contributions, key=lambda c: -int(c.get("weight", 0))
        )[:8],
    }
