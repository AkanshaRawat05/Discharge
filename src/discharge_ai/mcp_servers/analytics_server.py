"""
mcp_servers/analytics_server.py
===============================

**Secondary MCP Server — Analytics Server**
port 8201, streamable-HTTP, path `/analyticstools`

Its whole purpose is to prove *multi-server* MCP connectivity: agents hold live
sessions to :8200 and :8201 at the same time (specification §4, Table 8).  It
exposes the Tools primitive only.

    calculate_risk_score        composite discharge risk from the rules.yaml matrix
    get_population_benchmarks   readmission rates + peer benchmarks by diagnosis
    generate_risk_heatmap       per-domain risk heatmap for the HITL dashboard

Run:
    python -m discharge_ai.mcp_servers.analytics_server
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from ..common.rules import hard_guardrails, risk_thresholds, risk_weights, rules_version
from ..observability import tracing
from ..settings import configure_logging, settings

log = logging.getLogger(__name__)

mcp = FastMCP(
    name="analytics-server",
    instructions=(
        "Secondary Analytics MCP Server for St. Marian Regional Medical Center. "
        "Computes composite discharge risk scores, returns population "
        "readmission benchmarks, and renders risk heatmaps. Tools only."
    ),
    host=settings.service("mcp_analytics")["host"],
    port=int(settings.service("mcp_analytics")["port"]),
    streamable_http_path=settings.service("mcp_analytics")["path"],
)


# =========================================================================== #
#  Population benchmark reference data
# =========================================================================== #
#  Synthetic but internally consistent network-level statistics, keyed by
#  ICD-10.  A real deployment would read these from the hospital's analytics
#  warehouse; the shape of the payload is what matters here.
POPULATION_BENCHMARKS: dict[str, dict[str, Any]] = {
    "E11.9": {"diagnosis": "Type 2 Diabetes Mellitus",
              "readmission_30d_rate": 0.092, "network_median": 0.104,
              "avg_length_of_stay_days": 3.4, "med_error_rate": 0.031,
              "followup_attendance_rate": 0.78},
    "I10": {"diagnosis": "Essential Hypertension",
            "readmission_30d_rate": 0.061, "network_median": 0.070,
            "avg_length_of_stay_days": 2.6, "med_error_rate": 0.022,
            "followup_attendance_rate": 0.81},
    "I50.9": {"diagnosis": "Congestive Heart Failure",
              "readmission_30d_rate": 0.214, "network_median": 0.198,
              "avg_length_of_stay_days": 5.8, "med_error_rate": 0.058,
              "followup_attendance_rate": 0.69},
    "I21.9": {"diagnosis": "Acute Myocardial Infarction",
              "readmission_30d_rate": 0.156, "network_median": 0.171,
              "avg_length_of_stay_days": 6.1, "med_error_rate": 0.049,
              "followup_attendance_rate": 0.84},
    "J18.9": {"diagnosis": "Pneumonia",
              "readmission_30d_rate": 0.128, "network_median": 0.139,
              "avg_length_of_stay_days": 4.7, "med_error_rate": 0.041,
              "followup_attendance_rate": 0.73},
    "J44.9": {"diagnosis": "COPD exacerbation",
              "readmission_30d_rate": 0.223, "network_median": 0.205,
              "avg_length_of_stay_days": 5.2, "med_error_rate": 0.052,
              "followup_attendance_rate": 0.66},
    "J45.909": {"diagnosis": "Asthma exacerbation",
                "readmission_30d_rate": 0.084, "network_median": 0.091,
                "avg_length_of_stay_days": 2.9, "med_error_rate": 0.027,
                "followup_attendance_rate": 0.75},
    "J20.9": {"diagnosis": "Acute bronchitis",
              "readmission_30d_rate": 0.047, "network_median": 0.055,
              "avg_length_of_stay_days": 2.1, "med_error_rate": 0.018,
              "followup_attendance_rate": 0.71},
    "N39.0": {"diagnosis": "Urinary tract infection",
              "readmission_30d_rate": 0.073, "network_median": 0.081,
              "avg_length_of_stay_days": 2.4, "med_error_rate": 0.024,
              "followup_attendance_rate": 0.77},
    "N10": {"diagnosis": "Acute pyelonephritis",
            "readmission_30d_rate": 0.101, "network_median": 0.110,
            "avg_length_of_stay_days": 3.8, "med_error_rate": 0.033,
            "followup_attendance_rate": 0.74},
    "K35.80": {"diagnosis": "Acute appendicitis",
               "readmission_30d_rate": 0.038, "network_median": 0.045,
               "avg_length_of_stay_days": 2.2, "med_error_rate": 0.015,
               "followup_attendance_rate": 0.88},
    "K52.9": {"diagnosis": "Acute gastroenteritis",
              "readmission_30d_rate": 0.056, "network_median": 0.062,
              "avg_length_of_stay_days": 1.9, "med_error_rate": 0.020,
              "followup_attendance_rate": 0.70},
    "A01.0": {"diagnosis": "Typhoid (enteric) fever",
              "readmission_30d_rate": 0.067, "network_median": 0.074,
              "avg_length_of_stay_days": 4.1, "med_error_rate": 0.029,
              "followup_attendance_rate": 0.72},
}

NETWORK_AVERAGES = {
    "readmission_30d_rate": 0.113,
    "avg_length_of_stay_days": 3.9,
    "med_error_rate": 0.034,
    "followup_attendance_rate": 0.755,
    "auto_approval_rate": 0.41,
    "hitl_escalation_rate": 0.59,
}

#  Which findings/gaps roll up into which clinical risk domain.
RISK_DOMAINS: dict[str, tuple[str, ...]] = {
    "medication_safety": (
        "medication_omission", "medication_added", "high_risk_med_missing_in_ehr",
        "high_risk_med_no_counseling", "incomplete_prescription_fields",
    ),
    "allergy_safety": ("allergy_contradiction",),
    "diagnosis_alignment": ("diagnosis_mismatch",),
    "care_continuity": ("followup_missing", "abnormal_lab_unresolved"),
    "documentation_quality": (
        "missing_mandatory_field", "missing_address", "missing_gender",
    ),
    "financial_clearance": ("bill_unpaid_with_discharge_ok",),
}


# =========================================================================== #
#  TOOL 1 — calculate_risk_score
# =========================================================================== #
@mcp.tool(
    name="calculate_risk_score",
    description=(
        "Compute the composite discharge risk score from weighted risk keys "
        "using the rules.yaml risk matrix. Returns score, tier, band "
        "thresholds, per-domain contributions and hard-guardrail hits."
    ),
)
async def calculate_risk_score(
    ctx: Context,
    patient_id: str,
    risk_keys: list[str] | None = None,
    missing_field_count: int = 0,
    translation_confidence: float = 1.0,   # accepted but not scored (metadata only)
    discharge_blocked: bool = False,
) -> dict[str, Any]:
    """Composite risk score for one discharge case.

    Args:
        patient_id:             patient under review.
        risk_keys:              rules.yaml weight keys raised by validation,
                                e.g. ["allergy_contradiction", "followup_missing"].
        missing_field_count:    count of non-blocking missing mandatory fields.
        translation_confidence: 0..1 confidence from the Normalizer Agent.
                                Metadata only — never a risk contributor per PDF §2.3.
        discharge_blocked:      whether any Critical rule already blocks release.
    """
    with tracing.tool_span(
        "calculate_risk_score", params={"patient_id": patient_id, "risk_keys": risk_keys},
        mcp_server="analytics",
    ) as span:
        weights = risk_weights()
        thresholds = risk_thresholds()
        guardrail_catalogue = set(hard_guardrails())

        keys = list(risk_keys or [])
        breakdown: list[dict[str, Any]] = []
        total = 0

        for key in keys:
            weight = int(weights.get(key, 0))
            total += weight
            breakdown.append(
                {"risk_key": key, "weight": weight,
                 "known_key": key in weights,
                 "domain": _domain_for(key)}
            )

        if missing_field_count > 0:
            weight = int(weights.get("missing_mandatory_field", 3)) * missing_field_count
            total += weight
            breakdown.append(
                {"risk_key": "missing_mandatory_field", "weight": weight,
                 "count": missing_field_count, "known_key": True,
                 "domain": "documentation_quality"}
            )

        #  Translation confidence is Normalizer metadata (PDF §2.3), never a
        #  risk contributor and never a guardrail — it does not enter scoring.

        guardrails_hit = sorted(
            key for key in keys if key in guardrail_catalogue
        )

        if total <= thresholds["low_max"]:
            level = "Low"
        elif total <= thresholds["medium_max"]:
            level = "Medium"
        else:
            level = "High"

        if guardrails_hit:
            level = "High"
        elif discharge_blocked and level == "Low":
            level = "Medium"

        payload = {
            "patient_id": patient_id,
            "risk_score": total,
            "risk_level": level,
            "thresholds": thresholds,
            "recommendation": {"Low": "Approve", "Medium": "Edit", "High": "Reject"}[level],
            "hitl_required": level != "Low" or discharge_blocked or bool(guardrails_hit),
            "discharge_blocked": discharge_blocked or bool(
                set(guardrails_hit) & {"allergy_contradiction", "high_risk_med_missing_in_ehr"}
            ),
            "hard_guardrails_hit": guardrails_hit,
            "breakdown": breakdown,
            "domain_totals": _domain_totals(breakdown),
            "rules_version": rules_version(),
            "computed_by": "analytics-server:8201/calculate_risk_score",
        }
        span.set_output({"risk_score": total, "risk_level": level})
        return payload


# =========================================================================== #
#  TOOL 2 — get_population_benchmarks
# =========================================================================== #
@mcp.tool(
    name="get_population_benchmarks",
    description=(
        "Return 30-day readmission rates, length of stay, medication-error rates "
        "and follow-up attendance benchmarks for the given ICD-10 diagnoses, "
        "compared against the hospital-network average."
    ),
)
async def get_population_benchmarks(
    ctx: Context,
    icd10_codes: list[str] | None = None,
    patient_id: str | None = None,
) -> dict[str, Any]:
    """Peer benchmarks for the patient's diagnoses."""
    with tracing.tool_span(
        "get_population_benchmarks",
        params={"icd10_codes": icd10_codes, "patient_id": patient_id},
        mcp_server="analytics",
    ) as span:
        codes = [str(code).strip().upper() for code in (icd10_codes or []) if code]

        #  Allow the caller to pass only a patient id: look the codes up in the EHR.
        if not codes and patient_id:
            try:
                from ..ehr import ehr_client

                demographics = ehr_client.patient(patient_id) or {}
                codes = [str(c).upper() for c in demographics.get("primary_dx", [])]
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not resolve ICD-10 codes for %s: %s", patient_id, exc)

        matched: list[dict[str, Any]] = []
        unknown: list[str] = []
        for code in codes:
            benchmark = POPULATION_BENCHMARKS.get(code)
            if benchmark is None:
                unknown.append(code)
                continue
            entry = {"icd10": code, **benchmark}
            entry["vs_network_median"] = round(
                benchmark["readmission_30d_rate"] - benchmark["network_median"], 4
            )
            entry["performance"] = (
                "better than network median"
                if entry["vs_network_median"] < 0 else "worse than network median"
            )
            matched.append(entry)

        cohort_readmission = (
            round(sum(b["readmission_30d_rate"] for b in matched) / len(matched), 4)
            if matched else None
        )
        highest = max(matched, key=lambda b: b["readmission_30d_rate"]) if matched else None

        payload = {
            "patient_id": patient_id,
            "requested_codes": codes,
            "benchmarks": matched,
            "unknown_codes": unknown,
            "cohort_readmission_30d_rate": cohort_readmission,
            "highest_risk_diagnosis": (
                {"icd10": highest["icd10"], "diagnosis": highest["diagnosis"],
                 "readmission_30d_rate": highest["readmission_30d_rate"]}
                if highest else None
            ),
            "network_averages": NETWORK_AVERAGES,
            "interpretation": _benchmark_interpretation(cohort_readmission),
            "computed_by": "analytics-server:8201/get_population_benchmarks",
        }
        span.set_output(
            {"matched": len(matched), "cohort_readmission": cohort_readmission}
        )
        return payload


# =========================================================================== #
#  TOOL 3 — generate_risk_heatmap
# =========================================================================== #
@mcp.tool(
    name="generate_risk_heatmap",
    description=(
        "Build a per-domain risk heatmap (medication safety, allergy safety, "
        "diagnosis alignment, care continuity, documentation quality, language "
        "quality, financial clearance) for rendering in the HITL dashboard."
    ),
)
async def generate_risk_heatmap(
    ctx: Context,
    patient_id: str,
    risk_keys: list[str] | None = None,
    as_markdown: bool = True,
) -> dict[str, Any]:
    """Domain-level heatmap of where a discharge case is losing points."""
    with tracing.tool_span(
        "generate_risk_heatmap", params={"patient_id": patient_id},
        mcp_server="analytics",
    ) as span:
        weights = risk_weights()
        keys = list(risk_keys or [])

        cells: list[dict[str, Any]] = []
        for domain, domain_keys in RISK_DOMAINS.items():
            hits = [key for key in keys if key in domain_keys]
            score = sum(int(weights.get(key, 0)) for key in hits)
            #  Domain intensity: 0 clean, 1 minor, 2 moderate, 3 severe.
            intensity = 0 if score == 0 else 1 if score <= 2 else 2 if score <= 5 else 3
            cells.append(
                {
                    "domain": domain,
                    "label": domain.replace("_", " ").title(),
                    "score": score,
                    "intensity": intensity,
                    "severity": ["clean", "minor", "moderate", "severe"][intensity],
                    "triggered_keys": hits,
                }
            )

        cells.sort(key=lambda cell: -cell["score"])
        payload: dict[str, Any] = {
            "patient_id": patient_id,
            "cells": cells,
            "total_score": sum(cell["score"] for cell in cells),
            "worst_domain": cells[0]["domain"] if cells and cells[0]["score"] else None,
            "clean_domains": [c["domain"] for c in cells if c["score"] == 0],
            "legend": {"0": "clean", "1": "minor", "2": "moderate", "3": "severe"},
            "computed_by": "analytics-server:8201/generate_risk_heatmap",
        }

        if as_markdown:
            payload["markdown"] = _heatmap_markdown(patient_id, cells)

        span.set_output({"total_score": payload["total_score"],
                         "worst_domain": payload["worst_domain"]})
        return payload


# =========================================================================== #
#  Helpers
# =========================================================================== #
def _domain_for(risk_key: str) -> str:
    for domain, keys in RISK_DOMAINS.items():
        if risk_key in keys:
            return domain
    return "other"


def _domain_totals(breakdown: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for item in breakdown:
        domain = item.get("domain", "other")
        totals[domain] = totals.get(domain, 0) + int(item.get("weight", 0))
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def _benchmark_interpretation(cohort_rate: float | None) -> str:
    if cohort_rate is None:
        return "No benchmark data available for the supplied diagnoses."
    network = NETWORK_AVERAGES["readmission_30d_rate"]
    delta = cohort_rate - network
    direction = "above" if delta > 0 else "below"
    return (
        f"Cohort 30-day readmission rate {cohort_rate:.1%} is "
        f"{abs(delta):.1%} {direction} the network average of {network:.1%}."
    )


_BLOCK = {0: "░░░░", 1: "▓░░░", 2: "▓▓▓░", 3: "████"}


def _heatmap_markdown(patient_id: str, cells: list[dict[str, Any]]) -> str:
    lines = [
        f"### Risk heatmap — {patient_id}",
        "",
        "| Domain | Heat | Score | Severity | Triggered |",
        "| --- | :-: | --: | --- | --- |",
    ]
    for cell in cells:
        triggered = ", ".join(cell["triggered_keys"]) or "—"
        lines.append(
            f"| {cell['label']} | {_BLOCK[cell['intensity']]} | {cell['score']} "
            f"| {cell['severity']} | {triggered} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main() -> None:
    configure_logging("mcp-analytics")
    service = settings.service("mcp_analytics")
    log.info(
        "Secondary MCP Analytics Server → http://%s:%s%s (3 tools)",
        service["host"], service["port"], service["path"],
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
