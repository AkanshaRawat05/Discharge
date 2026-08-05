"""
ehr/server.py
=============

Mock EHR System — FastAPI REST API on port 8050.

Serves the five hospital data domains the Validation Agent cross-checks against
(specification Table 14: *5 JSON data files — patients, medications, allergies,
labs, care_plans*), reading them from the supplied `mock_ehr/data.py`.

    GET /health
    GET /patients                      → list of patient summaries
    GET /patients/{patient_id}         → demographics + service line
    GET /patients/{patient_id}/medications
    GET /patients/{patient_id}/allergies
    GET /patients/{patient_id}/labs
    GET /patients/{patient_id}/care-plan
    GET /patients/{patient_id}/bundle  → all of the above in one round-trip
    GET /guidelines                    → clinical-pathway lookup
    GET /guidelines/{icd10}

`mock_ehr/data.py` is imported read-only and never modified — it is the
immutable source of truth for the intended test-case mismatches.

Run:
    python -m discharge_ai.ehr.server
    # or:  uvicorn discharge_ai.ehr.server:app --port 8050
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ..settings import configure_logging, settings

#  `mock_ehr` lives at the repository root, not under src/.
if str(settings.root) not in sys.path:
    sys.path.insert(0, str(settings.root))

from mock_ehr.data import (  # noqa: E402  (path set up above)
    ALLERGIES,
    CARE_PLANS,
    GUIDELINES,
    LABS,
    MED_ORDERS,
    PATIENTS,
)

log = logging.getLogger(__name__)

app = FastAPI(
    title="Mock EHR System",
    description=(
        "Read-only hospital EHR simulation for the Agentic AI Discharge "
        "Summaries capstone. Backed by mock_ehr/data.py."
    ),
    version=settings.cfg.get("project", {}).get("version", "1.0.0"),
)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _normalise(patient_id: str) -> str:
    return (patient_id or "").strip().upper()


def _require_patient(patient_id: str) -> dict[str, Any]:
    record = PATIENTS.get(_normalise(patient_id))
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {patient_id!r} not found in the EHR",
        )
    return record


# --------------------------------------------------------------------------- #
#  Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "mock-ehr",
        "patients": len(PATIENTS),
        "med_order_sets": len(MED_ORDERS),
        "allergy_records": len(ALLERGIES),
        "lab_sets": len(LABS),
        "care_plans": len(CARE_PLANS),
        "guidelines": len(GUIDELINES),
        "source": "mock_ehr/data.py",
    }


@app.get("/patients", tags=["patients"])
def list_patients() -> dict[str, Any]:
    return {
        "count": len(PATIENTS),
        "patients": [
            {
                "patient_id": pid,
                "patient_name": record.get("patient_name"),
                "service_line": record.get("service_line"),
                "primary_dx": record.get("primary_dx", []),
            }
            for pid, record in PATIENTS.items()
        ],
    }


@app.get("/patients/{patient_id}", tags=["patients"])
def get_patient(patient_id: str) -> dict[str, Any]:
    return _require_patient(patient_id)


@app.get("/patients/{patient_id}/medications", tags=["clinical"])
def get_medications(patient_id: str) -> dict[str, Any]:
    _require_patient(patient_id)
    pid = _normalise(patient_id)
    return {"patient_id": pid, "medications": MED_ORDERS.get(pid, [])}


@app.get("/patients/{patient_id}/allergies", tags=["clinical"])
def get_allergies(patient_id: str) -> dict[str, Any]:
    _require_patient(patient_id)
    pid = _normalise(patient_id)
    return {"patient_id": pid, "allergies": ALLERGIES.get(pid, [])}


@app.get("/patients/{patient_id}/labs", tags=["clinical"])
def get_labs(patient_id: str) -> dict[str, Any]:
    _require_patient(patient_id)
    pid = _normalise(patient_id)
    labs = LABS.get(pid, [])
    return {
        "patient_id": pid,
        "labs": labs,
        "abnormal": [lab for lab in labs if lab.get("abnormal")],
    }


@app.get("/patients/{patient_id}/care-plan", tags=["clinical"])
def get_care_plan(patient_id: str) -> dict[str, Any]:
    _require_patient(patient_id)
    pid = _normalise(patient_id)
    return {"patient_id": pid, "care_plan": CARE_PLANS.get(pid, {})}


@app.get("/patients/{patient_id}/bundle", tags=["clinical"])
def get_bundle(patient_id: str) -> dict[str, Any]:
    """Everything the EHR Validation Tool needs, in one call."""
    record = _require_patient(patient_id)
    pid = _normalise(patient_id)
    labs = LABS.get(pid, [])

    return {
        "patient_id": pid,
        "demographics": record,
        "medications": MED_ORDERS.get(pid, []),
        "allergies": ALLERGIES.get(pid, []),
        "labs": labs,
        "abnormal_labs": [lab for lab in labs if lab.get("abnormal")],
        "care_plan": CARE_PLANS.get(pid, {}),
        "guidelines": [
            {"icd10": code, **GUIDELINES[code]}
            for code in record.get("primary_dx", [])
            if code in GUIDELINES
        ],
    }


@app.get("/guidelines", tags=["guidelines"])
def list_guidelines() -> dict[str, Any]:
    return {
        "count": len(GUIDELINES),
        "guidelines": [{"icd10": code, **body} for code, body in GUIDELINES.items()],
    }


@app.get("/guidelines/{icd10}", tags=["guidelines"])
def get_guideline(icd10: str) -> dict[str, Any]:
    code = (icd10 or "").strip().upper()
    if code not in GUIDELINES:
        raise HTTPException(status_code=404, detail=f"No guideline for ICD-10 {code!r}")
    return {"icd10": code, **GUIDELINES[code]}


@app.exception_handler(HTTPException)
def http_exception_handler(_request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    import uvicorn

    configure_logging("mock-ehr")
    service = settings.service("mock_ehr")
    log.info(
        "Mock EHR starting on http://%s:%s (%d patients from %s)",
        service["host"], service["port"], len(PATIENTS),
        Path(settings.root, "mock_ehr", "data.py").name,
    )
    uvicorn.run(app, host=service["host"], port=int(service["port"]), log_level="warning")


if __name__ == "__main__":
    main()
