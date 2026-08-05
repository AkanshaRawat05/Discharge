"""
ehr/client.py
=============

Client for the Mock EHR REST API used by the **EHR Validation Tool** on the
Primary MCP Server.

It talks HTTP to `:8050` (the real integration path the spec asks for) and falls
back to importing `mock_ehr.data` directly when the service is not running, so a
single-process demo or unit test still validates correctly.  The fallback is
logged and recorded in the audit trail so the degraded path is never silent.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import httpx

from ..settings import settings

log = logging.getLogger(__name__)


class EHRClient:
    """Read-only access to the hospital EHR (HTTP first, in-process fallback)."""

    def __init__(self, base_url: str | None = None, timeout: float = 10.0) -> None:
        self.base_url = (base_url or settings.ehr_base_url).rstrip("/")
        self.timeout = timeout
        self._offline_reason: str | None = None

    # ------------------------------------------------------------------ #
    @property
    def source(self) -> str:
        return "mock_ehr/data.py (in-process)" if self._offline_reason else self.base_url

    @property
    def degraded(self) -> bool:
        return self._offline_reason is not None

    def health(self) -> dict[str, Any]:
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            return {"status": "unreachable", "error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------ #
    def _get(self, path: str) -> dict[str, Any] | None:
        try:
            response = httpx.get(f"{self.base_url}{path}", timeout=self.timeout)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            self._offline_reason = None
            return response.json()
        except Exception as exc:  # noqa: BLE001
            if self._offline_reason is None:
                log.warning(
                    "Mock EHR at %s unreachable (%s) — falling back to the "
                    "in-process mock_ehr.data module.",
                    self.base_url, type(exc).__name__,
                )
            self._offline_reason = f"{type(exc).__name__}: {exc}"
            return None

    @staticmethod
    def _local_data() -> Any:
        if str(settings.root) not in sys.path:
            sys.path.insert(0, str(settings.root))
        from mock_ehr import data  # noqa: PLC0415 — deliberate lazy import

        return data

    # ------------------------------------------------------------------ #
    def bundle(self, patient_id: str) -> dict[str, Any]:
        """Demographics + meds + allergies + labs + care plan + guidelines."""
        pid = (patient_id or "").strip().upper()

        payload = self._get(f"/patients/{pid}/bundle")
        if payload is not None:
            payload["source"] = self.base_url
            return payload

        data = self._local_data()
        demographics = data.PATIENTS.get(pid)
        if demographics is None:
            return {
                "patient_id": pid,
                "found": False,
                "error": f"Patient {pid} not present in the EHR",
                "source": self.source,
            }

        labs = data.LABS.get(pid, [])
        return {
            "patient_id": pid,
            "found": True,
            "demographics": demographics,
            "medications": data.MED_ORDERS.get(pid, []),
            "allergies": data.ALLERGIES.get(pid, []),
            "labs": labs,
            "abnormal_labs": [lab for lab in labs if lab.get("abnormal")],
            "care_plan": data.CARE_PLANS.get(pid, {}),
            "guidelines": [
                {"icd10": code, **data.GUIDELINES[code]}
                for code in demographics.get("primary_dx", [])
                if code in data.GUIDELINES
            ],
            "source": self.source,
            "degraded_reason": self._offline_reason,
        }

    def patient(self, patient_id: str) -> dict[str, Any]:
        return self.bundle(patient_id).get("demographics", {})

    def medications(self, patient_id: str) -> list[dict[str, Any]]:
        return self.bundle(patient_id).get("medications", [])

    def allergies(self, patient_id: str) -> list[str]:
        return self.bundle(patient_id).get("allergies", [])

    def labs(self, patient_id: str) -> list[dict[str, Any]]:
        return self.bundle(patient_id).get("labs", [])

    def abnormal_labs(self, patient_id: str) -> list[dict[str, Any]]:
        return self.bundle(patient_id).get("abnormal_labs", [])

    def care_plan(self, patient_id: str) -> dict[str, Any]:
        return self.bundle(patient_id).get("care_plan", {})

    def guidelines(self, patient_id: str) -> list[dict[str, Any]]:
        return self.bundle(patient_id).get("guidelines", [])

    def list_patients(self) -> list[dict[str, Any]]:
        payload = self._get("/patients")
        if payload is not None:
            return payload.get("patients", [])

        data = self._local_data()
        return [
            {
                "patient_id": pid,
                "patient_name": record.get("patient_name"),
                "service_line": record.get("service_line"),
                "primary_dx": record.get("primary_dx", []),
            }
            for pid, record in data.PATIENTS.items()
        ]


#  Shared instance — the MCP tools are stateless, so one client is enough.
ehr_client = EHRClient()
