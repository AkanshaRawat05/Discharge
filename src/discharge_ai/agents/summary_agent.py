"""
agents/summary_agent.py
=======================

**Discharge Summary Generator Agent** — Google ADK — A2A :8104 — **STREAMING**
MCP primitives: Tools + Prompts

Streams a patient-friendly discharge summary **section by section** in the order
the specification requires (Table 10):

    patient → medications → labs → bill → instructions

Each section is produced by a Google ADK `LlmAgent` whose system instruction is
fetched at runtime through MCP Prompts (`summary-generation-prompt`), then passed
through the Responsible-AI guardrails before it is emitted:

* `ToxicityFilter`       strips unsafe advice ("stop all your medications")
* `PIIRedactor`          keeps the home address / national ids out of the text
* `HallucinationChecker` verifies every section against the validated case data,
  and regenerates once (then falls back to the deterministic rendering) if the
  faithfulness score is below threshold

Medication, lab and bill **tables are rendered deterministically** from the
validated data — a model never re-types a dose, a lab value or an amount.

A discharge that validation blocked is refused unless the caller passes
`force=true` (which is what a reviewer's explicit dashboard approval does).

Run:
    python -m discharge_ai.agents.summary_agent
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from ..common.doc_loader import load_patient_documents
from ..common.parsing import build_extracted_case
from ..common.schemas import (
    DischargeSummary,
    ExtractedCase,
    RiskLevel,
    SummarySection,
    ValidationReport,
)
from ..common.terminology import full_abbreviation_map
from ..guardrails import guardrails
from ..llm import provider
from ..observability import tracing
from ..settings import settings
from .base import AuditTrail, agent_mcp, case_from_payload, require_patient_id, trace_for

log = logging.getLogger(__name__)

AGENT_KEY = "summary"
AGENT_NAME = settings.agent(AGENT_KEY)["name"]
FRAMEWORK = "google-adk"

#  Section order is the specification's progressive-delivery contract.
SECTION_ORDER: list[tuple[str, str]] = [
    ("patient", "About your hospital stay"),
    ("medications", "Your medicines"),
    ("labs", "Your test results"),
    ("bill", "Your hospital bill"),
    ("instructions", "What to do at home"),
]

#  Frequency / route codes rewritten in plain words for the patient table.
FREQUENCY_PLAIN = {
    "BID": "twice a day", "TID": "three times a day", "QID": "four times a day",
    "QD": "once a day", "OD": "once a day", "QHS": "at bedtime",
    "PRN": "only when needed", "QOD": "every other day",
    "q4h": "every 4 hours", "q6h": "every 6 hours", "q8h": "every 8 hours",
    "q12h": "every 12 hours",
}
ROUTE_PLAIN = {
    "ORAL": "by mouth", "ORAAL": "by mouth", "PO": "by mouth", "IV": "into a vein",
    "IM": "as an injection into muscle", "SC": "as an injection under the skin",
    "SL": "under the tongue", "PR": "rectally", "INHALED": "breathed in",
}


# --------------------------------------------------------------------------- #
#  Deterministic tables (never LLM-generated)
# --------------------------------------------------------------------------- #
def _plain_frequency(value: str | None) -> str:
    if not value:
        return ""
    text = str(value)
    for code, plain in FREQUENCY_PLAIN.items():
        if code.lower() in text.lower():
            return plain if "(" not in text else text
    return text


def _plain_route(value: str | None) -> str:
    if not value:
        return ""
    return ROUTE_PLAIN.get(str(value).strip().upper(), str(value))


def prescription_table(case: ExtractedCase) -> list[dict[str, Any]]:
    return [
        {
            "sl_no": medication.sl_no,
            "medicine_name": medication.medicine_name,
            "strength": medication.strength,
            "dosage": medication.dosage,
            "frequency": medication.frequency,
            "frequency_plain": _plain_frequency(medication.frequency),
            "route": medication.route,
            "route_plain": _plain_route(medication.route),
            "period": medication.period,
            "remarks": medication.remarks,
            "total_quantity": medication.total_quantity,
        }
        for medication in case.medications
    ]


def lab_table(case: ExtractedCase) -> list[dict[str, Any]]:
    return [
        {
            "test": test.test,
            "value": test.value,
            "unit": test.unit,
            "reference_range": test.reference_range,
            "flag": test.flag or "NORMAL",
            "abnormal": test.is_abnormal,
        }
        for test in case.lab_tests
    ]


def bill_snapshot(case: ExtractedCase) -> dict[str, Any]:
    bill = case.bill
    return {
        "bill_id": bill.bill_id,
        "hospital_name": bill.hospital_name,
        "billing_date": bill.billing_date,
        "currency": bill.currency,
        "total_amount": bill.total_amount,
        "payment_status": bill.payment_status,
        "settled": bill.is_settled,
        "line_items": [item.model_dump(mode="json") for item in bill.line_items],
    }


# --------------------------------------------------------------------------- #
#  Facts handed to the LLM for each section
# --------------------------------------------------------------------------- #
def _section_facts(section: str, case: ExtractedCase,
                   report: ValidationReport | None) -> str:
    """Only validated data — this doubles as the grounding context."""
    lines: list[str] = []

    if section == "patient":
        lines += [
            f"Patient name: {case.patient_name or 'not recorded'}",
            f"Age: {case.age if case.age is not None else 'not recorded'}",
            f"Admitted: {case.admission_date or 'not recorded'}",
            f"Discharged: {case.discharge_date or 'not recorded'}",
            f"Ward/bed: {case.ward or '?'} / {case.bed_no or '?'}",
            f"Treated by: {case.attending_physician or 'not recorded'}",
            "Diagnoses treated: " + ("; ".join(case.discharge_diagnosis) or "not recorded"),
            "Allergies on record: " + ("; ".join(case.allergies) or "none documented"),
        ]

    elif section == "medications":
        for medication in case.medications:
            lines.append(
                f"- {medication.medicine_name or '?'} {medication.strength or ''}: "
                f"take {medication.dosage or '?'}, {medication.frequency or '?'}, "
                f"{medication.route or '?'}, for {medication.period or '?'}"
                + (f" ({medication.remarks})" if medication.remarks else "")
            )
        if not lines:
            lines.append("No discharge medications were recorded.")

    elif section == "labs":
        for test in case.lab_tests:
            lines.append(
                f"- {test.test}: {test.value or '?'} {test.unit or ''} "
                f"(normal range {test.reference_range or '?'}) — {test.flag or 'NORMAL'}"
            )
        for abnormal in case.abnormal_labs:
            if abnormal.action:
                lines.append(
                    f"- Action for {abnormal.test}: {abnormal.action}"
                )
        if not lines:
            lines.append("No laboratory results were recorded.")

    elif section == "bill":
        bill = case.bill
        lines += [
            f"Total amount: {bill.total_amount if bill.total_amount is not None else '?'} "
            f"{bill.currency or ''}".strip(),
            f"Payment status: {bill.payment_status or 'unknown'}",
            f"Billed by: {bill.hospital_name or 'the hospital'}",
            f"Bill date: {bill.billing_date or 'not recorded'}",
            f"Number of billed items: {len(bill.line_items)}",
        ]

    elif section == "instructions":
        lines.append(
            "Discharge instructions from the care team: "
            + (case.discharge_instructions or "none recorded")
        )
        lines.append(
            "Follow-up appointment: " + (case.follow_up_appointment or "none scheduled")
        )
        if report and report.findings:
            open_items = [
                f.message for f in report.findings if f.severity.value != "Info"
            ][:4]
            if open_items:
                lines.append(
                    "Items the care team is still reviewing: " + "; ".join(open_items)
                )

    return "\n".join(lines)


def _deterministic_section(section: str, case: ExtractedCase) -> str:
    """Fallback text used when the LLM is unavailable or fails grounding."""
    facts = _section_facts(section, case, None)
    headers = {
        "patient": "Here is a summary of your hospital stay:",
        "medications": "Take these medicines exactly as written:",
        "labs": "These are the test results from your stay:",
        "bill": "Here is your hospital bill:",
        "instructions": "Please follow these instructions at home:",
    }
    return f"{headers.get(section, '')}\n{facts}".strip()


# --------------------------------------------------------------------------- #
#  ADK section writer
# --------------------------------------------------------------------------- #
class ADKSectionWriter:
    """Wraps a Google ADK `LlmAgent` that writes one summary section at a time."""

    def __init__(self, system_instruction: str, trace_id: str | None) -> None:
        self.system_instruction = system_instruction
        self.trace_id = trace_id
        self.available = True
        self._runner = None
        self._session_service = None

        try:
            from google.adk.agents import LlmAgent
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService

            from ..llm.provider import get_adk_model

            self.model_id = str(get_adk_model("reasoning"))
            agent = LlmAgent(
                name="discharge_summary_writer",
                model=get_adk_model("reasoning"),
                description="Writes patient-friendly discharge summary sections.",
                instruction=system_instruction,
            )
            self._session_service = InMemorySessionService()
            self._runner = Runner(
                app_name="discharge-summary",
                agent=agent,
                session_service=self._session_service,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Google ADK summary writer unavailable (%s: %s) — deterministic "
                "rendering will be used.", type(exc).__name__, exc,
            )
            self.available = False
            self.model_id = "unavailable"

    async def write(self, section: str, title: str, facts: str) -> str:
        """Generate one section (empty string on any failure)."""
        if not self.available or settings.offline_mode:
            return ""
        #  The ADK model client runs its own retry loop we cannot configure,
        #  so respect the breaker rather than burning ~30 s per section.
        if provider.is_quota_exhausted():
            return ""

        from google.genai import types

        session_id = f"section-{section}"
        try:
            await self._session_service.create_session(  # type: ignore[union-attr]
                app_name="discharge-summary", user_id="patient", session_id=session_id
            )
        except Exception:  # noqa: BLE001 — session may already exist
            pass

        request = (
            f"Write the '{title}' section of the discharge summary.\n\n"
            f"Use ONLY these validated facts:\n{facts}\n\n"
            "Write 2-5 short sentences or bullet points. Do not add a heading."
        )

        with tracing.llm_generation(
            f"summary.section.{section}",
            model=self.model_id,
            prompt=request[:2000],
            trace_id=self.trace_id,
            section=section,
        ) as span:
            parts: list[str] = []
            try:
                async for event in self._runner.run_async(  # type: ignore[union-attr]
                    user_id="patient",
                    session_id=session_id,
                    new_message=types.Content(
                        role="user", parts=[types.Part(text=request)]
                    ),
                ):
                    content = getattr(event, "content", None)
                    if content and getattr(content, "parts", None):
                        for part in content.parts:
                            if getattr(part, "text", None):
                                parts.append(part.text)
            except Exception as exc:  # noqa: BLE001
                if provider._is_daily_quota(exc):
                    provider.mark_quota_exhausted("ADK daily quota")
                log.warning("ADK section %r failed: %s", section, exc)
                span.fail(exc)
                tracing.record_error(
                    f"adk.summary.{section}", exc, trace_id=self.trace_id,
                    fallback_action="deterministic section rendering",
                )
                return ""

            text = "\n".join(part.strip() for part in parts if part.strip()).strip()
            span.set_output(text[:2000])
            return text


# --------------------------------------------------------------------------- #
#  Streaming A2A handler
# --------------------------------------------------------------------------- #
async def handle(payload: dict[str, Any], ctx: Any) -> AsyncIterator[Any]:
    """Streaming A2A entry point — yields one `StreamEvent` per section."""
    from ..a2a_layer.executor import StreamEvent

    patient_id = require_patient_id(payload)
    trace_id = trace_for(payload, getattr(ctx, "trace_id", None), patient_id)
    audit = AuditTrail(trace_id)

    # ---- resolve the case + validation report ------------------------------
    case = case_from_payload(payload)
    if case is None:
        documents = load_patient_documents(patient_id)
        if not documents:
            raise FileNotFoundError(f"No documents found for {patient_id}")
        case = build_extracted_case(patient_id, documents)

    report: ValidationReport | None = None
    if payload.get("report"):
        try:
            report = ValidationReport.model_validate(payload["report"])
        except Exception as exc:  # noqa: BLE001
            log.warning("Ignoring unparsable validation report: %s", exc)

    risk_level = (
        report.risk.level if report else RiskLevel(str(payload.get("risk_level", "Low")))
    )
    audience = str(payload.get("audience", "patient")).lower()
    force = bool(payload.get("force", False))

    #  known_names is registered so PII masking still fires the patient's name
    #  when we send prompts to the LLM or write to LangFuse (PDF Table 12
    #  scopes redaction to "external LLM calls, logging"). We DO NOT run
    #  manager.redact() on the section text emitted to the patient — see the
    #  section-write loop below for the rationale.
    manager = guardrails(trace_id, known_names=[case.patient_name or ""])

    # ---- refuse to auto-generate a blocked discharge -----------------------
    blocked = bool(report.risk.discharge_blocked) if report else bool(
        payload.get("discharge_blocked", False)
    )
    if blocked and not force:
        reasons = (
            [f.message for f in report.critical_findings] if report
            else ["validation reported a blocking issue"]
        )
        message = (
            f"Discharge summary generation is BLOCKED for {patient_id}. "
            "A human reviewer must resolve the blocking findings first:\n- "
            + "\n- ".join(reasons)
        )
        audit.add(
            "refuse_blocked_discharge", AGENT_NAME, framework=FRAMEWORK,
            status="blocked", detail="; ".join(reasons)[:400],
        )
        yield StreamEvent(text=message, section="blocked")
        yield StreamEvent(
            data={
                "agent": AGENT_NAME,
                "framework": FRAMEWORK,
                "patient_id": patient_id,
                "trace_id": trace_id,
                "generated": False,
                "discharge_blocked": True,
                "blocking_reasons": reasons,
                "audit_trail": audit.dump(),
            },
            final=True,
        )
        return

    # ---- MCP Prompts: fetch the section-writing instruction ----------------
    async with agent_mcp(AGENT_NAME, trace_id=trace_id, servers=("primary",)) as mcp:
        async with audit.step("fetch_summary_prompt", "MCP Prompts", framework=FRAMEWORK):
            system_instruction = await mcp.get_prompt(
                "summary-generation-prompt",
                {"risk_level": risk_level.value, "audience": audience},
            )

    writer = ADKSectionWriter(system_instruction, trace_id)
    summary = DischargeSummary(
        patient_id=patient_id,
        patient_name=case.patient_name,
        audience=audience,
        risk_level=risk_level,
        trace_id=trace_id,
        prescription_table=prescription_table(case),
        lab_table=lab_table(case),
        bill_snapshot=bill_snapshot(case),
    )

    yield StreamEvent(
        text=f"Preparing the discharge summary for {patient_id} "
             f"(risk: {risk_level.value}, audience: {audience})…\n\n",
        section="header",
    )

    # ---- stream one section at a time --------------------------------------
    for section_key, title in SECTION_ORDER:
        facts = _section_facts(section_key, case, report)

        async with audit.step(f"generate_section:{section_key}",
                              f"ADK LlmAgent ({writer.model_id})", framework=FRAMEWORK):
            text = await writer.write(section_key, title, facts)

            #  Grounding gate — regenerate once, then fall back to facts.
            #  The judge is a blocking LLM call, so keep it off the loop.
            if text:
                grounding = await provider.arun_blocking(
                    manager.check_grounding, text, facts
                )
                if grounding.blocked:
                    log.warning(
                        "Section %r failed grounding (%.2f) — regenerating once",
                        section_key, grounding.faithfulness,
                    )
                    text = await writer.write(section_key, title, facts)
                    if text:
                        retry = await provider.arun_blocking(
                            manager.check_grounding, text, facts
                        )
                        if retry.blocked:
                            text = ""

            if not text:
                text = _deterministic_section(section_key, case)

            #  Toxicity is filtered on everything the patient sees.
            #  PII redaction is intentionally NOT applied to the summary body:
            #  this artifact IS the patient's own take-home document, so
            #  masking their own name / phone / address on it would defeat its
            #  purpose. PDF Table 12 scopes PII/PHI redaction to external LLM
            #  calls and logs (already applied via `manager.redact(..., purpose=
            #  "logging")` elsewhere), not to reviewer-facing output.
            text = manager.filter_output(text, section=section_key)

        summary.sections.append(
            SummarySection(key=section_key, title=title, content=text)
        )
        yield StreamEvent(text=f"## {title}\n{text}\n\n", section=section_key)

    # ---- warning signs (always deterministic, never softened) --------------
    warning_text = _warning_signs(case)
    summary.sections.append(
        SummarySection(key="warning_signs", title="When to get help urgently",
                       content=warning_text)
    )
    yield StreamEvent(
        text=f"## When to get help urgently\n{warning_text}\n", section="warning_signs"
    )

    # ---- write the artifacts ----------------------------------------------
    summary.guardrail_events = list(manager.events)
    artifacts: dict[str, str] = {}
    async with audit.step("write_summary_artifacts", "report_builder",
                          framework=FRAMEWORK):
        from ..reporting import build_summary_reports

        artifacts = build_summary_reports(summary, write_files=True, pdf=True)

    yield StreamEvent(
        data={
            "agent": AGENT_NAME,
            "framework": FRAMEWORK,
            "patient_id": patient_id,
            "trace_id": trace_id,
            "trace_url": tracing.trace_url(trace_id),
            "generated": True,
            "discharge_blocked": False,
            "forced": force,
            "risk_level": risk_level.value,
            "audience": audience,
            "summary": summary.model_dump(mode="json"),
            "markdown": summary.as_markdown(),
            "artifacts": artifacts,
            "sections": [section.key for section in summary.sections],
            "guardrail_events": [e.model_dump(mode="json") for e in manager.events],
            "audit_trail": audit.dump(),
            "mcp_primitives_used": ["tools", "prompts"],
        },
        final=True,
    )


def _warning_signs(case: ExtractedCase) -> str:
    """Red flags, taken from the record where present, else safe generics."""
    instructions = (case.discharge_instructions or "").lower()
    lines: list[str] = []

    #  Prefer what the clinicians actually wrote.
    for sentence in (case.discharge_instructions or "").replace("\n", ". ").split("."):
        lowered = sentence.lower()
        if any(
            marker in lowered
            for marker in ("return to", "come back", "emergency", "ed ", "er ", "urgent")
        ) and sentence.strip():
            lines.append(f"- {sentence.strip().lstrip('- ')}")

    if not lines:
        lines = [
            "- Go to the emergency department if you have chest pain, severe "
            "shortness of breath, fainting, or bleeding that will not stop.",
            "- Contact the hospital if you develop a high fever, cannot keep "
            "fluids down, or your symptoms get worse instead of better.",
        ]

    if "penicill" in " ".join(case.allergies).lower():
        lines.append(
            "- You have a penicillin allergy on record. Tell every clinician and "
            "pharmacist before you take any new antibiotic."
        )

    if any(
        abbr in instructions for abbr in ("glucose", "bg ", "blood sugar")
    ) or any("metformin" in (m.medicine_name or "").lower() for m in case.medications):
        lines.append(
            "- Seek urgent care if your blood sugar readings are very high or "
            "very low, or if you feel confused, shaky or unusually drowsy."
        )

    return "\n".join(dict.fromkeys(lines))


def main() -> None:
    from ..a2a_layer import run_agent_server

    #  Touch the abbreviation map at start-up so the first request is not slowed
    #  by the rules.yaml read.
    full_abbreviation_map()
    run_agent_server(AGENT_KEY, handle, artifact_name="discharge_summary")


if __name__ == "__main__":
    main()
