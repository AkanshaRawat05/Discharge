"""
pipeline.py
===========

End-to-end discharge processing, in two interchangeable modes:

* **A2A mode** (`mode="a2a"`) — the real distributed path: the Host Orchestrator
  calls each agent over the A2A Protocol (non-streaming for extract → normalise
  → validate, streaming for the summary).
* **In-process mode** (`mode="local"`) — the same agent handlers invoked
  directly in one process.  Used by the Streamlit dashboard so a reviewer can
  work even when the individual agent servers are not running, and by the tests.

Both modes produce the same `DischargeCase`, share one LangFuse trace id, and
accumulate one audit trail — so a report generated either way is comparable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from .common.schemas import (
    DischargeCase,
    DischargeSummary,
    ExtractedCase,
    ValidationReport,
)
from .observability import tracing
from .settings import settings

log = logging.getLogger(__name__)

#  Progress callback: (stage, message) → None
ProgressFn = Callable[[str, str], None]


@dataclass
class PipelineResult:
    patient_id: str
    trace_id: str
    mode: str
    case: ExtractedCase | None = None
    report: ValidationReport | None = None
    summary: DischargeSummary | None = None
    artefacts: dict[str, str] = field(default_factory=dict)
    stages: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    agents_used: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.report is not None and not self.errors

    @property
    def trace_url(self) -> str | None:
        return tracing.trace_url(self.trace_id)

    def to_case(self) -> DischargeCase:
        return DischargeCase(
            patient_id=self.patient_id,
            trace_id=self.trace_id,
            stage="summarised" if self.summary else ("validated" if self.report else "extracted"),
            extracted=self.case or ExtractedCase(),
            report=self.report,
            summary=self.summary,
            errors=self.errors,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "trace_id": self.trace_id,
            "trace_url": self.trace_url,
            "mode": self.mode,
            "ok": self.ok,
            "risk_level": self.report.risk.level.value if self.report else None,
            "risk_score": self.report.risk.score if self.report else None,
            "recommendation": (
                self.report.risk.recommendation.value if self.report else None
            ),
            "discharge_blocked": (
                self.report.risk.discharge_blocked if self.report else None
            ),
            "hitl_required": self.report.risk.hitl_required if self.report else None,
            "summary_generated": self.summary is not None,
            "artefacts": self.artefacts,
            "stages": self.stages,
            "errors": self.errors,
            "agents_used": self.agents_used,
        }


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #
async def run_pipeline(
    patient_id: str,
    *,
    mode: str = "local",
    generate_summary: bool = False,
    force_summary: bool = False,
    use_llm_extraction: bool = True,
    elicitation_answers: dict[str, Any] | None = None,
    elicitation_decline: bool = False,
    elicitation_cancel: bool = False,
    initial_case: ExtractedCase | None = None,
    trace_id: str | None = None,
    progress: ProgressFn | None = None,
) -> PipelineResult:
    """Run extract → normalise → validate for one patient.

    Per PDF §2.5 / §8 (Table 13), the discharge summary must only be produced on
    an explicit reviewer request — the dashboard's Generate button on view 5 —
    never as a silent side-effect of validation.  `generate_summary` therefore
    defaults to False; pass True only from a code path that represents an
    explicit user-initiated summary request.

    ``initial_case`` — the HITL round-trip: when the reviewer has edited the
    medication table (or answered elicitation prompts) on view 3, we must NOT
    re-parse the source documents from disk on the re-run because that would
    overwrite the reviewer's corrections. Pass the edited case here and the
    extract + normalise stages are skipped; validation runs against the
    reviewer-authoritative case verbatim.
    """
    patient_id = patient_id.strip().upper()
    trace_id = trace_id or tracing.start_case_trace(patient_id)
    result = PipelineResult(patient_id=patient_id, trace_id=trace_id, mode=mode)

    def report_progress(stage: str, message: str) -> None:
        result.stages.append({"stage": stage, "message": message})
        if progress:
            try:
                progress(stage, message)
            except Exception:  # noqa: BLE001 — a UI callback must never break the run
                pass

    runner = _A2ARunner(trace_id) if mode == "a2a" else _LocalRunner(trace_id)
    result.agents_used = runner.agent_labels

    with tracing.chain_span(
        f"pipeline:{patient_id}", trace_id=trace_id, input={"mode": mode}
    ) as span:
        # ---- 1. extract ----------------------------------------------------
        if initial_case is not None:
            #  HITL round-trip: use the reviewer's edited case verbatim — do NOT
            #  re-parse from disk (that would silently discard their corrections).
            #  Normalisation already ran on the first pass; skipping it here also
            #  ensures the reviewer's exact medication names survive.
            extracted = initial_case
            result.case = extracted
            report_progress(
                "extract",
                f"using reviewer-edited case ({len(extracted.medications)} medication(s), "
                f"{len(extracted.lab_tests)} lab result(s)) — extract + normalise "
                "skipped to preserve HITL corrections",
            )
            normalised = extracted
        else:
            report_progress("extract", "Clinical Extractor Agent (LangGraph :8100)…")
            try:
                extracted = await runner.extract(patient_id, use_llm_extraction)
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"extraction failed: {type(exc).__name__}: {exc}")
                span.fail(exc)
                report_progress("error", result.errors[-1])
                return result

            result.case = extracted
            report_progress(
                "extract",
                f"extracted {len(extracted.medications)} medication(s), "
                f"{len(extracted.lab_tests)} lab result(s); language "
                f"{extracted.detected_language}",
            )

            # ---- 2. normalise ----------------------------------------------
            report_progress("normalise", "Clinical Normalizer Agent (LangGraph :8102)…")
            try:
                normalised = await runner.normalise(patient_id, extracted)
                result.case = normalised
                report_progress(
                    "normalise",
                    f"translation confidence {normalised.translation_confidence:.2f} "
                    f"({len(normalised.expanded_abbreviations)} abbreviations expanded)",
                )
            except Exception as exc:  # noqa: BLE001
                #  Normalisation is enhancement, not a gate — carry on with the raw case.
                log.warning("Normalisation failed for %s: %s", patient_id, exc)
                result.errors.append(f"normalisation degraded: {type(exc).__name__}: {exc}")
                normalised = extracted
                report_progress("normalise", f"degraded: {exc}")

        # ---- 3. validate ---------------------------------------------------
        report_progress("validate", "Clinical Validation Agent (LangGraph :8101)…")
        try:
            report, artefacts = await runner.validate(
                patient_id,
                normalised,
                elicitation_answers=elicitation_answers,
                elicitation_decline=elicitation_decline,
                elicitation_cancel=elicitation_cancel,
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"validation failed: {type(exc).__name__}: {exc}")
            span.fail(exc)
            report_progress("error", result.errors[-1])
            return result

        result.report = report
        result.artefacts.update(artefacts)
        report_progress(
            "validate",
            f"risk {report.risk.level.value} (score {report.risk.score}), "
            f"{len(report.findings)} finding(s), "
            f"{'BLOCKED' if report.risk.discharge_blocked else 'not blocked'}",
        )

        # ---- 4. summarise --------------------------------------------------
        if generate_summary:
            allowed = force_summary or not report.risk.discharge_blocked
            if not allowed:
                report_progress(
                    "summarise",
                    "skipped — discharge is blocked; a reviewer must approve first",
                )
            else:
                report_progress(
                    "summarise", "Discharge Summary Generator (ADK :8104, streaming)…"
                )
                try:
                    summary, summary_artefacts = await runner.summarise(
                        patient_id, normalised, report, force=force_summary
                    )
                    result.summary = summary
                    result.artefacts.update(summary_artefacts)
                    report_progress(
                        "summarise",
                        f"generated {len(summary.sections) if summary else 0} section(s)",
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("Summary generation failed for %s: %s", patient_id, exc)
                    result.errors.append(
                        f"summary generation failed: {type(exc).__name__}: {exc}"
                    )
                    report_progress("summarise", f"failed: {exc}")

        # ---- 5. reindex so the RAG agent can answer about this case ---------
        try:
            from .rag import indexing

            indexing.build_index([patient_id], trace_id=trace_id)
            report_progress("index", "FAISS index refreshed for the RAG agent")
        except Exception as exc:  # noqa: BLE001
            log.debug("Re-indexing after the pipeline run failed: %s", exc)

        span.set_output(result.as_dict())

    tracing.flush()
    return result


# --------------------------------------------------------------------------- #
#  In-process runner
# --------------------------------------------------------------------------- #
class _LocalRunner:
    """Invoke the agent handlers directly, in this process."""

    agent_labels = [
        "Clinical Extractor Agent (LangGraph, in-process)",
        "Clinical Normalizer Agent (LangGraph, in-process)",
        "Clinical Validation Agent (LangGraph, in-process)",
        "Discharge Summary Generator (ADK, in-process)",
    ]

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        from types import SimpleNamespace

        self.ctx = SimpleNamespace(trace_id=trace_id)

    async def extract(self, patient_id: str, use_llm: bool) -> ExtractedCase:
        from .agents import extractor_agent

        payload = await extractor_agent.handle(
            {"patient_id": patient_id, "use_llm": use_llm, "trace_id": self.trace_id},
            self.ctx,
        )
        return ExtractedCase.model_validate(payload["case"])

    async def normalise(self, patient_id: str, case: ExtractedCase) -> ExtractedCase:
        from .agents import normalizer_agent

        payload = await normalizer_agent.handle(
            {
                "patient_id": patient_id,
                "case": case.model_dump(mode="json"),
                "trace_id": self.trace_id,
            },
            self.ctx,
        )
        return ExtractedCase.model_validate(payload["case"])

    async def validate(
        self, patient_id: str, case: ExtractedCase, **elicitation: Any
    ) -> tuple[ValidationReport, dict[str, str]]:
        from .agents import validator_agent

        payload = await validator_agent.handle(
            {
                "patient_id": patient_id,
                "case": case.model_dump(mode="json"),
                "trace_id": self.trace_id,
                **{key: value for key, value in elicitation.items() if value},
            },
            self.ctx,
        )
        return (
            ValidationReport.model_validate(payload["report"]),
            payload.get("artefacts", {}) or {},
        )

    async def summarise(
        self, patient_id: str, case: ExtractedCase, report: ValidationReport,
        *, force: bool = False,
    ) -> tuple[DischargeSummary | None, dict[str, str]]:
        from .agents import summary_agent

        final: dict[str, Any] | None = None
        async for event in summary_agent.handle(
            {
                "patient_id": patient_id,
                "case": case.model_dump(mode="json"),
                "report": report.model_dump(mode="json"),
                "trace_id": self.trace_id,
                "force": force,
            },
            self.ctx,
        ):
            if event.final:
                final = event.data

        if not final or not final.get("generated"):
            return None, {}
        return (
            DischargeSummary.model_validate(final["summary"]),
            final.get("artefacts", {}) or {},
        )


# --------------------------------------------------------------------------- #
#  A2A runner
# --------------------------------------------------------------------------- #
class _A2ARunner:
    """Call each agent over the A2A Protocol."""

    agent_labels = [
        "Clinical Extractor Agent (A2A :8100)",
        "Clinical Normalizer Agent (A2A :8102)",
        "Clinical Validation Agent (A2A :8101)",
        "Discharge Summary Generator (A2A :8104, streaming)",
    ]

    def __init__(self, trace_id: str) -> None:
        from .a2a_layer import A2AAgentClient

        self.trace_id = trace_id
        self.client = A2AAgentClient(trace_id=trace_id)

    @staticmethod
    def _require(result: Any, agent_key: str) -> dict[str, Any]:
        if not result.ok:
            raise RuntimeError(
                f"A2A call to {agent_key} failed: {result.error}. Is the agent "
                f"running? Start it with: python run_services.py --only {agent_key}"
            )
        return result.data

    async def extract(self, patient_id: str, use_llm: bool) -> ExtractedCase:
        data = self._require(
            await self.client.send(
                "extractor", {"patient_id": patient_id, "use_llm": use_llm}
            ),
            "extractor",
        )
        return ExtractedCase.model_validate(data["case"])

    async def normalise(self, patient_id: str, case: ExtractedCase) -> ExtractedCase:
        data = self._require(
            await self.client.send(
                "normalizer",
                {"patient_id": patient_id, "case": case.model_dump(mode="json")},
            ),
            "normalizer",
        )
        return ExtractedCase.model_validate(data["case"])

    async def validate(
        self, patient_id: str, case: ExtractedCase, **elicitation: Any
    ) -> tuple[ValidationReport, dict[str, str]]:
        data = self._require(
            await self.client.send(
                "validator",
                {
                    "patient_id": patient_id,
                    "case": case.model_dump(mode="json"),
                    **{key: value for key, value in elicitation.items() if value},
                },
            ),
            "validator",
        )
        return (
            ValidationReport.model_validate(data["report"]),
            data.get("artefacts", {}) or {},
        )

    async def summarise(
        self, patient_id: str, case: ExtractedCase, report: ValidationReport,
        *, force: bool = False,
    ) -> tuple[DischargeSummary | None, dict[str, str]]:
        final: dict[str, Any] | None = None
        async for chunk in self.client.stream(
            "summary",
            {
                "patient_id": patient_id,
                "case": case.model_dump(mode="json"),
                "report": report.model_dump(mode="json"),
                "force": force,
            },
        ):
            if chunk.data and chunk.data.get("summary") is not None:
                final = chunk.data

        if not final or not final.get("generated"):
            return None, {}
        return (
            DischargeSummary.model_validate(final["summary"]),
            final.get("artefacts", {}) or {},
        )


# --------------------------------------------------------------------------- #
#  Streaming summary passthrough (used by the dashboard's Summary page)
# --------------------------------------------------------------------------- #
async def stream_summary(
    patient_id: str,
    case: ExtractedCase,
    report: ValidationReport,
    *,
    mode: str = "local",
    force: bool = False,
    trace_id: str | None = None,
) -> AsyncIterator[tuple[str, str, dict[str, Any] | None]]:
    """Yield `(section, text, final_payload)` as the summary streams in."""
    trace_id = trace_id or tracing.start_case_trace(patient_id)
    payload = {
        "patient_id": patient_id,
        "case": case.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
        "trace_id": trace_id,
        "force": force,
    }

    if mode == "a2a":
        from .a2a_layer import A2AAgentClient

        client = A2AAgentClient(trace_id=trace_id)
        async for chunk in client.stream("summary", payload):
            yield chunk.section, chunk.text, chunk.data
        return

    from types import SimpleNamespace

    from .agents import summary_agent

    ctx = SimpleNamespace(trace_id=trace_id)
    async for event in summary_agent.handle(payload, ctx):
        yield event.section, event.text, event.data if event.final else None


# --------------------------------------------------------------------------- #
def pipeline_modes() -> list[str]:
    return ["local", "a2a"]


def default_mode() -> str:
    """Prefer the real A2A path when the agents are actually listening."""
    import socket

    for agent_key in ("extractor", "validator"):
        port = int(settings.agent(agent_key)["port"])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.3)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return "local"
    return "a2a"
