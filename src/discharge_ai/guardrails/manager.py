"""
guardrails/manager.py
=====================

`GuardrailManager` — the single entry point every agent uses for Responsible AI,
and the owner of the **HITL escalation** rule from specification Table 12:

    risk_level == High  or  discharge_blocked == True
        →  mandatory human review, no auto-approve

Each check emits a `GuardrailEvent` (recorded on the validation report) *and* a
LangFuse guardrail span, so an auditor can see which guardrail fired, what it
decided, and whether content was blocked or allowed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..common.schemas import GuardrailEvent, RiskLevel
from ..observability import tracing
from ..settings import settings
from .hallucination import HallucinationChecker, HallucinationResult
from .injection import InjectionResult, PromptInjectionGuard
from .pii import PIIRedactor
from .toxicity import ToxicityFilter

log = logging.getLogger(__name__)


@dataclass
class EscalationDecision:
    hitl_required: bool = False
    auto_approve_allowed: bool = True
    reasons: list[str] = field(default_factory=list)

    @property
    def detail(self) -> str:
        if not self.hitl_required:
            return "auto-approval permitted"
        return "HITL required: " + "; ".join(self.reasons)


class GuardrailManager:
    """Orchestrates the five Responsible-AI guardrails."""

    name = "GuardrailManager"

    def __init__(self, trace_id: str | None = None,
                 known_names: list[str] | None = None) -> None:
        self.trace_id = trace_id
        cfg = settings.guardrail_cfg

        self.enabled = {
            "pii": bool(cfg.get("pii_redaction", True)),
            "hallucination": bool(cfg.get("hallucination_check", True)),
            "injection": bool(cfg.get("prompt_injection_guard", True)),
            "toxicity": bool(cfg.get("toxicity_filter", True)),
            "escalation": bool(cfg.get("escalate_on_high_risk", True)),
        }

        self.pii = PIIRedactor(known_names=known_names)
        self.hallucination = HallucinationChecker()
        self.injection = PromptInjectionGuard()
        self.toxicity = ToxicityFilter()

        self.events: list[GuardrailEvent] = []

    # ------------------------------------------------------------------ #
    #  1. PII / PHI redaction
    # ------------------------------------------------------------------ #
    def redact(self, value: Any, *, purpose: str = "logging") -> Any:
        """Mask PII before logging or calling an external API."""
        if not self.enabled["pii"]:
            return value

        if isinstance(value, str):
            with tracing.guardrail_span(
                "PIIRedactor", trace_id=self.trace_id, purpose=purpose
            ) as span:
                result = self.pii.redact(value)
                span.set_output({"triggered": result.triggered, "count": result.count})
                self._record(
                    "PIIRedactor",
                    triggered=result.triggered,
                    action="masked before logging/API call" if result.triggered else "pass",
                    detail=result.detail,
                )
                return result.text

        return self.pii.redact_any(value)

    def safe_for_logging(self, value: Any) -> Any:
        """Redact without emitting an event (used on hot logging paths)."""
        return self.pii.redact_any(value) if self.enabled["pii"] else value

    # ------------------------------------------------------------------ #
    #  2. Prompt-injection guard
    # ------------------------------------------------------------------ #
    def check_query(self, query: str) -> InjectionResult:
        """Screen a user question before it reaches the RAG pipeline."""
        if not self.enabled["injection"]:
            return InjectionResult(sanitised_text=query)

        with tracing.guardrail_span(
            "PromptInjectionGuard", trace_id=self.trace_id, content=query
        ) as span:
            result = self.injection.check(query)
            span.set_output(
                {
                    "is_injection": result.is_injection,
                    "action": result.action,
                    "patterns": result.patterns_matched,
                }
            )
            if result.is_injection:
                log.warning(
                    "Prompt injection %s: %s", result.action, result.patterns_matched
                )
            self._record(
                "PromptInjectionGuard",
                triggered=result.is_injection,
                action=(
                    "rejected; alert logged" if result.blocked
                    else "sanitised; alert logged" if result.is_injection
                    else "pass"
                ),
                detail=result.detail,
            )
            return result

    def sanitise_context(self, text: str) -> str:
        """Neutralise instructions embedded in retrieved document text."""
        if not self.enabled["injection"]:
            return text
        return self.injection.sanitise_context(text)

    # ------------------------------------------------------------------ #
    #  3. Hallucination check
    # ------------------------------------------------------------------ #
    def check_grounding(self, answer: str, context: str) -> HallucinationResult:
        """Faithfulness gate: < 0.70 blocks the response for regeneration."""
        if not self.enabled["hallucination"]:
            return HallucinationResult()

        with tracing.guardrail_span(
            "HallucinationChecker", trace_id=self.trace_id, content=answer
        ) as span:
            result = self.hallucination.check(answer, context)
            span.set_output(
                {
                    "faithfulness": result.faithfulness,
                    "verdict": result.verdict,
                    "blocked": result.blocked,
                    "unsupported_numbers": result.unsupported_numbers,
                }
            )
            span.score("faithfulness", result.faithfulness, result.verdict)
            self._record(
                "HallucinationChecker",
                triggered=result.blocked,
                action="blocked response; regeneration requested" if result.blocked else "pass",
                detail=result.detail,
                score=result.faithfulness,
            )
            return result

    # ------------------------------------------------------------------ #
    #  4. Toxicity filter
    # ------------------------------------------------------------------ #
    def filter_output(self, text: str, *, section: str = "") -> str:
        """Filter unsafe/toxic language out of generated clinical text."""
        if not self.enabled["toxicity"] or not text:
            return text

        with tracing.guardrail_span(
            "ToxicityFilter", trace_id=self.trace_id, content=text, section=section
        ) as span:
            result = self.toxicity.check(text)
            span.set_output(
                {"is_toxic": result.is_toxic, "categories": result.categories}
            )
            if result.is_toxic:
                log.warning(
                    "Toxicity filter removed %d span(s) from section %r",
                    len(result.removed_spans), section,
                )
            self._record(
                "ToxicityFilter",
                triggered=result.is_toxic,
                action="filtered before inclusion in summary" if result.is_toxic else "pass",
                detail=f"{result.detail}{f' (section: {section})' if section else ''}",
            )
            return result.cleaned_text

    # ------------------------------------------------------------------ #
    #  5. HITL escalation
    # ------------------------------------------------------------------ #
    def evaluate_escalation(
        self,
        *,
        risk_level: RiskLevel | str,
        discharge_blocked: bool,
        hard_guardrails_hit: list[str] | None = None,
    ) -> EscalationDecision:
        """Mandatory human review for High risk or a blocked discharge."""
        level = risk_level.value if isinstance(risk_level, RiskLevel) else str(risk_level)
        decision = EscalationDecision()

        if not self.enabled["escalation"]:
            return decision

        if level == RiskLevel.HIGH.value:
            decision.reasons.append("risk_level=High")
        if discharge_blocked:
            decision.reasons.append("discharge_blocked=True")
        for guardrail in hard_guardrails_hit or []:
            decision.reasons.append(f"hard guardrail: {guardrail}")

        decision.hitl_required = bool(decision.reasons)
        decision.auto_approve_allowed = not decision.hitl_required

        with tracing.guardrail_span(
            "GuardrailManager.HITLEscalation",
            trace_id=self.trace_id,
            risk_level=level,
            discharge_blocked=discharge_blocked,
        ) as span:
            span.set_output(
                {
                    "hitl_required": decision.hitl_required,
                    "auto_approve_allowed": decision.auto_approve_allowed,
                    "reasons": decision.reasons,
                }
            )

        self._record(
            "GuardrailManager",
            triggered=decision.hitl_required,
            action=(
                "mandatory human review; no auto-approve"
                if decision.hitl_required else "auto-approve permitted"
            ),
            detail=decision.detail,
        )
        return decision

    # ------------------------------------------------------------------ #
    def _record(self, guardrail: str, *, triggered: bool, action: str,
                detail: str = "", score: float | None = None) -> None:
        self.events.append(
            GuardrailEvent(
                guardrail=guardrail,
                triggered=triggered,
                action=action,
                detail=detail,
                score=score,
            )
        )

    @property
    def triggered_events(self) -> list[GuardrailEvent]:
        return [event for event in self.events if event.triggered]

    def summary(self) -> dict[str, Any]:
        return {
            "checks_run": len(self.events),
            "checks_triggered": len(self.triggered_events),
            "guardrails": sorted({event.guardrail for event in self.events}),
            "triggered": [
                {"guardrail": e.guardrail, "action": e.action, "detail": e.detail}
                for e in self.triggered_events
            ],
        }


#  Convenience factory so callers read as `guardrails(trace_id).check_query(q)`.
def guardrails(trace_id: str | None = None,
               known_names: list[str] | None = None) -> GuardrailManager:
    return GuardrailManager(trace_id=trace_id, known_names=known_names)
