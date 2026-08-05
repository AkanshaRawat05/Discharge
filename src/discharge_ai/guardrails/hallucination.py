"""
guardrails/hallucination.py
===========================

Hallucination check (LLM-as-judge).

Trigger : a RAG-generated response whose faithfulness score is < 0.70.
Action  : block the response and request regeneration.

Scoring is two-layered so the guardrail still works without an LLM:

1.  **Lexical grounding** — what fraction of the answer's content tokens and,
    critically, its *numbers* (doses, lab values, amounts, dates) appear in the
    retrieved context.  A number in the answer that is absent from the context
    is the classic RAG hallucination and is penalised hard.
2.  **LLM judge** — `internal_prompts.hallucination_judge` from prompts.yaml.
    When the judge answers, its score is blended with the lexical score
    (weighted toward the judge); when it fails/quota-limits, the lexical score
    stands on its own.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..common.rules import quality_thresholds
from ..llm import provider
from ..settings import settings

log = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "at", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "you", "your", "their",
    "has", "have", "had", "will", "would", "should", "can", "could", "may",
    "not", "no", "do", "does", "did", "there", "then", "than", "also", "per",
    "patient", "records", "record", "information", "available", "source",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z-]{2,}")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

#  The mandated refusal string is by definition grounded.
_REFUSAL_MARKERS = (
    "i don't know", "i do not know", "not available in the patient records",
)


@dataclass
class HallucinationResult:
    faithfulness: float = 1.0
    lexical_score: float = 1.0
    judge_score: float | None = None
    verdict: str = "grounded"          # grounded | partially_grounded | hallucinated
    unsupported_claims: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    blocked: bool = False
    threshold: float = 0.70
    judged_by: str = "lexical"

    @property
    def detail(self) -> str:
        return (
            f"faithfulness={self.faithfulness:.2f} (threshold {self.threshold:.2f}, "
            f"judge={self.judged_by}) verdict={self.verdict}"
        )


class HallucinationChecker:
    """Grounding gate for RAG answers and generated summaries."""

    name = "HallucinationChecker"

    def __init__(self, threshold: float | None = None, use_llm_judge: bool = True) -> None:
        configured = settings.guardrail_cfg.get("hallucination_faithfulness_min")
        rules_min = quality_thresholds().get("rag_groundedness_min")
        self.threshold = float(
            threshold if threshold is not None else (configured or rules_min or 0.70)
        )
        self.use_llm_judge = use_llm_judge and not settings.offline_mode

    # ------------------------------------------------------------------ #
    def check(self, answer: str, context: str) -> HallucinationResult:
        result = HallucinationResult(threshold=self.threshold)

        if not answer or not answer.strip():
            result.faithfulness = 0.0
            result.verdict = "hallucinated"
            result.blocked = True
            result.unsupported_claims = ["empty answer"]
            return result

        if any(marker in answer.lower() for marker in _REFUSAL_MARKERS):
            #  Refusing to answer is the *correct* grounded behaviour.
            return result

        if not context or not context.strip():
            result.faithfulness = 0.0
            result.lexical_score = 0.0
            result.verdict = "hallucinated"
            result.blocked = True
            result.unsupported_claims = ["answer produced with no retrieved context"]
            return result

        lexical, unsupported_tokens, unsupported_numbers = self._lexical_grounding(
            answer, context
        )
        result.lexical_score = lexical
        result.unsupported_claims = unsupported_tokens
        result.unsupported_numbers = unsupported_numbers
        result.faithfulness = lexical

        if self.use_llm_judge:
            judge = self._llm_judge(answer, context)
            if judge is not None:
                result.judge_score = judge["faithfulness"]
                result.judged_by = "llm+lexical"
                #  Trust the judge but never let it fully override hard numeric
                #  evidence that a dose/value was invented.
                result.faithfulness = round(0.7 * judge["faithfulness"] + 0.3 * lexical, 3)
                if judge.get("unsupported_claims"):
                    result.unsupported_claims = (
                        judge["unsupported_claims"] + result.unsupported_claims
                    )[:10]

        if unsupported_numbers:
            result.faithfulness = min(result.faithfulness, 0.55)

        if result.faithfulness >= self.threshold:
            result.verdict = "grounded"
        elif result.faithfulness >= self.threshold * 0.7:
            result.verdict = "partially_grounded"
        else:
            result.verdict = "hallucinated"

        result.blocked = result.faithfulness < self.threshold
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _lexical_grounding(answer: str, context: str) -> tuple[float, list[str], list[str]]:
        context_lower = context.lower()
        context_tokens = set(_TOKEN_RE.findall(context_lower))
        context_numbers = set(_NUMBER_RE.findall(context_lower))

        answer_tokens = [
            token for token in _TOKEN_RE.findall(answer.lower())
            if token not in _STOPWORDS
        ]
        answer_numbers = _NUMBER_RE.findall(answer)

        unsupported_tokens = sorted(
            {token for token in answer_tokens if token not in context_tokens}
        )
        unsupported_numbers = sorted(
            {number for number in answer_numbers if number not in context_numbers}
        )

        if not answer_tokens and not answer_numbers:
            return 1.0, [], []

        supported_tokens = len(answer_tokens) - len(
            [t for t in answer_tokens if t not in context_tokens]
        )
        token_ratio = supported_tokens / len(answer_tokens) if answer_tokens else 1.0

        number_ratio = 1.0
        if answer_numbers:
            supported_numbers = sum(1 for n in answer_numbers if n in context_numbers)
            number_ratio = supported_numbers / len(answer_numbers)

        #  Numbers carry more clinical risk than prose, so weight them heavily.
        score = round(0.55 * token_ratio + 0.45 * number_ratio, 3)
        return score, unsupported_tokens[:10], unsupported_numbers[:10]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _llm_judge(answer: str, context: str) -> dict | None:
        from ..common.prompt_store import internal_prompt

        template = internal_prompt("hallucination_judge")
        if not template:
            return None

        prompt = template.format(context=context[:8000], answer=answer[:4000])
        payload = provider.complete_json(prompt, purpose="fast")

        if not isinstance(payload, dict) or "faithfulness" not in payload:
            return None
        try:
            score = float(payload["faithfulness"])
        except (TypeError, ValueError):
            return None

        claims = payload.get("unsupported_claims") or []
        return {
            "faithfulness": max(0.0, min(1.0, score)),
            "unsupported_claims": [str(c) for c in claims][:10]
            if isinstance(claims, list) else [],
            "verdict": str(payload.get("verdict", "")),
        }
