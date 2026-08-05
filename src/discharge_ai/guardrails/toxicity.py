"""
guardrails/toxicity.py
======================

Toxicity filter.

Trigger : LLM output destined for clinical instructions.
Action  : filter the offending content out before it enters the summary.

Clinical bluntness is *not* toxicity — "the patient is obese" and "non-compliant
with therapy" are legitimate clinical statements.  The filter targets demeaning,
discriminatory or abusive language, plus unsafe advice that a discharge summary
must never carry (e.g. telling a patient to stop a prescribed medicine).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm import provider
from ..settings import settings

TOXIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("abusive_language", re.compile(
        r"\b(stupid|idiot|idiotic|moron|worthless|pathetic|disgusting|"
        r"lazy\s+patient|drug\s*seeker|frequent\s+flyer|waste\s+of\s+(a\s+)?bed)\b",
        re.IGNORECASE)),
    ("discriminatory", re.compile(
        r"\b(because\s+(he|she|they)\s+(is|are)\s+(too\s+)?(old|foreign|poor|"
        r"illiterate|uneducated)|these\s+people\s+(always|never))\b",
        re.IGNORECASE)),
    ("blaming", re.compile(
        r"\b(the\s+patient\s+(is\s+)?(to\s+blame|deserves|brought\s+this\s+on)|"
        r"self[- ]inflicted\s+and\s+undeserving)\b",
        re.IGNORECASE)),
    ("hopelessness", re.compile(
        r"\b(nothing\s+(more\s+)?can\s+be\s+done|there\s+is\s+no\s+hope|"
        r"you\s+will\s+die\s+soon)\b",
        re.IGNORECASE)),
]

#  Clinically unsafe instructions that must never reach a patient summary.
UNSAFE_ADVICE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("unsafe_medication_advice", re.compile(
        r"\b(stop|discontinue|skip|double|triple)\b[^.\n]{0,30}\b"
        r"(all\s+)?(your\s+)?(medication|medicine|dose|tablet|insulin|antibiotic)s?\b"
        r"(?![^.\n]{0,40}\b(as\s+(directed|instructed|advised)|only\s+if\s+your\s+"
        r"(doctor|clinician|physician))\b)",
        re.IGNORECASE)),
    ("discourage_care", re.compile(
        r"\b(do\s+not|don't|no\s+need\s+to)\b[^.\n]{0,25}\b"
        r"(see\s+(a\s+)?(doctor|physician)|go\s+to\s+(the\s+)?(ed|er|emergency)|"
        r"seek\s+(medical\s+)?(help|care|attention)|attend\s+follow[- ]up)\b",
        re.IGNORECASE)),
]


@dataclass
class ToxicityResult:
    is_toxic: bool = False
    categories: list[str] = field(default_factory=list)
    cleaned_text: str = ""
    removed_spans: list[str] = field(default_factory=list)
    judged_by: str = "rules"

    @property
    def detail(self) -> str:
        if not self.is_toxic:
            return "no toxic or unsafe content detected"
        return f"filtered categories: {', '.join(sorted(set(self.categories)))}"


class ToxicityFilter:
    """Screen generated clinical text before it enters a discharge summary."""

    name = "ToxicityFilter"
    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n")

    def __init__(self, use_llm_judge: bool = False) -> None:
        #  Rule-based by default: this filter runs on every generated section,
        #  and an extra LLM round-trip per section is not worth the latency.
        self.use_llm_judge = use_llm_judge and not settings.offline_mode

    def check(self, text: str) -> ToxicityResult:
        result = ToxicityResult(cleaned_text=text or "")
        if not text or not isinstance(text, str):
            return result

        patterns = TOXIC_PATTERNS + UNSAFE_ADVICE_PATTERNS
        offending: list[str] = []

        for label, pattern in patterns:
            for match in pattern.finditer(text):
                result.is_toxic = True
                if label not in result.categories:
                    result.categories.append(label)
                offending.append(match.group(0))

        if result.is_toxic:
            #  Drop whole sentences: a half-removed clinical instruction is
            #  more dangerous than no instruction at all.
            kept: list[str] = []
            for sentence in self._SENTENCE_SPLIT.split(text):
                if any(
                    pattern.search(sentence) for _label, pattern in patterns
                ):
                    result.removed_spans.append(sentence.strip()[:200])
                    continue
                if sentence.strip():
                    kept.append(sentence.strip())
            result.cleaned_text = " ".join(kept)

        if self.use_llm_judge and not result.is_toxic:
            judged = self._llm_judge(text)
            if judged is not None and judged["is_toxic"]:
                result.is_toxic = True
                result.judged_by = "llm"
                result.categories = judged["categories"] or ["llm_flagged"]
                result.cleaned_text = judged["cleaned_text"] or result.cleaned_text

        return result

    def filter_text(self, text: str) -> str:
        return self.check(text).cleaned_text

    # ------------------------------------------------------------------ #
    @staticmethod
    def _llm_judge(text: str) -> dict | None:
        from ..common.prompt_store import internal_prompt

        template = internal_prompt("toxicity_filter")
        if not template:
            return None

        payload = provider.complete_json(
            template.format(text=text[:4000]), purpose="fast"
        )
        if not isinstance(payload, dict) or "is_toxic" not in payload:
            return None

        categories = payload.get("categories") or []
        return {
            "is_toxic": bool(payload.get("is_toxic")),
            "categories": [str(c) for c in categories] if isinstance(categories, list) else [],
            "cleaned_text": str(payload.get("cleaned_text") or ""),
        }
