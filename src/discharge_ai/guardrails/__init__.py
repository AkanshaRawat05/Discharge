"""
Responsible-AI guardrails (specification Table 12).

    PIIRedactor          mask patient name / phone / Aadhaar / PAN before logging
    HallucinationChecker LLM-as-judge faithfulness gate on RAG answers (<0.70)
    PromptInjectionGuard sanitise or reject prompt-injection attempts
    ToxicityFilter       filter unsafe language out of clinical instructions
    GuardrailManager     orchestration + HITL escalation
"""

from .pii import PIIRedactor  # noqa: F401
from .hallucination import HallucinationChecker  # noqa: F401
from .injection import PromptInjectionGuard  # noqa: F401
from .toxicity import ToxicityFilter  # noqa: F401
from .manager import GuardrailManager, guardrails  # noqa: F401
