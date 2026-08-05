"""
common/schemas.py
=================

Pydantic models shared by every agent, MCP tool and UI page.  They are the
contract that lets a LangGraph agent, an ADK agent and an Agno agent exchange
the same case object over A2A without guessing each other's field names.

Severity / risk vocabulary comes straight from `configs/rules.yaml`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
#  Enumerations
# --------------------------------------------------------------------------- #
class Severity(str, Enum):
    CRITICAL = "Critical"
    WARNING = "Warning"
    INFO = "Info"


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Recommendation(str, Enum):
    APPROVE = "Approve"
    EDIT = "Edit"
    REJECT = "Reject"


class DocType(str, Enum):
    DISCHARGE_REPORT = "discharge_report"
    LAB_REPORT = "lab_report"
    BILL = "bill"


class ElicitationOutcome(str, Enum):
    ACCEPT = "accept"
    DECLINE = "decline"
    CANCEL = "cancel"
    NOT_REQUESTED = "not_requested"


# --------------------------------------------------------------------------- #
#  Clinical primitives
# --------------------------------------------------------------------------- #
class Medication(BaseModel):
    """One row of the discharge prescription table (spec Table 3)."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    sl_no: int | None = None
    medicine_name: str | None = Field(default=None, alias="name")
    strength: str | None = None
    dosage: str | None = None
    frequency: str | None = None
    route: str | None = None
    period: str | None = None
    remarks: str | None = None
    total_quantity: str | None = None

    def is_blank(self) -> bool:
        return not (self.medicine_name or "").strip()


class LabTest(BaseModel):
    model_config = ConfigDict(extra="allow")

    test: str | None = None
    value: str | None = None
    unit: str | None = None
    reference_range: str | None = None
    flag: str | None = None

    @property
    def is_abnormal(self) -> bool:
        flag = (self.flag or "").strip().lower()
        return flag not in {"", "normal", "सामान्य", "normaal", "normale"}


class AbnormalLab(BaseModel):
    model_config = ConfigDict(extra="allow")

    test: str | None = None
    value: str | None = None
    action: str | None = None


class BillLineItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    item_code: str | None = None
    description: str | None = None
    qty: float | None = None
    unit_price: float | None = None
    total: float | None = None


class Bill(BaseModel):
    model_config = ConfigDict(extra="allow")

    bill_id: str | None = None
    patient_id: str | None = None
    hospital_name: str | None = None
    billing_date: str | None = None
    currency: str | None = None
    line_items: list[BillLineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total_amount: float | None = None
    payment_status: str | None = None
    payment_method: str | None = None
    insurance_guarantee_letter: bool | None = None

    @property
    def is_settled(self) -> bool:
        status = (self.payment_status or "").strip().upper()
        if status in {"PAID", "SETTLED", "BETAALD", "PAGADO", "BEZAHLT"}:
            return True
        return bool(self.insurance_guarantee_letter)


# --------------------------------------------------------------------------- #
#  Extraction output  (Clinical Extractor Agent, :8100)
# --------------------------------------------------------------------------- #
class ExtractedCase(BaseModel):
    """Everything the Extractor pulled out of the three source documents."""

    model_config = ConfigDict(extra="allow")

    # --- demographics -------------------------------------------------------
    patient_id: str | None = None
    patient_name: str | None = None
    dob: str | None = None
    age: int | None = None
    gender: str | None = None
    sex: str | None = None
    address: str | None = None

    # --- encounter ----------------------------------------------------------
    admission_date: str | None = None
    discharge_date: str | None = None
    ward: str | None = None
    bed_no: str | None = None
    service_line: str | None = None
    attending_physician: str | None = None
    consulting_doctors: list[str] = Field(default_factory=list)

    # --- clinical -----------------------------------------------------------
    discharge_diagnosis: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    follow_up_appointment: str | None = None
    discharge_instructions: str | None = None
    discharge_approved: bool | None = None
    discharge_approved_by: str | None = None

    # --- labs ---------------------------------------------------------------
    lab_name: str | None = None
    lab_vendor_name: str | None = None
    lab_report_date: str | None = None
    lab_tests: list[LabTest] = Field(default_factory=list)
    abnormal_labs: list[AbnormalLab] = Field(default_factory=list)

    # --- financial ----------------------------------------------------------
    bill: Bill = Field(default_factory=Bill)

    # --- provenance ---------------------------------------------------------
    detected_language: str = "en"
    source_files: dict[str, str] = Field(default_factory=dict)
    raw_text: dict[str, str] = Field(default_factory=dict)
    doc_types_present: list[str] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)
    extraction_method: str = "deterministic"     # "llm" | "deterministic" | "hybrid"

    # --- normalisation (filled by the Normalizer Agent) ---------------------
    translated_text: dict[str, str] = Field(default_factory=dict)
    translation_confidence: float = 1.0
    translation_notes: list[str] = Field(default_factory=list)
    expanded_abbreviations: list[dict[str, str]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Validation output  (Clinical Validation Agent, :8101)
# --------------------------------------------------------------------------- #
class MissingField(BaseModel):
    field: str
    document: str = "discharge_report"
    blocking: bool = False
    field_type: Literal["str", "int", "float", "bool", "list"] = "str"
    prompt: str = ""
    row: int | None = None            # prescription row number, when applicable


class ValidationFinding(BaseModel):
    """One cross-validation or completeness finding (spec Table 4)."""

    rule_id: str
    severity: Severity = Severity.WARNING
    message: str
    action: str = "Flag for review"
    blocks_discharge: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
    risk_key: str | None = None       # key into rules.yaml risk_scoring_matrix.weights
    clinical_impact: str | None = None
    suggested_action: str | None = None


class CompletenessResult(BaseModel):
    score: float = 100.0
    total_required: int = 0
    present: int = 0
    missing_fields: list[MissingField] = Field(default_factory=list)
    blocking_missing: list[str] = Field(default_factory=list)
    non_blocking_missing: list[str] = Field(default_factory=list)
    prescription_gaps: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def has_blocking(self) -> bool:
        return bool(self.blocking_missing)


class ElicitationRecord(BaseModel):
    """Audit record of one MCP `ctx.elicit()` round-trip."""

    requested_fields: list[MissingField] = Field(default_factory=list)
    outcome: ElicitationOutcome = ElicitationOutcome.NOT_REQUESTED
    reviewer_values: dict[str, Any] = Field(default_factory=dict)
    schema_sent: dict[str, Any] = Field(default_factory=dict)
    note: str = ""
    timestamp: str = Field(default_factory=utc_now_iso)


class RiskAssessment(BaseModel):
    score: int = 0
    level: RiskLevel = RiskLevel.LOW
    recommendation: Recommendation = Recommendation.APPROVE
    recommendation_text: str = ""
    contributions: list[dict[str, Any]] = Field(default_factory=list)
    hard_guardrails_hit: list[str] = Field(default_factory=list)
    discharge_blocked: bool = False
    hitl_required: bool = False


class GuardrailEvent(BaseModel):
    guardrail: str
    triggered: bool
    action: str
    detail: str = ""
    score: float | None = None
    timestamp: str = Field(default_factory=utc_now_iso)


class AuditTrailEntry(BaseModel):
    step: str
    actor: str                        # agent / tool / guardrail name
    framework: str = ""
    status: str = "ok"
    detail: str = ""
    duration_ms: int | None = None
    langfuse_trace_id: str | None = None
    timestamp: str = Field(default_factory=utc_now_iso)


class ValidationReport(BaseModel):
    """The JSON artefact consumed by the dashboard, HTML/PDF renderer and RAG."""

    model_config = ConfigDict(extra="allow")

    patient_id: str
    patient_name: str | None = None
    generated_at: str = Field(default_factory=utc_now_iso)
    rules_version: str = ""
    trace_id: str | None = None

    completeness: CompletenessResult = Field(default_factory=CompletenessResult)
    findings: list[ValidationFinding] = Field(default_factory=list)
    risk: RiskAssessment = Field(default_factory=RiskAssessment)
    elicitation: ElicitationRecord = Field(default_factory=ElicitationRecord)
    guardrail_events: list[GuardrailEvent] = Field(default_factory=list)
    audit_trail: list[AuditTrailEntry] = Field(default_factory=list)

    translation_confidence: float = 1.0
    detected_language: str = "en"
    bill_total_amount: float | None = None
    bill_currency: str | None = None
    bill_payment_status: str | None = None
    ehr_verdict: str = ""

    analytics: dict[str, Any] = Field(default_factory=dict)
    hitl_feedback: dict[str, Any] = Field(default_factory=dict)

    @property
    def critical_findings(self) -> list[ValidationFinding]:
        return [f for f in self.findings if f.severity == Severity.CRITICAL]


# --------------------------------------------------------------------------- #
#  Summary output  (Discharge Summary Generator, :8104)
# --------------------------------------------------------------------------- #
class SummarySection(BaseModel):
    key: str
    title: str
    content: str = ""


class DischargeSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    patient_id: str
    patient_name: str | None = None
    audience: str = "patient"
    risk_level: RiskLevel = RiskLevel.LOW
    sections: list[SummarySection] = Field(default_factory=list)
    prescription_table: list[dict[str, Any]] = Field(default_factory=list)
    lab_table: list[dict[str, Any]] = Field(default_factory=list)
    bill_snapshot: dict[str, Any] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=utc_now_iso)
    trace_id: str | None = None
    guardrail_events: list[GuardrailEvent] = Field(default_factory=list)

    def as_markdown(self) -> str:
        parts = [f"# Discharge Summary — {self.patient_name or self.patient_id}", ""]
        for section in self.sections:
            parts += [f"## {section.title}", section.content.strip(), ""]
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
#  RAG output  (Agno Clinical RAG Q&A Agent, :8105)
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    text: str
    source: str
    patient_id: str | None = None
    doc_type: str | None = None
    score: float = 0.0
    rerank_score: float | None = None
    chunk_index: int = 0


class RagTriad(BaseModel):
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_relevance: float = 0.0
    reasoning: str = ""
    passed: bool = True


class RagAnswer(BaseModel):
    question: str
    answer: str = ""
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    triad: RagTriad = Field(default_factory=RagTriad)
    guardrail_events: list[GuardrailEvent] = Field(default_factory=list)
    prompt_source: str = "mcp://clinical-tools/prompts/rag-answer-prompt"
    trace_id: str | None = None
    out_of_context: bool = False


# --------------------------------------------------------------------------- #
#  Case bundle — what flows between agents over A2A
# --------------------------------------------------------------------------- #
class DischargeCase(BaseModel):
    """End-to-end state for one patient discharge."""

    model_config = ConfigDict(extra="allow")

    patient_id: str
    trace_id: str | None = None
    stage: str = "detected"
    extracted: ExtractedCase = Field(default_factory=lambda: ExtractedCase())
    report: ValidationReport | None = None
    summary: DischargeSummary | None = None
    errors: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: str | None = None
