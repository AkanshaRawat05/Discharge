"""
a2a_layer/cards.py
==================

AgentCard construction.

Every agent publishes its card at `GET /.well-known/agent.json` (the a2a-sdk
0.2.x default) so peers can discover its skills, streaming capability and the
authentication scheme it requires.

All descriptive content is derived from `configs/agent_config.yaml`, so adding
an agent or changing a port never means editing Python.
"""

from __future__ import annotations

from typing import Any

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    APIKeySecurityScheme,
    In,
    SecurityScheme,
)

from ..settings import settings

#  Human-readable skill catalogue, keyed by the `skill_id` in agent_config.yaml.
SKILL_CATALOGUE: dict[str, dict[str, Any]] = {
    "scan_incoming_documents": {
        "name": "Scan incoming discharge documents",
        "description": (
            "Monitor the Roots-scoped input workspace for new discharge reports, "
            "lab reports and hospital bills, and report which patients have a "
            "complete document set ready for processing."
        ),
        "tags": ["monitoring", "mcp-roots", "file-discovery"],
        "examples": [
            "Scan for new discharge cases",
            "Which documents has P1019 submitted?",
        ],
    },
    "extract_clinical_data": {
        "name": "Extract structured clinical data",
        "description": (
            "Extract demographics, diagnoses, prescriptions, lab results and bill "
            "details from multi-format, multi-language discharge documentation."
        ),
        "tags": ["extraction", "multilingual", "multimodal", "mcp-resources"],
        "examples": ["Extract clinical data for P1022"],
    },
    "normalize_clinical_language": {
        "name": "Translate and normalise clinical language",
        "description": (
            "Translate Hindi / Spanish / German / Dutch clinical content into "
            "English via MCP Sampling and expand medical abbreviations, reporting "
            "a translation confidence score."
        ),
        "tags": ["translation", "normalisation", "mcp-sampling"],
        "examples": ["Normalise the Dutch discharge note for P1024"],
    },
    "validate_discharge": {
        "name": "Validate discharge completeness and EHR consistency",
        "description": (
            "Validate mandatory fields against rules.yaml, elicit missing values "
            "from a human reviewer via MCP Elicitation, cross-check the discharge "
            "against the EHR / care plan / labs, and compute the risk tier."
        ),
        "tags": ["validation", "ehr", "risk-scoring", "mcp-elicitation"],
        "examples": ["Validate the discharge for P1021"],
    },
    "generate_discharge_summary": {
        "name": "Generate a patient-friendly discharge summary",
        "description": (
            "Stream a plain-English discharge summary section by section "
            "(patient → medicines → labs → bill → instructions) for auto-approved "
            "or reviewer-cleared cases."
        ),
        "tags": ["summarisation", "streaming", "patient-facing", "mcp-prompts"],
        "examples": ["Generate the discharge summary for P1019"],
    },
    "answer_clinical_question": {
        "name": "Answer questions about patient records",
        "description": (
            "Agentic RAG over indexed discharge documentation with grounded, "
            "citation-bearing answers and RAG Triad quality scoring. Replies "
            "\"I don't know\" when the records do not contain the answer."
        ),
        "tags": ["rag", "question-answering", "streaming", "faiss"],
        "examples": [
            "What medications was P1019 discharged on?",
            "Which patients have an unpaid bill?",
        ],
    },
    "orchestrate_discharge_pipeline": {
        "name": "Orchestrate the discharge pipeline",
        "description": (
            "Coordinate the monitor, extractor, normalizer, validator, summary and "
            "RAG agents over A2A to process a discharge case end to end."
        ),
        "tags": ["orchestration", "a2a-client"],
        "examples": ["Process the discharge for P1023"],
    },
}


def skill_for(skill_id: str) -> AgentSkill:
    """Build an `AgentSkill` from the catalogue (with a safe fallback)."""
    spec = SKILL_CATALOGUE.get(
        skill_id,
        {
            "name": skill_id.replace("_", " ").title(),
            "description": f"Skill {skill_id}",
            "tags": ["clinical"],
            "examples": [],
        },
    )
    return AgentSkill(
        id=skill_id,
        name=spec["name"],
        description=spec["description"],
        tags=spec["tags"],
        examples=spec.get("examples") or None,
    )


def build_agent_card(agent_key: str) -> AgentCard:
    """AgentCard for one agent declared in `agent_config.yaml`."""
    config = settings.agent(agent_key)
    project = settings.cfg.get("project", {})

    #  Shared-secret authentication, advertised so peers know to send the header.
    auth_scheme = SecurityScheme(
        root=APIKeySecurityScheme(
            type="apiKey",
            name=settings.a2a_auth_header,
            in_=In.header,
            description=(
                "Shared secret required on every A2A call. Value comes from "
                "A2A_AUTH_TOKEN in the environment."
            ),
        )
    )

    streaming = bool(config.get("streaming"))
    primitives = ", ".join(config.get("mcp_primitives", []) or [])

    return AgentCard(
        name=config["name"],
        description=(
            f"{SKILL_CATALOGUE.get(config.get('skill_id', ''), {}).get('description', '')} "
            f"Framework: {config.get('framework', 'n/a')}. "
            f"MCP primitives used: {primitives or 'none'}. "
            f"A2A mode: {'streaming' if streaming else 'non-streaming'}."
        ).strip(),
        url=config["url"],
        version=project.get("version", "1.0.0"),
        provider=AgentProvider(
            organization=project.get("hospital", "St. Marian Regional Medical Center"),
            url=settings.service("orchestrator_ui")["url"],
        ),
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=True,
            state_transition_history=True,
        ),
        default_input_modes=["text", "text/plain", "application/json"],
        default_output_modes=["text", "text/plain", "application/json"],
        skills=[skill_for(config["skill_id"])] if config.get("skill_id") else [],
        security_schemes={"agentAuthToken": auth_scheme},
        security=[{"agentAuthToken": []}],
        supports_authenticated_extended_card=False,
    )


def agent_card_dict(agent_key: str) -> dict[str, Any]:
    return build_agent_card(agent_key).model_dump(mode="json", exclude_none=True, by_alias=True)
