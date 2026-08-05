"""
discharge_ai
============

Agentic AI System for Automated Discharge Summaries (St. Marian Regional
Medical Center capstone).

Layout
------
    settings.py        central config (.env + configs/agent_config.yaml)
    llm/               provider abstraction — Amazon Bedrock (Nova Lite)
    common/            schemas, document loading, rules, terminology
    observability/     LangFuse tracing
    guardrails/        Responsible-AI guardrails
    ehr/               Mock EHR FastAPI service          (:8050)
    mcp_servers/       Primary (:8200) + Analytics (:8201) MCP servers
    mcp_client/        multi-server MCP client (sampling/elicitation/roots)
    a2a/               AgentCards, shared-secret auth, server + client helpers
    agents/            the six A2A agents + host orchestrator
    rag/               five Agno RAG roles + FAISS store
    reporting/         JSON / HTML / PDF report builders
    pipeline.py        in-process orchestration used by the HITL dashboard
"""

__version__ = "1.0.0"
