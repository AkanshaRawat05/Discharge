"""
The six A2A agents plus the host orchestrator.

    monitor_agent.py     Discharge Monitor          Google ADK   :8103
    extractor_agent.py   Clinical Extractor         LangGraph    :8100
    normalizer_agent.py  Clinical Normalizer        LangGraph    :8102
    validator_agent.py   Clinical Validation        LangGraph    :8101
    summary_agent.py     Discharge Summary Gen.     Google ADK   :8104  STREAMING
    rag_agent.py         Clinical RAG Q&A           Agno         :8105  STREAMING
    orchestrator.py      Host Orchestrator          Google ADK   :8083  A2A client
"""
