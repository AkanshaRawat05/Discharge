"""
Agentic RAG — the five roles of specification Table 5, all Agno-based.

    indexing.py      Indexing Agent      parse + index documents into FAISS
    retrieval.py     Retrieval Agent     embed the question, fetch top-k chunks
    augmentation.py  Augmentation Agent  re-rank retrieved chunks by keyword relevance
    generation.py    Generation Agent    grounded answer; prompt via MCP Prompts
    reflection.py    Reflection Agent    RAG Triad scoring
"""

from .augmentation import rerank_chunks  # noqa: F401
from .indexing import ClinicalVectorStore, build_index, get_store  # noqa: F401
from .reflection import score_triad  # noqa: F401
from .retrieval import retrieve  # noqa: F401
