"""
agents
------
Agentic RAG architecture builders and the shared experiment state.
Each pattern lives in its own submodule and exposes a ``build_graph``
callable; the dispatcher in ``factory`` resolves the requested
pattern at runtime.
"""

from sources.agents.factory import build_graph
from sources.agents.state import (
    ExperimentState,
    deduplicate_chunks,
    make_initial_state,
)

__all__ = [
    "ExperimentState",
    "build_graph",
    "deduplicate_chunks",
    "make_initial_state",
]
