"""
agents.factory
--------------
Looks up the module that implements the requested architecture
pattern and returns its compiled LangGraph. Architectures are
resolved lazily through ``importlib`` so adding a new pattern only
requires a new entry in ``config.patterns.pattern_builders`` and a
module exposing a ``build_graph`` callable.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from sources.config import config as app_config
from sources.config.graph import GraphConfig

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


ALL_PATTERNS: list[str] = list(app_config.patterns.pattern_builders.keys())


def build_graph(config: GraphConfig) -> "CompiledStateGraph":
    """Instantiate and compile the graph for the selected pattern.

    Args:
        config (GraphConfig): Runtime configuration; ``config.pattern_name``
            determines which architecture module is loaded.

    Returns:
        CompiledStateGraph: Compiled LangGraph for the requested pattern.

    Raises:
        ValueError: If ``config.pattern_name`` is not in the known pattern registry.
    """
    pattern_builders = app_config.patterns.pattern_builders
    name = config.pattern_name
    if name not in pattern_builders:
        raise ValueError(
            f"Unknown pattern '{name}'. Choose from: {list(pattern_builders)}"
        )

    module = importlib.import_module(pattern_builders[name])
    return module.build_graph(config)
