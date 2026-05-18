"""
tools.prompt_selector
---------------------
Resolves the answer-generation system prompt for the current intent
by looking it up in ``PROMPT_TEMPLATES``. Falls back to the ``unknown``
template when the parsed intent is missing or not registered.
"""

from __future__ import annotations

import structlog

from sources.config.graph import GraphConfig
from sources.config.prompts import PROMPT_TEMPLATES
from sources.agents.state import ExperimentState
from sources.tracker import record_tool_call

logger = structlog.get_logger(__name__)


async def select_prompt(
    state: ExperimentState,
    *,
    config: GraphConfig,
) -> dict:
    """Choose the system prompt template for the current question intent.

    Args:
        state (ExperimentState): Current experiment state; uses the ``intent`` field
            to look up the template.
        config (GraphConfig): Runtime configuration (not used directly but required
            by the uniform tool signature).

    Returns:
        dict: State-update dict with keys ``selected_prompt_template`` (str),
            ``system_prompt`` (str), and ``metadata``.
    """
    intent = state.get("intent") or "unknown"
    metadata = record_tool_call(state.get("metadata"), "prompt_selector")

    template_name = intent if intent in PROMPT_TEMPLATES else "unknown"
    system_prompt = PROMPT_TEMPLATES[template_name]

    logger.info("select_prompt", intent=intent, template=template_name)

    return {
        "selected_prompt_template": template_name,
        "system_prompt": system_prompt,
        "metadata": metadata,
    }
