"""
tools.answer_synthesizer
------------------------
Generates the final Polish answer from the selected evidence and
citations. Calls the synthesis LLM with a structured context block,
then normalises the answer body (two-section format, inline citation
placeholders, fake clause references) and appends a deterministic
references footer.
"""

from __future__ import annotations

import json
import re

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from sources.config import config as app_config
from sources.config.graph import GraphConfig
from sources.config.prompts import CONTEXT_INSTRUCTION, FAQ_PROMPT
from sources.agents.state import ExperimentState, deduplicate_chunks
from sources.tracker import (
    estimate_message_tokens,
    estimate_text_tokens,
    record_model_usage,
    record_tool_call,
)

logger = structlog.get_logger(__name__)

# [N] and [?] are placeholder citation tokens the LLM emits when it doesn't
# know the actual reference number; they are replaced with real inline_ref values.
_PLACEHOLDER_CITATION_RE = re.compile(r"\[(?:N|\?)\]")
# § X ust. Y (pkt Z) is a fake Polish legal citation pattern the model tends to
# hallucinate; these are stripped entirely because no real clause maps to them.
_PLACEHOLDER_CLAUSE_PATTERNS = [
    re.compile(r"\(\s*§\s*X\s+ust\.\s*Y(?:\s+pkt\s+Z)?\s*\)"),
    re.compile(r"§\s*X\s+ust\.\s*Y(?:\s+pkt\s+Z)?"),
]


def _build_context_block(state: ExperimentState) -> str:
    """Build the context payload sent to the answer model.

    Args:
        state (ExperimentState): Current experiment state containing evidence chunks,
            citations, question, company, product, and intent fields.

    Returns:
        str: Formatted multi-section string ready to be used as the human message
            for the synthesis LLM.
    """
    chunks = state.get("evidence_chunks") or state.get("retrieved_chunks", [])
    citations = state.get("citations", [])

    context_lines: list[str] = []
    for i, chunk in enumerate(chunks):
        ref = f"[{i + 1}]"
        src = chunk.get("source_file", "unknown")
        meta = chunk.get("metadata", {})
        header = meta.get("header_2") or meta.get("header_1") or ""
        text = chunk.get("text", "")
        context_lines.append(f"Source {ref} — {src} / {header}:\n{text}\n")

    context_block = (
        "\n".join(context_lines) if context_lines else "(no context retrieved)"
    )

    return (
        f"QUESTION: {state.get('question', '')}\n"
        f"COMPANY: {state.get('company') or 'Not specified'}\n"
        f"PRODUCT: {state.get('product') or 'Not specified'}\n"
        f"INTENT: {state.get('intent') or 'unknown'}\n\n"
        f"SELECTED EVIDENCE:\n{context_block}\n\n"
        f"CITATIONS AVAILABLE: {json.dumps(citations, ensure_ascii=False)}\n\n"
        f"{CONTEXT_INSTRUCTION}"
    )


def build_references_section(citations: list[dict]) -> str:
    """Render a references footer from citation objects.

    Args:
        citations (list[dict]): Citation dicts, each with ``inline_ref``,
            ``source_file``, and ``excerpt`` keys.

    Returns:
        str: Markdown references footer string, or an empty string when the
            citations list is empty.
    """
    if not citations:
        return ""
    lines = ["\n\n---\n**Źródła:**"]
    for cit in citations:
        ref = cit.get("inline_ref", "")
        src = cit.get("source_file", "")
        excerpt = cit.get("excerpt", "")[:100]
        lines.append(f'- {ref} {src} — "{excerpt}..."')
    return "\n".join(lines)


def _strip_existing_references(answer_body: str) -> str:
    """Remove any model-generated references footer to avoid duplicates.

    Args:
        answer_body (str): Raw answer text that may contain a trailing references
            section.

    Returns:
        str: Answer text with any ``Źródła`` / ``Sources`` footer removed.
    """
    text = (answer_body or "").rstrip()
    return re.split(
        r"\n\s*(?:---\s*)?\*\*(?:Źródła|Zrodla|Sources):\*\*", text, maxsplit=1
    )[0].rstrip()


def attach_references(answer_body: str, citations: list[dict]) -> str:
    """Append a references footer to a plain answer body.

    Args:
        answer_body (str): Answer text without a references footer.
        citations (list[dict]): Citation objects to render as the footer.

    Returns:
        str: Answer text with the references footer appended.
    """
    cleaned_answer = _strip_existing_references(answer_body)
    return cleaned_answer + build_references_section(citations)


def _normalize_inline_citations(answer_body: str, citations: list[dict]) -> str:
    """Replace placeholder citation markers and strip fake clause references.

    Args:
        answer_body (str): Raw answer text that may contain ``[N]``, ``[?]``, or
            ``§ X ust. Y`` placeholder patterns.
        citations (list[dict]): Available citation objects used to supply the
            fallback inline reference.

    Returns:
        str: Cleaned answer text with valid inline references and no stray placeholders.
    """
    text = (answer_body or "").strip()
    if not text or not citations:
        for pattern in _PLACEHOLDER_CLAUSE_PATTERNS:
            text = pattern.sub("", text)
        return re.sub(r"\s{2,}", " ", text)

    citation_refs = [str(cit.get("inline_ref", "")).strip() for cit in citations]
    citation_refs = [ref for ref in citation_refs if ref]
    fallback_ref = citation_refs[0] if citation_refs else "[1]"

    text = _PLACEHOLDER_CITATION_RE.sub(fallback_ref, text)
    for pattern in _PLACEHOLDER_CLAUSE_PATTERNS:
        text = pattern.sub("", text)

    # Clean up spacing left behind after removing placeholders.
    text = re.sub(r"\(\s*([[]\d+[]])\s*\)", r"\1", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def _normalize_answer_structure(answer_body: str) -> str:
    """Enforce a stable two-section answer format.

    Args:
        answer_body (str): Raw answer text from the LLM, possibly missing the
            required ``**Odpowiedź:**`` and ``**Szczegóły:**`` section headers.

    Returns:
        str: Answer text guaranteed to contain both required section headers
            separated by a Markdown horizontal rule.
    """
    text = (answer_body or "").strip()
    if not text:
        return "**Odpowiedź:** Brak odpowiedzi.\n\n---\n\n**Szczegóły:**\n- Brak dodatkowych szczegółów."

    if text == app_config.answer_synthesizer.insufficient_context_msg:
        return text

    # Normalise common LLM misspellings of the Polish section header so that the
    # two-section structure check below finds the expected bold marker reliably.
    text = re.sub(r"^\s*Odzpowiedź\s*:", "**Odpowiedź:**", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*Odpowiedz\s*:", "**Odpowiedź:**", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*Odpowiedź\s*:", "**Odpowiedź:**", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^\s*Szczegóły\s*:", "**Szczegóły:**", text, flags=re.IGNORECASE | re.MULTILINE
    )

    if "**Odpowiedź:**" not in text:
        return (
            f"**Odpowiedź:** {text}\n\n---\n\n"
            "**Szczegóły:**\n"
            "- Brak dodatkowych szczegółów."
        )

    if "**Szczegóły:**" not in text:
        return f"{text}\n\n---\n\n" "**Szczegóły:**\n" "- Brak dodatkowych szczegółów."

    text = re.sub(
        r"(\*\*Odpowiedź:\*\*.*?)(?:\n+\s*---\s*)?\n+\s*(\*\*Szczegóły:\*\*)",
        r"\1\n\n---\n\n\2",
        text,
        count=1,
        flags=re.DOTALL,
    )

    return text


async def synthesize_answer(
    state: ExperimentState,
    *,
    config: GraphConfig,
) -> dict:
    """Invoke the synthesis LLM and return the final answer payload.

    Args:
        state (ExperimentState): Current experiment state containing evidence chunks,
            citations, system prompt, question, company, and product fields.
        config (GraphConfig): Runtime configuration providing the synthesis LLM client
            and model name.

    Returns:
        dict: State-update dict with keys ``answer_body``, ``answer_with_references``,
            ``answer``, and ``metadata``. On failure, ``answer`` is ``None`` and
            ``error`` contains the exception message.
    """
    system_prompt = state.get("system_prompt") or FAQ_PROMPT
    user_message = _build_context_block(state)
    citations = state.get("citations", [])
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    metadata = record_tool_call(state.get("metadata"), "answer_synthesizer")

    try:
        response = await config.synthesis_llm.ainvoke(messages)
        raw_answer_body = response.content if isinstance(response.content, str) else ""
        answer_body = _normalize_answer_structure(raw_answer_body)
        answer_body = _normalize_inline_citations(answer_body, citations)
        answer_text = attach_references(answer_body, citations)
        metadata = record_model_usage(
            metadata,
            category="agent_llm",
            model_name=config.synthesis_model,
            input_tokens=estimate_message_tokens(messages, config.synthesis_model),
            output_tokens=estimate_text_tokens(answer_body, config.synthesis_model),
        )

        logger.info(
            "synthesize_answer",
            answer_len=len(answer_text),
            n_citations=len(citations),
        )

        return {
            "answer_body": answer_body,
            "answer_with_references": answer_text,
            "answer": answer_text,
            "metadata": metadata,
        }

    except Exception as exc:
        logger.error("synthesize_answer_failed", error=str(exc))
        return {
            "answer_body": None,
            "answer_with_references": None,
            "answer": None,
            "error": str(exc),
            "metadata": metadata,
        }
