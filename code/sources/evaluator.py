"""
evaluator
---------
Re-evaluates persisted runner output with the Arize Phoenix evaluation
stack. Computes answer-level metrics (faithfulness, conciseness,
reference correctness), document relevance over saved citation
excerpts, and tool selection/invocation scores; writes per-row metric
files and a per-architecture summary CSV.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from phoenix.evals import LLM, create_classifier
from phoenix.evals.evaluators import async_evaluate_dataframe
from phoenix.evals.metrics import (
    DocumentRelevanceEvaluator,
    FaithfulnessEvaluator,
    ToolInvocationEvaluator,
    ToolSelectionEvaluator,
)
import structlog

from sources.config import config as app_config
from sources.config.prompts import CONCISENESS_PROMPT, REFERENCE_CORRECTNESS_PROMPT
from sources.observer import evaluator_span
from sources.tools import TOOL_DESCRIPTIONS
from sources.tracker import format_tool_invocation

logger = structlog.get_logger(__name__)


class Evaluator:
    """Re-evaluate saved experiment results with the Arize Phoenix evaluation stack.

    Computes answer-level metrics (faithfulness, conciseness, reference correctness),
    document relevance over citation excerpts, and tool selection/invocation scores.
    Writes per-row metric files and a per-architecture summary CSV.
    """

    def __init__(
        self,
        *,
        results_dir: Path | None = None,
        model_name: str | None = None,
        concurrency: int | None = None,
    ) -> None:
        """Initialize the evaluator with optional configuration overrides.

        Args:
            results_dir (Path | None): Directory containing runner result files.
                Defaults to the path configured in AppConfig.
            model_name (str | None): Name of the OpenAI model used for evaluation.
                Defaults to the evaluation model configured in AppConfig.
            concurrency (int | None): Maximum number of concurrent Phoenix evaluation
                requests. Defaults to the AppConfig concurrency setting.
        """
        self.results_dir = results_dir or app_config.paths.results_dir
        self.model_name = model_name or app_config.llm.model_for_role("evaluation")
        self.concurrency = concurrency or app_config.concurrency.max_workers
        self._evaluation_config = app_config.evaluation
        self.eval_llm = LLM(provider="openai", model=self.model_name)
        self._available_tools_text = "\n".join(
            f"{tool_name}: {description}"
            for tool_name, description in TOOL_DESCRIPTIONS.items()
        )

    @staticmethod
    def _find_metric_column(frame: pd.DataFrame, suffix: str) -> str | None:
        """Resolve a Phoenix metric column by matching its suffix.

        Args:
            frame (pd.DataFrame): DataFrame returned by a Phoenix evaluator.
            suffix (str): Column name suffix to search for (e.g. "score", "label").

        Returns:
            str | None: Name of the first matching column, or None if not found.
        """
        if suffix in frame.columns:
            return suffix
        return next((col for col in frame.columns if str(col).endswith(suffix)), None)

    @staticmethod
    def _extract_score_fields(value: Any) -> tuple[Any, Any, Any]:
        """Extract numeric score, label, and explanation from a Phoenix score payload.

        Handles plain scalars, JSON strings, single-element lists, and dict payloads.

        Args:
            value (Any): Raw value from a Phoenix score column.

        Returns:
            tuple[Any, Any, Any]: A (score, label, explanation) triple. Fields absent
                in the input are returned as None.
        """
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    return Evaluator._extract_score_fields(json.loads(stripped))
                except json.JSONDecodeError:
                    pass
            return value, None, None

        if isinstance(value, list) and len(value) == 1:
            return Evaluator._extract_score_fields(value[0])

        if isinstance(value, dict):
            return (
                value.get("score"),
                value.get("label"),
                value.get("explanation"),
            )

        return value, None, None

    @classmethod
    def _has_embedded_metric_payload(cls, series: pd.Series) -> bool:
        """Return whether a Phoenix score column contains structured payload objects.

        Args:
            series (pd.Series): Column from a Phoenix evaluator result DataFrame.

        Returns:
            bool: True if any non-null value is a dict, a list of dicts, or a
                JSON-encoded dict/list string.
        """
        for value in series.dropna():
            if isinstance(value, dict):
                return True
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return True
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    return True
        return False

    @classmethod
    def _rename_metric_result(
        cls,
        frame: pd.DataFrame,
        score_column_name: str,
        label_column_name: str,
        explanation_column_name: str,
        *,
        require_score: bool = True,
    ) -> pd.DataFrame:
        """Normalize Phoenix evaluator output to canonical score/label/explanation column names.

        Args:
            frame (pd.DataFrame): Raw DataFrame from a Phoenix evaluator.
            score_column_name (str): Target column name for the numeric score.
            label_column_name (str): Target column name for the categorical label.
            explanation_column_name (str): Target column name for the text explanation.
            require_score (bool): If True and no score column is found, raise ValueError.

        Returns:
            pd.DataFrame: Copy of frame with renamed metric columns and numeric score coercion.

        Raises:
            ValueError: If require_score is True and the frame contains no score column.
        """
        renamed = frame.copy()
        raw_score_cols = [col for col in frame.columns if str(col).endswith("_score")]
        payload_score_col = next(
            (
                col
                for col in raw_score_cols
                if cls._has_embedded_metric_payload(frame[col])
            ),
            None,
        )
        if payload_score_col is not None:
            extracted_fields = frame[payload_score_col].apply(cls._extract_score_fields)
            extracted_df = pd.DataFrame(
                extracted_fields.tolist(),
                index=frame.index,
                columns=[
                    score_column_name,
                    label_column_name,
                    explanation_column_name,
                ],
            )
            renamed[score_column_name] = pd.to_numeric(
                extracted_df[score_column_name],
                errors="coerce",
            )

            if label_column_name in renamed.columns:
                renamed[label_column_name] = renamed[label_column_name].where(
                    renamed[label_column_name].notna(),
                    extracted_df[label_column_name],
                )
            else:
                renamed[label_column_name] = extracted_df[label_column_name]

            if explanation_column_name in renamed.columns:
                renamed[explanation_column_name] = renamed[
                    explanation_column_name
                ].where(
                    renamed[explanation_column_name].notna(),
                    extracted_df[explanation_column_name],
                )
            else:
                renamed[explanation_column_name] = extracted_df[explanation_column_name]

            return renamed

        score_col = cls._find_metric_column(frame, "score")
        label_col = cls._find_metric_column(frame, "label")
        explanation_col = cls._find_metric_column(frame, "explanation")

        if require_score and score_col is None:
            raise ValueError(
                "Evaluator output does not contain a score column. "
                f"Columns: {list(frame.columns)}"
            )

        rename_map: dict[str, str] = {}
        if score_col is not None:
            rename_map[score_col] = score_column_name
        if label_col is not None:
            rename_map[label_col] = label_column_name
        if explanation_col is not None:
            rename_map[explanation_col] = explanation_column_name

        renamed = renamed.rename(columns=rename_map)
        if score_column_name in renamed.columns:
            renamed[score_column_name] = pd.to_numeric(
                renamed[score_column_name],
                errors="coerce",
            )
        return renamed

    @staticmethod
    def _normalize_citations(value: Any) -> list[dict[str, Any]]:
        """Return citations as a clean list of dictionaries.

        Args:
            value (Any): Raw citations value from a result record.

        Returns:
            list[dict[str, Any]]: List of citation dicts; empty list if value is not a list.
        """
        if not isinstance(value, list):
            return []
        return [citation for citation in value if isinstance(citation, dict)]

    @staticmethod
    def _build_citation_context(citations: list[dict[str, Any]]) -> str:
        """Concatenate citation excerpts into a single Phoenix context string.

        Args:
            citations (list[dict[str, Any]]): List of citation dicts, each expected to
                contain "source_file" and "excerpt" keys.

        Returns:
            str: Newline-separated citation excerpts, with source file prefixed to each.
        """
        parts: list[str] = []
        for citation in citations:
            source_file = str(citation.get("source_file", "")).strip()
            excerpt = str(citation.get("excerpt", "")).strip()
            if not excerpt:
                continue
            parts.append(f"{source_file}: {excerpt}" if source_file else excerpt)
        return "\n\n".join(parts).strip()

    @staticmethod
    def _has_text(value: Any) -> bool:
        """Return whether a value contains non-empty text after stripping whitespace.

        Args:
            value (Any): Value to test; coerced to string before checking.

        Returns:
            bool: True if the stripped string representation is non-empty.
        """
        return bool(str(value or "").strip())

    @classmethod
    def _build_expected_answer_for_eval(
        cls, expected_answer: Any, reference_context: Any
    ) -> str:
        """Combine expected answer and reference context into a single gold-standard string.

        Args:
            expected_answer (Any): Expected answer text, or any falsy value if absent.
            reference_context (Any): Full reference passage, or any falsy value if absent.

        Returns:
            str: Concatenation of non-empty parts separated by a blank line.
        """
        parts: list[str] = []
        if cls._has_text(expected_answer):
            parts.append(str(expected_answer).strip())
        if cls._has_text(reference_context):
            parts.append(str(reference_context).strip())
        return "\n\n".join(parts)

    @staticmethod
    def _metric_columns(
        frame: pd.DataFrame,
        allowed_columns: set[str],
    ) -> list[str]:
        """Return the subset of frame columns present in allowed_columns.

        Args:
            frame (pd.DataFrame): DataFrame whose columns are to be filtered.
            allowed_columns (set[str]): Set of column names to keep.

        Returns:
            list[str]: Ordered list of column names present in both frame and allowed_columns.
        """
        return [column for column in frame.columns if str(column) in allowed_columns]

    def _reference_correctness_evaluator(self) -> Any:
        """Create a Phoenix classifier for reference-based answer correctness.

        Returns:
            Any: Phoenix classifier evaluator configured with REFERENCE_CORRECTNESS_PROMPT
                and the correctness_choices score mapping.
        """
        return create_classifier(
            name="reference_correctness",
            llm=self.eval_llm,
            prompt_template=REFERENCE_CORRECTNESS_PROMPT,
            choices=app_config.evaluation.correctness_choices,
        )

    def _iter_tool_invocations(self, metadata: dict[str, Any]) -> list[dict[str, str]]:
        """Extract normalized tool invocations from one run's metadata.

        Prefers explicit tool_invocations records; falls back to tool_counts when
        invocations are not recorded.

        Args:
            metadata (dict[str, Any]): Metadata dict from a runner result record.

        Returns:
            list[dict[str, str]]: List of dicts with "tool_name" and "invocation" keys.
        """
        process = metadata.get("_process", {}) if isinstance(metadata, dict) else {}
        raw_invocations = process.get("tool_invocations", [])

        normalized_invocations: list[dict[str, str]] = []
        if isinstance(raw_invocations, list):
            for item in raw_invocations:
                if not isinstance(item, dict):
                    continue
                tool_name = str(item.get("tool_name", "")).strip()
                if tool_name not in TOOL_DESCRIPTIONS:
                    continue

                arguments = item.get("arguments")
                invocation = str(item.get("invocation") or "").strip()
                if not invocation:
                    invocation = format_tool_invocation(
                        tool_name,
                        arguments if isinstance(arguments, dict) else None,
                    )
                normalized_invocations.append(
                    {"tool_name": tool_name, "invocation": invocation}
                )

        if normalized_invocations:
            return normalized_invocations

        tool_counts = process.get("tool_counts", {})
        top_k = process.get("last_requested_top_k")
        fallback_invocations: list[dict[str, str]] = []
        if isinstance(tool_counts, dict):
            for tool_name, count in tool_counts.items():
                if tool_name not in TOOL_DESCRIPTIONS:
                    continue
                arguments = (
                    {"top_k": top_k}
                    if tool_name == "retriever" and top_k is not None
                    else None
                )
                invocation = format_tool_invocation(tool_name, arguments)
                for _ in range(max(1, int(count))):
                    fallback_invocations.append(
                        {"tool_name": tool_name, "invocation": invocation}
                    )
        return fallback_invocations

    def latest_results_path(self) -> Path:
        """Return the most recent saved raw results file in results_dir.

        Returns:
            Path: Path to the latest runner_results_*.jsonl file.

        Raises:
            FileNotFoundError: If no matching files exist in results_dir.
        """
        result_files = sorted(self.results_dir.glob("runner_results_*.jsonl"))
        if not result_files:
            raise FileNotFoundError(
                f"No runner_results_*.jsonl files found in {self.results_dir}"
            )
        return result_files[-1]

    def resolve_results_path(self, results_path: str | Path | None = None) -> Path:
        """Resolve a results file path, defaulting to the latest if none is given.

        Args:
            results_path (str | Path | None): Explicit path, relative or absolute.
                When None, delegates to latest_results_path.

        Returns:
            Path: Absolute path to the results file.

        Raises:
            FileNotFoundError: If the resolved path does not exist.
        """
        if results_path is None:
            return self.latest_results_path()

        path = Path(results_path)
        if not path.is_absolute():
            path = self.results_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        return path

    def load_results(
        self, results_path: str | Path | None = None
    ) -> tuple[str, Path, pd.DataFrame]:
        """Load one saved raw results file and prepare it for evaluation.

        Derives answer_text from available answer columns, builds citation_context,
        and adds has_* indicator columns.

        Args:
            results_path (str | Path | None): Path to a JSONL results file, or None to
                use the latest file.

        Returns:
            tuple[str, Path, pd.DataFrame]: A (run_id, resolved_path, results_df) triple
                where run_id is the timestamp extracted from the filename.

        Raises:
            FileNotFoundError: If the results file does not exist.
            ValueError: If the results file contains no records.
        """
        path = self.resolve_results_path(results_path)
        run_id = path.stem.replace("runner_results_", "")

        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        results_df = pd.DataFrame(records)
        results_df.index.name = "row_id"
        results_df = results_df.reset_index()

        if results_df.empty:
            raise ValueError(f"Results file is empty: {path}")

        answer_text = pd.Series(index=results_df.index, data="", dtype=object)
        for column in ["answer_body", "answer_with_references", "answer"]:
            if column not in results_df.columns:
                continue
            answer_text = answer_text.where(
                answer_text.astype(str).str.strip().ne(""),
                results_df[column].fillna(""),
            )
        results_df["answer_text"] = answer_text.fillna("")

        for column in [
            "reference_context",
            "expected_answer",
            "question",
            "pattern_name",
        ]:
            if column not in results_df.columns:
                results_df[column] = ""
            results_df[column] = results_df[column].fillna("")

        results_df["expected_answer_for_eval"] = [
            self._build_expected_answer_for_eval(expected_answer, reference_context)
            for expected_answer, reference_context in zip(
                results_df["expected_answer"],
                results_df["reference_context"],
                strict=False,
            )
        ]

        if "citations" not in results_df.columns:
            results_df["citations"] = [[] for _ in range(len(results_df))]
        results_df["citations"] = results_df["citations"].apply(
            self._normalize_citations
        )
        results_df["citation_context"] = results_df["citations"].apply(
            self._build_citation_context
        )
        results_df["phoenix_context"] = results_df["citation_context"]
        results_df.loc[results_df["phoenix_context"].eq(""), "phoenix_context"] = (
            results_df.loc[results_df["phoenix_context"].eq(""), "reference_context"]
        )
        results_df["has_context"] = (
            results_df["phoenix_context"].astype(str).str.strip().ne("")
        )
        results_df["has_answer"] = (
            results_df["answer_text"].astype(str).str.strip().ne("")
        )
        results_df["has_expected_answer"] = (
            results_df["expected_answer_for_eval"].astype(str).str.strip().ne("")
        )

        return run_id, path, results_df

    async def _evaluate_dataframe(
        self,
        metric_name: str,
        dataframe: pd.DataFrame,
        evaluators: list[Any],
    ) -> pd.DataFrame:
        """Run one Phoenix evaluation batch and restore identifier columns if dropped.

        Args:
            metric_name (str): Display name for the metric, used in logging and spans.
            dataframe (pd.DataFrame): Input DataFrame formatted for Phoenix evaluators.
            evaluators (list[Any]): List of Phoenix evaluator instances to apply.

        Returns:
            pd.DataFrame: Evaluation result DataFrame with identifier columns preserved.
        """
        logger.info(
            "phoenix_metric_start",
            metric=metric_name,
            rows=len(dataframe),
            evaluators=len(evaluators),
        )
        with evaluator_span(metric_name, enabled=app_config.phoenix.enabled) as span:
            if span is not None:
                span.set_attribute("eval.row_count", len(dataframe))
                span.set_attribute("eval.evaluator_count", len(evaluators))

            try:
                result = await async_evaluate_dataframe(
                    dataframe=dataframe,
                    evaluators=evaluators,
                    concurrency=self.concurrency,
                )
            except TypeError:
                result = await async_evaluate_dataframe(
                    dataframe=dataframe,
                    evaluators=evaluators,
                )

            logger.info(
                "phoenix_raw_result",
                metric=metric_name,
                columns=list(result.columns),
                index_type=str(result.index.dtype),
                shape=result.shape,
                sample=result.head(2).to_dict(),
            )

        # Phoenix may drop identifier columns from the returned frame.
        for column in ("row_id", "tool_trace_id"):
            if column in dataframe.columns and column not in result.columns:
                result[column] = dataframe[column].reset_index(drop=True).to_numpy()

        return result

    async def _evaluate_answers(
        self, results_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate faithfulness, conciseness, and reference correctness for each answer row.

        Args:
            results_df (pd.DataFrame): Prepared results DataFrame from load_results.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Per-row metrics DataFrame and a
                pattern-level summary DataFrame with mean scores.
        """
        qa_eval_df = results_df.loc[results_df["has_answer"]].copy()
        qa_eval_df = qa_eval_df[
            [
                "row_id",
                "question",
                "pattern_name",
                "answer_text",
                "phoenix_context",
                "expected_answer_for_eval",
            ]
        ].rename(
            columns={
                "question": "input",
                "answer_text": "output",
                "phoenix_context": "context",
                "expected_answer_for_eval": "expected_answer",
            }
        )

        if qa_eval_df.empty:
            logger.warning("phoenix_answer_eval_skipped", reason="no_answer_rows")
            return pd.DataFrame(), pd.DataFrame(
                columns=["pattern_name", *self._evaluation_config.answer_score_cols]
            )

        faithfulness_evaluator = FaithfulnessEvaluator(llm=self.eval_llm)
        conciseness_evaluator = create_classifier(
            name="conciseness",
            llm=self.eval_llm,
            prompt_template=CONCISENESS_PROMPT,
            choices=app_config.evaluation.conciseness_choices,
        )
        correctness_evaluator = self._reference_correctness_evaluator()
        correctness_input_df = qa_eval_df.loc[
            qa_eval_df["expected_answer"].astype(str).str.strip().ne("")
        ]

        faithfulness_raw = await self._evaluate_dataframe(
            "faithfulness",
            qa_eval_df[
                [
                    "row_id",
                    "input",
                    "output",
                    "context",
                    "pattern_name",
                    "expected_answer",
                ]
            ],
            [faithfulness_evaluator],
        )
        conciseness_raw = await self._evaluate_dataframe(
            "conciseness",
            qa_eval_df[
                ["row_id", "input", "output", "pattern_name", "expected_answer"]
            ],
            [conciseness_evaluator],
        )
        if correctness_input_df.empty:
            logger.warning(
                "phoenix_correctness_eval_skipped",
                reason="no_reference_answers",
            )
            correctness_raw = pd.DataFrame(columns=["row_id"])
        else:
            correctness_raw = await self._evaluate_dataframe(
                "reference_correctness",
                correctness_input_df[
                    [
                        "row_id",
                        "input",
                        "output",
                        "pattern_name",
                        "expected_answer",
                    ]
                ],
                [correctness_evaluator],
            )

        faithfulness_eval_df = self._rename_metric_result(
            faithfulness_raw,
            score_column_name="phoenix_faithfulness_score",
            label_column_name="phoenix_faithfulness_label",
            explanation_column_name="phoenix_faithfulness_explanation",
        )
        conciseness_eval_df = self._rename_metric_result(
            conciseness_raw,
            score_column_name="phoenix_conciseness_score",
            label_column_name="phoenix_conciseness_label",
            explanation_column_name="phoenix_conciseness_explanation",
        )
        correctness_eval_df = self._rename_metric_result(
            correctness_raw,
            score_column_name="phoenix_correctness_score",
            label_column_name="phoenix_correctness_label",
            explanation_column_name="phoenix_correctness_explanation",
            require_score=not correctness_input_df.empty,
        )

        answer_metric_df = (
            qa_eval_df.merge(
                faithfulness_eval_df[
                    self._metric_columns(
                        faithfulness_eval_df,
                        {
                            "row_id",
                            "phoenix_faithfulness_score",
                            "phoenix_faithfulness_label",
                            "phoenix_faithfulness_explanation",
                        },
                    )
                ],
                on="row_id",
                how="left",
            )
            .merge(
                conciseness_eval_df[
                    self._metric_columns(
                        conciseness_eval_df,
                        {
                            "row_id",
                            "phoenix_conciseness_score",
                            "phoenix_conciseness_label",
                            "phoenix_conciseness_explanation",
                        },
                    )
                ],
                on="row_id",
                how="left",
            )
            .merge(
                correctness_eval_df[
                    self._metric_columns(
                        correctness_eval_df,
                        {
                            "row_id",
                            "phoenix_correctness_score",
                            "phoenix_correctness_label",
                            "phoenix_correctness_explanation",
                        },
                    )
                ],
                on="row_id",
                how="left",
            )
        )

        for column in self._evaluation_config.answer_score_cols:
            answer_metric_df[column] = pd.to_numeric(
                answer_metric_df[column], errors="coerce"
            )

        answer_summary_df = (
            answer_metric_df.groupby("pattern_name")[
                self._evaluation_config.answer_score_cols
            ]
            .mean()
            .sort_values("phoenix_correctness_score", ascending=False)
            .reset_index()
        )
        return answer_metric_df, answer_summary_df

    async def _evaluate_documents(
        self, results_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate document relevance for each citation excerpt.

        Args:
            results_df (pd.DataFrame): Prepared results DataFrame from load_results.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Per-citation metrics DataFrame and a
                pattern-level summary DataFrame with mean document relevance scores.
        """
        document_rows: list[dict[str, Any]] = []
        for row in results_df.itertuples(index=False):
            citations = row.citations if isinstance(row.citations, list) else []
            for citation_index, citation in enumerate(citations, start=1):
                if not isinstance(citation, dict):
                    continue
                document_text = str(citation.get("excerpt", "")).strip()
                if not document_text:
                    continue
                document_rows.append(
                    {
                        "row_id": row.row_id,
                        "pattern_name": row.pattern_name,
                        "question": row.question,
                        "citation_index": citation_index,
                        "source_file": citation.get("source_file", ""),
                        "input": row.question,
                        "document_text": document_text,
                    }
                )

        document_eval_input_df = pd.DataFrame(document_rows)
        if document_eval_input_df.empty:
            logger.warning("phoenix_document_eval_skipped", reason="no_citation_rows")
            return pd.DataFrame(), pd.DataFrame(
                columns=["pattern_name", *self._evaluation_config.document_score_cols]
            )

        document_relevance_raw = await self._evaluate_dataframe(
            "document_relevance",
            document_eval_input_df,
            [DocumentRelevanceEvaluator(llm=self.eval_llm)],
        )
        document_metric_df = self._rename_metric_result(
            document_relevance_raw,
            score_column_name="phoenix_document_relevance_score",
            label_column_name="phoenix_document_relevance_label",
            explanation_column_name="phoenix_document_relevance_explanation",
        )
        document_metric_df["phoenix_document_relevance_score"] = pd.to_numeric(
            document_metric_df["phoenix_document_relevance_score"],
            errors="coerce",
        )

        document_summary_df = (
            document_metric_df.groupby("pattern_name")[
                ["phoenix_document_relevance_score"]
            ]
            .mean()
            .sort_values("phoenix_document_relevance_score", ascending=False)
            .reset_index()
        )
        return document_metric_df, document_summary_df

    async def _evaluate_tools(
        self, results_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate tool selection and invocation quality for each recorded tool call.

        Args:
            results_df (pd.DataFrame): Prepared results DataFrame from load_results.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Per-invocation metrics DataFrame and a
                pattern-level summary DataFrame with mean tool scores.
        """
        tool_rows: list[dict[str, Any]] = []
        for row in results_df.itertuples(index=False):
            metadata = row.metadata if isinstance(row.metadata, dict) else {}
            invocations = self._iter_tool_invocations(metadata)
            if not invocations:
                continue

            for invocation_index, invocation in enumerate(invocations, start=1):
                tool_name = invocation["tool_name"]
                tool_call = invocation["invocation"]
                tool_rows.append(
                    {
                        "tool_trace_id": f"{row.row_id}:{invocation_index}",
                        "row_id": row.row_id,
                        "pattern_name": row.pattern_name,
                        "question": row.question,
                        "input": row.question,
                        "available_tools": self._available_tools_text,
                        "tool_selection": tool_name,
                        "tool_invocation": tool_call,
                    }
                )

        tool_eval_input_df = pd.DataFrame(tool_rows)
        if tool_eval_input_df.empty:
            logger.warning("phoenix_tool_eval_skipped", reason="no_explicit_tool_rows")
            return pd.DataFrame(), pd.DataFrame(
                columns=["pattern_name", *self._evaluation_config.tool_score_cols]
            )

        tool_selection_raw = await self._evaluate_dataframe(
            "tool_selection",
            tool_eval_input_df[
                [
                    "tool_trace_id",
                    "row_id",
                    "pattern_name",
                    "question",
                    "input",
                    "available_tools",
                    "tool_selection",
                ]
            ],
            [ToolSelectionEvaluator(llm=self.eval_llm)],
        )
        tool_invocation_input_df = tool_eval_input_df[
            [
                "tool_trace_id",
                "row_id",
                "pattern_name",
                "question",
                "input",
                "available_tools",
                "tool_invocation",
            ]
        ].rename(columns={"tool_invocation": "tool_selection"})
        tool_invocation_raw = await self._evaluate_dataframe(
            "tool_invocation",
            tool_invocation_input_df[
                [
                    "tool_trace_id",
                    "row_id",
                    "pattern_name",
                    "question",
                    "input",
                    "available_tools",
                    "tool_selection",
                ]
            ],
            [ToolInvocationEvaluator(llm=self.eval_llm)],
        )

        tool_selection_df = self._rename_metric_result(
            tool_selection_raw,
            score_column_name="phoenix_tool_selection_score",
            label_column_name="phoenix_tool_selection_label",
            explanation_column_name="phoenix_tool_selection_explanation",
        )
        tool_invocation_df = self._rename_metric_result(
            tool_invocation_raw,
            score_column_name="phoenix_tool_invocation_score",
            label_column_name="phoenix_tool_invocation_label",
            explanation_column_name="phoenix_tool_invocation_explanation",
        )

        tool_metric_df = tool_eval_input_df.merge(
            tool_selection_df[
                self._metric_columns(
                    tool_selection_df,
                    {
                        "tool_trace_id",
                        "phoenix_tool_selection_score",
                        "phoenix_tool_selection_label",
                        "phoenix_tool_selection_explanation",
                    },
                )
            ],
            on=["tool_trace_id"],
            how="left",
        ).merge(
            tool_invocation_df[
                self._metric_columns(
                    tool_invocation_df,
                    {
                        "tool_trace_id",
                        "phoenix_tool_invocation_score",
                        "phoenix_tool_invocation_label",
                        "phoenix_tool_invocation_explanation",
                    },
                )
            ],
            on=["tool_trace_id"],
            how="left",
        )

        for column in self._evaluation_config.tool_score_cols:
            tool_metric_df[column] = pd.to_numeric(
                tool_metric_df[column], errors="coerce"
            )

        tool_run_summary_df = (
            tool_metric_df.groupby(["row_id", "pattern_name"])[
                self._evaluation_config.tool_score_cols
            ]
            .mean()
            .reset_index()
        )
        tool_summary_df = (
            tool_run_summary_df.groupby("pattern_name")[
                self._evaluation_config.tool_score_cols
            ]
            .mean()
            .reset_index()
        )
        sort_col = next(
            (
                col
                for col in self._evaluation_config.tool_score_cols
                if col in tool_summary_df.columns
            ),
            None,
        )
        if sort_col is not None:
            tool_summary_df = tool_summary_df.sort_values(sort_col, ascending=False)
        return tool_metric_df, tool_summary_df

    def _build_summary(
        self,
        answer_summary_df: pd.DataFrame,
        document_summary_df: pd.DataFrame,
        tool_summary_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join per-metric pattern-level summaries into one combined DataFrame.

        Args:
            answer_summary_df (pd.DataFrame): Answer metric summary with a pattern_name column.
            document_summary_df (pd.DataFrame): Document relevance summary with a pattern_name column.
            tool_summary_df (pd.DataFrame): Tool metric summary with a pattern_name column.

        Returns:
            pd.DataFrame: Outer-joined summary with one row per pattern, sorted by pattern_name.
        """
        summary_frames = [
            frame.set_index("pattern_name")
            for frame in [answer_summary_df, document_summary_df, tool_summary_df]
            if not frame.empty and "pattern_name" in frame.columns
        ]
        if not summary_frames:
            return pd.DataFrame(columns=["pattern_name"])

        summary_df = summary_frames[0]
        for frame in summary_frames[1:]:
            summary_df = summary_df.join(frame, how="outer")
        return summary_df.reset_index().sort_values("pattern_name")

    def _save_evaluation_artifacts(
        self,
        *,
        run_id: str,
        results_path: Path,
        results_df: pd.DataFrame,
        answer_metric_df: pd.DataFrame,
        answer_summary_df: pd.DataFrame,
        document_metric_df: pd.DataFrame,
        document_summary_df: pd.DataFrame,
        tool_metric_df: pd.DataFrame,
        tool_summary_df: pd.DataFrame,
        phoenix_summary_df: pd.DataFrame,
        artifact_prefix: str,
    ) -> dict[str, Any]:
        """Write evaluation CSVs to results_dir and return a summary of saved paths.

        Args:
            run_id (str): Timestamp identifier extracted from the source results filename.
            results_path (Path): Path to the source raw results file.
            results_df (pd.DataFrame): Loaded results DataFrame.
            answer_metric_df (pd.DataFrame): Per-row answer metrics.
            answer_summary_df (pd.DataFrame): Pattern-level answer metric means.
            document_metric_df (pd.DataFrame): Per-citation document relevance metrics.
            document_summary_df (pd.DataFrame): Pattern-level document relevance means.
            tool_metric_df (pd.DataFrame): Per-invocation tool metrics.
            tool_summary_df (pd.DataFrame): Pattern-level tool metric means.
            phoenix_summary_df (pd.DataFrame): Combined pattern-level summary.
            artifact_prefix (str): Filename prefix for all output CSVs.

        Returns:
            dict[str, Any]: Dictionary of run metadata, DataFrames, and output file paths.
        """
        answer_metric_path = (
            self.results_dir / f"{artifact_prefix}_answer_metrics_{run_id}.csv"
        )
        document_metric_path = (
            self.results_dir / f"{artifact_prefix}_document_relevance_{run_id}.csv"
        )
        tool_metric_path = (
            self.results_dir / f"{artifact_prefix}_tool_metrics_{run_id}.csv"
        )
        phoenix_summary_path = (
            self.results_dir / f"{artifact_prefix}_summary_{run_id}.csv"
        )

        answer_metric_df.to_csv(answer_metric_path, index=False)
        if not document_metric_df.empty:
            document_metric_df.to_csv(document_metric_path, index=False)
        elif document_metric_path.exists():
            document_metric_path.unlink()

        if not tool_metric_df.empty:
            tool_metric_df.to_csv(tool_metric_path, index=False)
        elif tool_metric_path.exists():
            tool_metric_path.unlink()

        phoenix_summary_df.to_csv(phoenix_summary_path, index=False)

        logger.info(
            "phoenix_eval_saved",
            run_id=run_id,
            artifact_prefix=artifact_prefix,
            answer_metrics=str(answer_metric_path),
            document_metrics=(
                str(document_metric_path) if not document_metric_df.empty else None
            ),
            tool_metrics=str(tool_metric_path) if not tool_metric_df.empty else None,
            summary=str(phoenix_summary_path),
        )

        return {
            "run_id": run_id,
            "results_path": results_path,
            "results_df": results_df,
            "answer_metric_df": answer_metric_df,
            "answer_summary_df": answer_summary_df,
            "document_metric_df": document_metric_df,
            "document_summary_df": document_summary_df,
            "tool_metric_df": tool_metric_df,
            "tool_summary_df": tool_summary_df,
            "phoenix_summary_df": phoenix_summary_df,
            "answer_metric_path": answer_metric_path,
            "document_metric_path": document_metric_path,
            "tool_metric_path": tool_metric_path,
            "phoenix_summary_path": phoenix_summary_path,
        }

    async def evaluate(
        self,
        results_path: str | Path | None = None,
        *,
        artifact_prefix: str = "evaluator",
    ) -> dict[str, Any]:
        """Run answer and document evaluation for one saved run.

        Args:
            results_path (str | Path | None): Path to the JSONL results file, or None to
                use the latest file.
            artifact_prefix (str): Filename prefix for output CSVs.

        Returns:
            dict[str, Any]: Dictionary of DataFrames and output file paths produced
                by _save_evaluation_artifacts.
        """
        run_id, resolved_results_path, results_df = self.load_results(results_path)
        logger.info(
            "phoenix_eval_start",
            results_path=str(resolved_results_path),
            run_id=run_id,
            model=self.model_name,
            rows=len(results_df),
            concurrency=self.concurrency,
            artifact_prefix=artifact_prefix,
        )

        answer_metric_df, answer_summary_df = await self._evaluate_answers(results_df)
        document_metric_df, document_summary_df = await self._evaluate_documents(
            results_df
        )
        tool_metric_df = pd.DataFrame()
        tool_summary_df = pd.DataFrame(
            columns=["pattern_name", *self._evaluation_config.tool_score_cols]
        )
        phoenix_summary_df = self._build_summary(
            answer_summary_df, document_summary_df, tool_summary_df
        )

        return self._save_evaluation_artifacts(
            run_id=run_id,
            results_path=resolved_results_path,
            results_df=results_df,
            answer_metric_df=answer_metric_df,
            answer_summary_df=answer_summary_df,
            document_metric_df=document_metric_df,
            document_summary_df=document_summary_df,
            tool_metric_df=tool_metric_df,
            tool_summary_df=tool_summary_df,
            phoenix_summary_df=phoenix_summary_df,
            artifact_prefix=artifact_prefix,
        )

    async def evaluate_tools_only(
        self,
        results_path: str | Path | None = None,
        *,
        artifact_prefix: str = "evaluator_tools_only",
    ) -> dict[str, Any]:
        """Run tool evaluation only for one saved run and persist tool artifacts.

        Args:
            results_path (str | Path | None): Path to the JSONL results file, or None to
                use the latest file.
            artifact_prefix (str): Filename prefix for output CSVs.

        Returns:
            dict[str, Any]: Dictionary of DataFrames and output file paths produced
                by _save_evaluation_artifacts.
        """
        run_id, resolved_results_path, results_df = self.load_results(results_path)
        logger.info(
            "phoenix_tool_eval_start",
            results_path=str(resolved_results_path),
            run_id=run_id,
            model=self.model_name,
            rows=len(results_df),
            concurrency=self.concurrency,
            artifact_prefix=artifact_prefix,
        )

        tool_metric_df, tool_summary_df = await self._evaluate_tools(results_df)
        answer_metric_df = pd.DataFrame()
        answer_summary_df = pd.DataFrame(
            columns=["pattern_name", *self._evaluation_config.answer_score_cols]
        )
        document_metric_df = pd.DataFrame()
        document_summary_df = pd.DataFrame(
            columns=["pattern_name", *self._evaluation_config.document_score_cols]
        )
        phoenix_summary_df = self._build_summary(
            answer_summary_df, document_summary_df, tool_summary_df
        )

        return self._save_evaluation_artifacts(
            run_id=run_id,
            results_path=resolved_results_path,
            results_df=results_df,
            answer_metric_df=answer_metric_df,
            answer_summary_df=answer_summary_df,
            document_metric_df=document_metric_df,
            document_summary_df=document_summary_df,
            tool_metric_df=tool_metric_df,
            tool_summary_df=tool_summary_df,
            phoenix_summary_df=phoenix_summary_df,
            artifact_prefix=artifact_prefix,
        )
