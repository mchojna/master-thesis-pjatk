"""
runner
------
Executes agent architectures over a question dataset and streams raw
results to disk. Each (question, architecture) pair runs in a worker
thread; per-record results are written to JSONL/CSV immediately and a
pattern-level summary is materialised at the end of the run.
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd
import structlog
from tqdm.auto import tqdm

from sources.agents.factory import ALL_PATTERNS, build_graph
from sources.agents.state import make_initial_state
from sources.config import config as app_config
from sources.config.graph import GraphConfig
from sources.observer import agent_span

logger = structlog.get_logger(__name__)


METRIC_COLS = list(app_config.evaluation.metric_cols)


class Runner:
    """Execute agent patterns on a question set and save raw results.

    Loads evaluation questions from CSV or JSON, submits each
    (question, pattern) pair to a thread pool, and writes results to
    JSONL, CSV, and JSON files incrementally as they complete.
    """

    def __init__(
        self,
        *,
        questions_path: str | Path | None = None,
        patterns: list[str] | None = None,
        config_overrides: dict[str, Any] | None = None,
        max_workers: int | None = None,
    ) -> None:
        """Initialize the runner and load the question dataset.

        Args:
            questions_path (str | Path | None): Path to the questions file (CSV or JSON).
                Defaults to the first CSV file found in the configured questions directory.
            patterns (list[str] | None): Agent pattern names to evaluate. Defaults to
                ALL_PATTERNS from the agents factory.
            config_overrides (dict[str, Any] | None): Key/value overrides forwarded
                to GraphConfig for each run.
            max_workers (int | None): Maximum number of parallel worker threads.
                Defaults to the AppConfig concurrency setting.
        """
        self.questions_path = self._resolve_questions_path(questions_path)
        self.samples = self._load_question_samples(self.questions_path)
        self.patterns = patterns or ALL_PATTERNS
        self.overrides = config_overrides or {}
        self.max_workers = max_workers or app_config.concurrency.max_workers
        self.results_dir = app_config.paths.results_dir
        self.last_run_id: str | None = None
        self.last_artifacts: dict[str, Path] = {}

    @staticmethod
    def _to_scalar_numeric(value: Any) -> Any:
        """Convert a single-item container to its sole element; leave other values unchanged.

        Args:
            value (Any): Value to normalise.

        Returns:
            Any: The single element if value is a length-1 list or tuple, None if
                value is a multi-element container, or value unchanged otherwise.
        """
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return value[0]
            # Multi-element containers indicate an unexpected aggregate from the graph;
            # return None to surface the anomaly in coerce_metric_columns rather than
            # silently picking one element.
            return None
        return value

    @staticmethod
    def _normalize_optional_text(value: Any) -> str | None:
        """Normalise an optional dataset field to a stripped string or None.

        Args:
            value (Any): Raw value from a dataset row.

        Returns:
            str | None: Stripped string if non-empty, otherwise None.
        """
        if value is None or pd.isna(value):
            return None

        text = str(value).strip()
        return text or None

    @staticmethod
    def _get_usage_bucket(metadata: dict[str, Any], category: str) -> dict[str, Any]:
        """Return one usage bucket from run metadata.

        Args:
            metadata (dict[str, Any]): Metadata dict from a completed run.
            category (str): Usage category key (e.g. "agent_llm", "embedding").

        Returns:
            dict[str, Any]: Usage dict for the requested category, or an empty dict
                if the category is absent.
        """
        return metadata.get("_usage", {}).get(category, {})

    @staticmethod
    def _extract_iteration_count(
        pattern_name: str, metadata: dict[str, Any]
    ) -> int | None:
        """Return the iteration count for one completed pattern run.

        The field used depends on the pattern architecture.

        Args:
            pattern_name (str): Name of the agent pattern.
            metadata (dict[str, Any]): Metadata dict from a completed run.

        Returns:
            int | None: Iteration count if available for the pattern, otherwise None.
        """
        # Each pattern stores its iteration count under a different metadata key because
        # the loop semantics differ: plan steps, blackboard dispatcher ticks, parallel
        # hierarchical workers, or ReAct observation rounds.
        if pattern_name == "planner_executor":
            return metadata.get("_plan_index")
        if pattern_name == "blackboard":
            return metadata.get("_bb_iter")
        if pattern_name == "hierarchical":
            return metadata.get("_process", {}).get("worker_iteration_count")
        if pattern_name == "react":
            return len(metadata.get("_react_observations", []))
        return None

    @staticmethod
    def _serialize_record_value(value: Any) -> Any:
        """Convert nested dicts and lists to a CSV-safe JSON string.

        Args:
            value (Any): Value from a result record.

        Returns:
            Any: JSON string for dicts and lists; the original value otherwise.
        """
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return value

    def _resolve_questions_path(self, questions_path: str | Path | None) -> Path:
        """Resolve the question dataset path.

        Args:
            questions_path (str | Path | None): Explicit path, or None to use the
                first CSV file in the configured questions directory.

        Returns:
            Path: Resolved path to the questions file.

        Raises:
            FileNotFoundError: If questions_path is None and no CSV files are found
                in the configured questions directory.
        """
        if questions_path is None:
            csv_files = sorted(app_config.paths.questions_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(
                    f"No CSV question files in {app_config.paths.questions_dir}"
                )
            return csv_files[0]

        resolved_path = Path(questions_path)
        if not resolved_path.is_absolute():
            resolved_path = (
                app_config.paths.questions_dir.parent.parent.parent / resolved_path
            )
        return resolved_path

    def _load_question_samples(self, path: Path) -> list[dict[str, Any]]:
        """Load evaluation samples from a CSV or JSON file.

        Args:
            path (Path): Path to the questions file. Dispatches to the CSV or JSON
                loader based on the file extension.

        Returns:
            list[dict[str, Any]]: List of sample dicts, each containing at minimum
                a "question" key.
        """
        if path.suffix.lower() == ".csv":
            return self._load_csv_question_samples(path)
        return self._load_json_question_samples(path)

    def _load_csv_question_samples(self, path: Path) -> list[dict[str, Any]]:
        """Load question samples from a semicolon-delimited CSV file.

        Recognises Polish and English column name variants for question, answer,
        context, product, company, and difficulty fields.

        Args:
            path (Path): Path to the CSV file.

        Returns:
            list[dict[str, Any]]: List of sample dicts with normalised field names.

        Raises:
            ValueError: If the CSV contains no recognised question column.
        """
        df = pd.read_csv(path, sep=";")
        df = df.rename(columns={col: col.strip() for col in df.columns})

        question_col = next(
            (col for col in ["Pytanie", "question", "Question"] if col in df.columns),
            None,
        )
        if question_col is None:
            raise ValueError(
                f"CSV file '{path}' does not contain a recognized question column."
            )

        answer_col = next(
            (
                col
                for col in ["Odpowiedź moja", "expected_answer", "answer"]
                if col in df.columns
            ),
            None,
        )
        context_col = next(
            (
                col
                for col in ["OWU pełne", "reference_context", "context"]
                if col in df.columns
            ),
            None,
        )

        samples: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            question = str(row.get(question_col, "")).strip()
            if not question:
                continue

            samples.append(
                {
                    "question": question,
                    "expected_answer": (
                        None
                        if answer_col is None
                        else self._normalize_optional_text(row.get(answer_col))
                    ),
                    "reference_context": (
                        None
                        if context_col is None
                        else self._normalize_optional_text(row.get(context_col))
                    ),
                    "product": (
                        self._normalize_optional_text(row.get("Produkt"))
                        if "Produkt" in df.columns
                        else None
                    ),
                    "company": (
                        self._normalize_optional_text(row.get("Firma"))
                        if "Firma" in df.columns
                        else None
                    ),
                    "difficulty": (
                        self._normalize_optional_text(
                            row.get("Stopień trudności pytania")
                        )
                        if "Stopień trudności pytania" in df.columns
                        else None
                    ),
                }
            )

        return samples

    def _load_json_question_samples(self, path: Path) -> list[dict[str, Any]]:
        """Load question samples from a JSON file.

        Accepts a list of plain strings or a list of dicts with at least a
        "question" key.

        Args:
            path (Path): Path to the JSON file.

        Returns:
            list[dict[str, Any]]: List of sample dicts with normalised field names.

        Raises:
            ValueError: If the JSON structure is not a supported format.
        """
        with path.open(encoding="utf-8") as handle:
            raw_questions = json.load(handle)

        if (
            isinstance(raw_questions, list)
            and raw_questions
            and isinstance(raw_questions[0], str)
        ):
            return [
                {"question": question, "expected_answer": None}
                for question in raw_questions
                if question
            ]

        if (
            isinstance(raw_questions, list)
            and raw_questions
            and isinstance(raw_questions[0], dict)
        ):
            samples: list[dict[str, Any]] = []
            for item in raw_questions:
                question = str(item.get("question", "")).strip()
                if not question:
                    continue
                samples.append(
                    {
                        "question": question,
                        "expected_answer": self._normalize_optional_text(
                            item.get("expected_answer")
                        ),
                        "reference_context": self._normalize_optional_text(
                            item.get("reference_context")
                        ),
                        "product": self._normalize_optional_text(item.get("product")),
                        "company": self._normalize_optional_text(item.get("company")),
                        "difficulty": self._normalize_optional_text(
                            item.get("difficulty")
                        ),
                    }
                )
            return samples

        raise ValueError(f"Unsupported questions format in '{path}'.")

    async def _run_single(
        self,
        question: str,
        pattern_name: str,
        config: GraphConfig,
        *,
        company: str | None = None,
        product: str | None = None,
    ) -> dict[str, Any]:
        """Run one question through one agent pattern and collect result metrics.

        Args:
            question (str): The question to answer.
            pattern_name (str): Agent pattern identifier.
            config (GraphConfig): Graph configuration for this run; closed via
                aclose() in the finally block.
            company (str | None): Optional company filter forwarded to the initial state.
            product (str | None): Optional product filter forwarded to the initial state.

        Returns:
            dict[str, Any]: Flat result record containing the answer, latency,
                citation count, token usage, cost estimates, and full metadata.
        """
        state = make_initial_state(
            question,
            pattern_name,
            company=company,
            product=product,
        )

        start = time.perf_counter()
        try:
            graph = build_graph(config)

            with agent_span(pattern_name, question=question) as span:
                try:
                    result = await graph.ainvoke(state)
                    error = result.get("error")
                except Exception as exc:
                    logger.error(
                        "run_failed",
                        pattern=pattern_name,
                        question=question[:60],
                        error=str(exc),
                    )
                    result = state
                    result["error"] = str(exc)
                    error = str(exc)

                if span is not None:
                    answer_text = result.get("answer") or ""
                    span.set_attribute("output.value", answer_text[:5000])
                    if error:
                        span.set_attribute("error", True)
                        span.set_attribute("status_message", str(error))

            elapsed_ms = (time.perf_counter() - start) * 1000
            citations = result.get("citations", [])
            answer_with_references = (
                result.get("answer_with_references") or result.get("answer") or ""
            )
            answer_body = result.get("answer_body") or answer_with_references
            metadata = result.get("metadata", {})
            process = metadata.get("_process", {})
            agent_llm_usage = self._get_usage_bucket(metadata, "agent_llm")
            embedding_usage = self._get_usage_bucket(metadata, "embedding")
            completion_flag = int(not error and bool(answer_body.strip()))
            error_flag = int(bool(error))

            return {
                "question": question,
                "pattern_name": pattern_name,
                "answer_body": answer_body,
                "answer_with_references": answer_with_references,
                "answer": answer_with_references,
                "error": error,
                "completion_flag": completion_flag,
                "error_flag": error_flag,
                "latency_ms": round(elapsed_ms, 1),
                "citation_count": len(citations),
                "tool_count": process.get("tool_call_count", 0),
                "iteration_count": self._extract_iteration_count(
                    pattern_name, metadata
                ),
                "plan_length": process.get("plan_length"),
                "retrieved_chunk_count": len(result.get("retrieved_chunks", [])),
                "retrieved_chunk_count_total": process.get(
                    "retrieved_chunk_count_total"
                ),
                "retrieval_query_count": process.get("retrieval_query_count"),
                "reranked_chunk_count": sum(
                    1
                    for chunk in result.get("retrieved_chunks", [])
                    if chunk.get("rerank_score") is not None
                ),
                "reranked_chunk_count_total": process.get("reranked_chunk_count_total"),
                "agent_input_tokens_est": agent_llm_usage.get("input_tokens", 0),
                "agent_output_tokens_est": agent_llm_usage.get("output_tokens", 0),
                "embedding_input_tokens_est": embedding_usage.get("input_tokens", 0),
                "estimated_agent_cost_usd": round(
                    float(agent_llm_usage.get("estimated_cost_usd", 0.0)),
                    6,
                ),
                "estimated_embedding_cost_usd": round(
                    float(embedding_usage.get("estimated_cost_usd", 0.0)),
                    6,
                ),
                "estimated_total_cost_usd": round(
                    float(agent_llm_usage.get("estimated_cost_usd", 0.0))
                    + float(embedding_usage.get("estimated_cost_usd", 0.0)),
                    6,
                ),
                "citations": citations,
                "metadata": metadata,
                "usage": metadata.get("_usage", {}),
            }
        finally:
            await config.aclose()

    def _run_single_sync(
        self,
        sample: dict[str, Any],
        pattern_name: str,
    ) -> dict[str, Any]:
        """Run one evaluation sample inside a worker thread.

        Creates a new event loop via asyncio.run and merges dataset fields
        (expected_answer, reference_context, product, company, difficulty)
        into the returned record.

        Args:
            sample (dict[str, Any]): Evaluation sample dict with at least a "question" key.
            pattern_name (str): Agent pattern identifier.

        Returns:
            dict[str, Any]: Completed result record including dataset fields.
        """
        question = sample["question"]
        logger.info("run_start", pattern=pattern_name, question=question[:60])

        config = GraphConfig(pattern_name=pattern_name, **self.overrides)
        record = asyncio.run(
            self._run_single(
                question,
                pattern_name,
                config,
                company=sample.get("company"),
                product=sample.get("product"),
            )
        )
        record["expected_answer"] = sample.get("expected_answer")
        record["reference_context"] = sample.get("reference_context")
        record["product"] = sample.get("product")
        record["company"] = sample.get("company")
        record["difficulty"] = sample.get("difficulty")

        logger.info(
            "run_done",
            pattern=pattern_name,
            latency_ms=record["latency_ms"],
            error=record.get("error"),
        )
        return record

    def _append_result_record(
        self,
        record: dict[str, Any],
        jsonl_handle: Any,
        csv_writer: csv.DictWriter[str] | None,
        csv_handle: Any | None,
    ) -> None:
        """Persist one completed result record to JSONL and optionally CSV.

        Args:
            record (dict[str, Any]): Result record to persist.
            jsonl_handle (Any): Open file handle for the JSONL output file.
            csv_writer (csv.DictWriter[str] | None): Initialised CSV writer, or None
                if the CSV output is not yet set up.
            csv_handle (Any | None): Open file handle for the CSV output file, or None.
        """
        jsonl_handle.write(json.dumps(record, ensure_ascii=False, default=str))
        jsonl_handle.write("\n")
        jsonl_handle.flush()

        if csv_writer is not None and csv_handle is not None:
            csv_writer.writerow(
                {
                    key: self._serialize_record_value(value)
                    for key, value in record.items()
                }
            )
            csv_handle.flush()

    def _coerce_metric_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Convert configured metric columns to numeric values where possible.

        Args:
            dataframe (pd.DataFrame): DataFrame of result records.

        Returns:
            pd.DataFrame: DataFrame with metric columns coerced to numeric dtype.
        """
        existing_cols = [
            column
            for column in app_config.evaluation.metric_cols
            if column in dataframe.columns
        ]
        for column in existing_cols:
            dataframe[column] = pd.to_numeric(
                dataframe[column].map(self._to_scalar_numeric),
                errors="coerce",
            )
        return dataframe

    async def run(self) -> pd.DataFrame:
        """Run all configured patterns on all loaded questions.

        Submits every (sample, pattern) pair to a thread pool, writes results
        incrementally to JSONL and CSV, then writes a final JSON dump and a
        pattern-level summary CSV.

        Returns:
            pd.DataFrame: DataFrame of all result records with metric columns
                coerced to numeric dtype.
        """
        logger.info(
            "experiment_start",
            n_questions=len(self.samples),
            n_patterns=len(self.patterns),
            max_workers=self.max_workers,
        )

        self.results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        jsonl_path = self.results_dir / f"runner_results_{ts}.jsonl"
        csv_results_path = self.results_dir / f"runner_results_{ts}.csv"

        jobs = [
            (sample, pattern) for sample in self.samples for pattern in self.patterns
        ]
        tqdm_kwargs = cast(dict[str, Any], app_config.tqdm.to_kwargs())
        pbar = tqdm(
            total=len(jobs),
            desc="Running",
            unit="run",
            **tqdm_kwargs,
        )

        records: list[dict[str, Any]] = []
        csv_writer: csv.DictWriter[str] | None = None

        with jsonl_path.open(
            "a", encoding="utf-8"
        ) as jsonl_handle, csv_results_path.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as csv_handle:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_job = {
                    executor.submit(self._run_single_sync, sample, pattern): (
                        sample,
                        pattern,
                    )
                    for sample, pattern in jobs
                }

                try:
                    for future in as_completed(future_to_job):
                        record = future.result()
                        records.append(record)

                        if csv_writer is None:
                            # Initialise the CSV writer on the first completed record so
                            # the column schema is derived from actual output fields
                            # rather than locked in at startup — patterns may add or
                            # omit optional columns.
                            csv_writer = csv.DictWriter(
                                csv_handle,
                                fieldnames=list(record.keys()),
                            )
                            if csv_handle.tell() == 0:
                                csv_writer.writeheader()
                                csv_handle.flush()

                        self._append_result_record(
                            record,
                            jsonl_handle,
                            csv_writer,
                            csv_handle,
                        )
                        pbar.set_postfix_str(
                            f"{record['pattern_name']}  {record['latency_ms']:.0f}ms",
                            refresh=False,
                        )
                        pbar.update(1)
                finally:
                    pbar.close()

        json_path = self.results_dir / f"runner_results_{ts}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2, default=str)

        logger.info(
            "results_saved",
            json=str(json_path),
            jsonl=str(jsonl_path),
            csv=str(csv_results_path),
        )

        df = self._coerce_metric_columns(pd.DataFrame(records))
        existing_cols = [
            column
            for column in app_config.evaluation.metric_cols
            if column in df.columns
        ]
        summary = (
            df.groupby("pattern_name")[existing_cols].agg(["mean", "std"]).round(3)
        )
        summary_path = self.results_dir / f"runner_summary_{ts}.csv"
        summary.to_csv(summary_path)

        self.last_run_id = ts
        self.last_artifacts = {
            "json": json_path,
            "jsonl": jsonl_path,
            "csv": csv_results_path,
            "summary": summary_path,
        }
        logger.info("summary_saved", path=str(summary_path))

        return df
