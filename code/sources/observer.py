"""
observer
--------
Thin wrapper around Phoenix/OpenTelemetry tracing used by both the
agent runner and the evaluator. Exposes a singleton tracing service
that lazily registers the project once and provides scoped span
context managers for agent runs and evaluator metric calls.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry.trace import Tracer
from phoenix.otel import register
import structlog

from sources.config import config as app_config

logger = structlog.get_logger(__name__)


class Observer:
    """Manage Phoenix tracing lifecycle and provide span context managers."""

    def __init__(self) -> None:
        """Initialize the tracing service in an uninitialized state."""
        self._initialized = False
        self._tracer_provider = None

    def init_tracing(self) -> None:
        """Register the Phoenix tracer provider once.

        Does nothing if tracing has already been initialized or is disabled in
        the configuration.
        """
        if self._initialized:
            logger.info("tracing_already_initialized")
            return

        phoenix_cfg = app_config.phoenix
        if not phoenix_cfg.enabled:
            logger.info("tracing_disabled_by_config")
            return

        self._tracer_provider = register(
            project_name=phoenix_cfg.project_name,
            endpoint=phoenix_cfg.endpoint,
            auto_instrument=True,
        )
        self._initialized = True
        logger.info("tracing_initialized", endpoint=phoenix_cfg.endpoint)

    def get_tracer(self) -> Tracer | None:
        """Return the configured tracer, initializing tracing if necessary.

        Returns:
            Tracer | None: The OpenTelemetry tracer, or None if tracing is
                disabled or initialization has not succeeded.
        """
        if not self._initialized and app_config.phoenix.enabled:
            self.init_tracing()

        if not self._initialized or self._tracer_provider is None:
            return None

        return self._tracer_provider.get_tracer(__name__)

    @contextmanager
    def agent_span(
        self,
        pattern_name: str,
        question: str | None = None,
    ) -> Iterator[Any | None]:
        """Create the root span for one agent run.

        Yields None when tracing is disabled or unavailable.

        Args:
            pattern_name (str): Agent pattern identifier, used as the span name.
            question (str | None): Input question to record as a span attribute.

        Yields:
            Any | None: The active span, or None if tracing is unavailable.
        """
        tracer = self.get_tracer()
        if tracer is None:
            yield None
            return

        with tracer.start_as_current_span(pattern_name) as span:
            span.set_attribute("openinference.span.kind", "AGENT")
            span.set_attribute("agent.name", pattern_name)
            if question is not None:
                span.set_attribute("input.value", question)
            yield span

    @contextmanager
    def evaluator_span(
        self,
        metric_name: str,
        question: str | None = None,
        *,
        enabled: bool = True,
    ) -> Iterator[Any | None]:
        """Create a span for one evaluator metric call.

        Yields None when tracing is disabled, not enabled for this call,
        or unavailable.

        Args:
            metric_name (str): Metric identifier, used as part of the span name.
            question (str | None): Input question to record as a span attribute.
            enabled (bool): If False, tracing is skipped for this call regardless
                of the global configuration.

        Yields:
            Any | None: The active span, or None if tracing is unavailable.
        """
        if not enabled:
            yield None
            return

        tracer = self.get_tracer()
        if tracer is None:
            yield None
            return

        with tracer.start_as_current_span(f"evaluator.{metric_name}") as span:
            span.set_attribute("openinference.span.kind", "EVALUATOR")
            span.set_attribute("eval.metric_name", metric_name)
            if question is not None:
                span.set_attribute("input.value", question)
            yield span


_OBSERVER = Observer()


def init_tracing() -> None:
    """Register the Phoenix tracer provider for the default tracing service."""
    _OBSERVER.init_tracing()


def get_tracer() -> Tracer | None:
    """Return the tracer from the default tracing service.

    Returns:
        Tracer | None: The OpenTelemetry tracer, or None if tracing is
            disabled or unavailable.
    """
    return _OBSERVER.get_tracer()


@contextmanager
def agent_span(pattern_name: str, question: str | None = None) -> Iterator[Any | None]:
    """Create the root span for one agent run using the default tracing service.

    Args:
        pattern_name (str): Agent pattern identifier, used as the span name.
        question (str | None): Input question to record as a span attribute.

    Yields:
        Any | None: The active span, or None if tracing is unavailable.
    """
    with _OBSERVER.agent_span(pattern_name, question=question) as span:
        yield span


@contextmanager
def evaluator_span(
    metric_name: str,
    question: str | None = None,
    *,
    enabled: bool = True,
) -> Iterator[Any | None]:
    """Create a span for one evaluator metric call using the default tracing service.

    Args:
        metric_name (str): Metric identifier, used as part of the span name.
        question (str | None): Input question to record as a span attribute.
        enabled (bool): If False, tracing is skipped for this call regardless
            of the global configuration.

    Yields:
        Any | None: The active span, or None if tracing is unavailable.
    """
    with _OBSERVER.evaluator_span(
        metric_name,
        question=question,
        enabled=enabled,
    ) as span:
        yield span
