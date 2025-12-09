"""
Telemetry and Tracing module.

This module provides a wrapper around OpenTelemetry to facilitate distributed tracing
and metrics collection. It respects the application's configuration to enable or disable
telemetry.
"""

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Import your settings
from prompt_compiler.config import settings

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class TelemetryManager:
    """
    A wrapper around OpenTelemetry that respects the USE_OPENTEL setting.
    If False, all operations are no-ops and no collectors are initialized.
    """

    _enabled: bool
    _service_name: str
    _tracer: trace.Tracer | None
    _meter: metrics.Meter | None

    def __init__(self) -> None:
        self._enabled = settings.USE_OPENTELEMETRY
        self._service_name = settings.OPENTEL.SERVICE_NAME or "LLM_Prompt_Transpiler"
        self._tracer = None
        self._meter = None

    def setup(self) -> None:
        """
        Initialize the SDK. Call this ONCE at app startup (e.g., main.py).
        If USE_OPENTEL is False, this returns immediately.
        """
        if not self._enabled:
            logger.info("Telemetry disabled. No traces will be sent.")
            return

        try:
            resource = Resource.create(
                attributes={
                    "service.name": self._service_name,
                    "service.version": settings.OPENTEL.SERVICE_VERSION,
                }
            )

            provider = TracerProvider(resource=resource)

            if settings.OPENTEL.OTEL_ENDPOINT:
                # Insecure=True is standard for local sidecars.
                # For remote backends, you may need SSL/headers.
                exporter = OTLPSpanExporter(endpoint=settings.OPENTEL.OTEL_ENDPOINT, insecure=True)
            else:
                exporter = ConsoleSpanExporter()  # type: ignore[assignment]

            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)

            self._tracer = trace.get_tracer(self._service_name)
            self._meter = metrics.get_meter(self._service_name)

            logger.info(f"Telemetry initialized for {self._service_name}")

        except Exception as e:
            logger.error(f"Failed to initialize telemetry: {e}")
            self._enabled = False

    @contextmanager
    def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> Generator[trace.Span | None]:
        """
        Context manager for creating a span.

        Usage:
            with telemetry.span("my_operation", {"user_id": 123}):
                do_work()
        """
        if not self._enabled:
            yield None
            return

        # Uses the global tracer configured in setup()
        tracer = trace.get_tracer(self._service_name)
        with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
            yield span

    def instrument(self, name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """
        Decorator to trace a function automatically.

        Usage:
            @telemetry.instrument()
            def my_func(): ...
        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            span_name = name or func.__name__

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                if not self._enabled:
                    return func(*args, **kwargs)

                with self.span(span_name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_counter(self, name: str, description: str = "", unit: str = "1") -> metrics.Counter:
        """
        Get or create a counter metric.
        If telemetry is disabled, returns a NoOpCounter.
        """
        if not self._enabled or self._meter is None:
            return cast(metrics.Counter, _NoOpCounter())

        return self._meter.create_counter(name, description=description, unit=unit)

    def get_histogram(self, name: str, description: str = "", unit: str = "1") -> metrics.Histogram:
        """Get or create a histogram metric."""
        if not self._enabled or self._meter is None:
            return cast(metrics.Histogram, _NoOpHistogram())

        return self._meter.create_histogram(name, description=description, unit=unit)


class _NoOpCounter:
    """Dummy counter for when telemetry is disabled."""

    def add(self, amount: float | int, attributes: dict[str, Any] | None = None) -> None:
        pass


class _NoOpHistogram:
    """Dummy histogram for when telemetry is disabled."""

    def record(self, amount: float | int, attributes: dict[str, Any] | None = None) -> None:
        pass


# Singleton Instance
telemetry = TelemetryManager()
