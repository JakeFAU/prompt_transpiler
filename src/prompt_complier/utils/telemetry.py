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
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Import your settings
from prompt_complier.config import settings

logger = logging.getLogger(__name__)


class TelemetryManager:
    _endabled: bool
    _service_name: str
    _tracer: trace.Tracer | None
    _meter: metrics.Meter | None

    """
    A wrapper around OpenTelemetry that respects the USE_OPENTEL setting.
    If False, all operations are no-ops and no collectors are initialized.
    """

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
            # 1. Setup Resource (Service Name, etc.)
            resource = Resource.create(
                attributes={
                    "service.name": self._service_name,
                    "service.version": settings.OPENTEL.SERVICE_VERSION,
                }
            )

            # 2. Setup Trace Provider
            provider = TracerProvider(resource=resource)

            # 3. Configure Exporter (The part that causes connection errors)
            if settings.OPENTEL.OTEL_ENDPOINT:
                # Production: Send to Jaeger/Tempo/Datadog
                exporter = OTLPSpanExporter(endpoint=settings.OPENTEL.OTEL_ENDPOINT, insecure=True)
                processor = BatchSpanProcessor(exporter)
            else:
                # Local Debug: Just print to console if enabled but no endpoint
                exporter = ConsoleSpanExporter()  # type: ignore[assignment]
                processor = BatchSpanProcessor(exporter)

            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)

            # 4. Initialize Objects
            self._tracer = trace.get_tracer(self._service_name)
            self._meter = metrics.get_meter(self._service_name)

            logger.info(f"Telemetry initialized for {self._service_name}")

        except Exception as e:
            logger.error(f"Failed to initialize telemetry: {e}")
            # Fallback to disabled to prevent app crash
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

    def instrument(self, name: str | None = None) -> Callable[..., Any]:
        """
        Decorator to trace a function automatically.
        Usage:
            @telemetry.instrument()
            def my_func(): ...
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            span_name = name or func.__name__

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not self._enabled:
                    return func(*args, **kwargs)

                with self.span(span_name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


# Singleton Instance
telemetry = TelemetryManager()
