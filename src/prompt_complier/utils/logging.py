"""
logging.py

One-time structlog initialization with:
- JSON logs (rich metadata) for prod
- Pretty colorized console logs for dev
- Singleton-style init: safe to call configure_logging() many times
"""

import logging
import os
import sys
import threading
from typing import Any, cast

import structlog
from structlog.processors import CallsiteParameterAdder

from prompt_complier.config import settings

# ---- internal singleton state ----

_IS_CONFIGURED = False
_CONFIG_LOCK = threading.Lock()


def _get_log_format_from_env() -> str:
    """
    Decide log format based on env var.

    LOG_FORMAT=JSON    -> JSON logs
    LOG_FORMAT=CONSOLE -> pretty console logs

    Default: CONSOLE
    """
    value = os.getenv("LOG_FORMAT", "CONSOLE").strip().lower()
    if value in {"json", "structured"}:
        return "json"
    return "console"


def configure_logging(
    level: int = settings.LOG_LEVEL,
    log_format: str | None = None,
) -> None:
    """
    Configure structlog + stdlib logging once.

    Args:
        level: root log level (default INFO).
        log_format: "json" or "console". If None, uses LOG_FORMAT env.
    """
    global _IS_CONFIGURED  # noqa: PLW0603

    with _CONFIG_LOCK:
        if _IS_CONFIGURED:
            return

        if log_format is None:
            log_format = _get_log_format_from_env()

        # --- stdlib logging base config ---
        #
        # We let structlog format the message, so the stdlib formatter
        # just needs %(message)s.
        logging.basicConfig(
            level=level,
            format="%(message)s",
            stream=sys.stdout,
        )

        # Basic callsite metadata
        callsite_adder = CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.MODULE,
            ]
        )

        # Shared processors for both console and JSON
        shared_processors = [
            structlog.contextvars.merge_contextvars,   # include contextvars
            structlog.stdlib.add_log_level,           # level
            structlog.stdlib.add_logger_name,         # logger name
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            callsite_adder,                           # filename, lineno, func, module
            structlog.processors.StackInfoRenderer(), # stack_info=True support
            structlog.processors.format_exc_info,     # exc_info to field
        ]

        if log_format == "json":
            # Rich JSON logs: all the metadata
            processors = [
                structlog.stdlib.filter_by_level,
                *shared_processors,
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(sort_keys=True),
            ]
        else:
            # Pretty dev logs: subset of info, colorized
            processors = [
                structlog.stdlib.filter_by_level,
                *shared_processors,
                structlog.dev.ConsoleRenderer(
                    exception_formatter=structlog.dev.rich_traceback,  # nicer tracebacks
                ),
            ]

        structlog.configure(
            processors=processors, # type: ignore[arg-type]
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        _IS_CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Public entrypoint: get a structlog logger.

    Safe to call from anywhere; config happens only once.
    """
    configure_logging()  # idempotent
    if name is None:
        return cast(structlog.stdlib.BoundLogger, structlog.get_logger())
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def set_context(**kwargs: Any) -> None:
    """
    Bind contextvars for all subsequent log entries in this context.

    Example:
        set_context(request_id="abc123", user_id="u-42")
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context(*keys: str) -> None:
    """
    Clear specific context keys, or all if none provided.
    """
    if keys:
        structlog.contextvars.unbind_contextvars(*keys)
    else:
        structlog.contextvars.clear_contextvars()
