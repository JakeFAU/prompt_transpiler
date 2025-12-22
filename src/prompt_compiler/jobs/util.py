"""Shared utilities for job processing and environment parsing."""

import asyncio
import json
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def generate_job_id() -> str:
    """Generate a unique job identifier."""
    return uuid.uuid4().hex


def json_dumps(value: Any) -> str:
    """Serialize a value to compact JSON."""
    return json.dumps(value, default=str, separators=(",", ":"))


def json_loads(value: str | None) -> Any:
    """Deserialize JSON when a non-empty string is provided."""
    if not value:
        return None
    return json.loads(value)


def parse_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean from environment variables."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int_env(name: str, default: int) -> int:
    """Parse an integer from environment variables."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parse_float_env(name: str, default: float) -> float:
    """Parse a float from environment variables."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def sleep_ms(ms: int) -> None:
    """Sleep for the requested number of milliseconds."""
    time.sleep(ms / 1000.0)


def run_coroutine_sync(coro: Any) -> Any:
    """Run an async coroutine from sync code, managing an event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    if loop is None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    return loop.run_until_complete(coro)
