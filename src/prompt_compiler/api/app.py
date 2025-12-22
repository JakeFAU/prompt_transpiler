import logging
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path

from apiflask import APIFlask

from prompt_compiler.api.errors import register_error_handlers
from prompt_compiler.api.routes import register_routes
from prompt_compiler.core.registry import ModelRegistry
from prompt_compiler.jobs.service import JobService
from prompt_compiler.jobs.store import DuckDBJobStore, JobStore, MemoryJobStore, SQLiteJobStore
from prompt_compiler.jobs.util import parse_bool_env, parse_int_env
from prompt_compiler.jobs.worker import start_worker
from prompt_compiler.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def create_app() -> APIFlask:
    app = APIFlask(
        __name__,
        title="Prompt Compiler API",
        version=_get_version(),
    )

    app.config["DOCS_PATH"] = "/docs"
    app.config["REDOC_PATH"] = "/redoc"
    app.config["OPENAPI_JSON_PATH"] = "/openapi.json"
    app.config["PROMPT_COMPILER_ENV"] = os.getenv("PROMPT_COMPILER_ENV", "dev")

    _configure_logging()

    registry = ModelRegistry()
    job_store = _create_job_store()
    job_service = JobService(job_store, registry)

    app.extensions["job_service"] = job_service
    app.extensions["model_registry"] = registry

    register_routes(app)
    register_error_handlers(app)

    app.config["TAGS"] = [
        {"name": "meta", "description": "Health and version endpoints."},
        {"name": "compile", "description": "Compile prompts asynchronously."},
        {"name": "jobs", "description": "Manage compile job lifecycle."},
        {"name": "registry", "description": "Model registry and dynamic registration."},
        {"name": "scoring", "description": "Supported scoring strategies."},
    ]

    if parse_bool_env("WORKER_ENABLED", True):
        poll_interval_ms = parse_int_env("WORKER_POLL_INTERVAL_MS", 500)
        concurrency = parse_int_env("WORKER_CONCURRENCY", 1)
        retention = parse_int_env("JOB_RETENTION_HOURS", 24)
        start_worker(
            job_service, poll_interval_ms, concurrency=concurrency, retention_hours=retention
        )
        logger.info("Worker started", concurrency=concurrency, poll_interval_ms=poll_interval_ms)

    return app


def _configure_logging() -> None:
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    configure_logging(level=log_level)


def _create_job_store() -> JobStore:
    job_store = os.getenv("JOB_STORE", "duckdb").lower()
    db_path = os.getenv("JOB_DB_PATH", "/tmp/prompt_compiler_jobs.duckdb")
    if job_store == "sqlite":
        if not db_path.endswith(".sqlite"):
            db_path = "/tmp/prompt_compiler_jobs.sqlite"
        return SQLiteJobStore(db_path)
    if job_store == "memory":
        return MemoryJobStore()
    return DuckDBJobStore(db_path)


def _get_version() -> str:
    try:
        return get_version("prompt-complier")
    except PackageNotFoundError:
        version_file = Path(__file__).parent.parent.parent.parent / "VERSION.txt"
        if version_file.exists():
            return version_file.read_text().strip()
        return "0.0.0"


app = create_app()


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    debug = os.getenv("PROMPT_COMPILER_ENV", "dev").lower() == "dev"
    app.run(host=host, port=port, debug=debug)
