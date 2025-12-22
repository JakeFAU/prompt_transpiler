"""Error response helpers and APIFlask error handler registration."""

from typing import Any

from apiflask import APIFlask, HTTPError
from marshmallow import ValidationError
from werkzeug.exceptions import HTTPException

from prompt_compiler.utils.logging import get_logger

logger = get_logger(__name__)


def make_error_response(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard API error response payload."""
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        }
    }


def register_error_handlers(app: APIFlask) -> None:
    """Register validation and HTTP exception handlers on the API app."""

    @app.errorhandler(ValidationError)
    def handle_validation_error(error: ValidationError) -> tuple[dict[str, Any], int]:
        details: dict[str, Any]
        if isinstance(error.messages, dict):
            details = error.messages
        else:
            details = {"messages": error.messages}
        return make_error_response("validation_error", "Invalid request payload", details), 400

    @app.errorhandler(HTTPError)
    def handle_http_error(error: HTTPError) -> tuple[dict[str, Any], int]:
        details: dict[str, Any] = {}
        if isinstance(error.detail, dict):
            details = error.detail
        elif error.detail:
            details = {"detail": error.detail}
        message = error.message or "HTTP error"
        payload = make_error_response("http_error", message, details)
        return payload, error.status_code

    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException) -> tuple[dict[str, Any], int]:
        message = error.description or "HTTP error"
        return make_error_response("http_error", message, {}), error.code or 500

    @app.errorhandler(Exception)
    def handle_exception(error: Exception) -> tuple[dict[str, Any], int]:
        logger.error("Unhandled API error", error=str(error))
        return make_error_response("internal_error", "Internal server error", {}), 500
