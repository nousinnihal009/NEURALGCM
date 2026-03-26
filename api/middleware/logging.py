"""
Request / Response Logging Middleware
======================================
Structured JSON-compatible request logging via loguru.
Logs: method, path, status code, elapsed ms, client IP, request size.
Adds X-Request-ID header to every response for tracing.
"""

import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from loguru import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured logging middleware.
    Every request gets a UUID request_id attached to response headers
    and included in the log line for distributed tracing.
    """

    # Paths to skip (health / readiness spam)
    SKIP_PATHS = {"/health", "/ready", "/metrics/cache"}

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        # Attach request_id so downstream handlers can reference it
        request.state.request_id = request_id

        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # Log incoming request
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or getattr(request.client, "host", "unknown")
        )
        logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=client_ip,
        ).debug("→ incoming request")

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.bind(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                elapsed_ms=round(elapsed_ms, 1),
            ).error(f"Unhandled exception: {exc}")
            raise

        elapsed_ms = (time.perf_counter() - t0) * 1000

        log = logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=round(elapsed_ms, 1),
            client=client_ip,
        )

        if response.status_code >= 500:
            log.error("← response")
        elif response.status_code >= 400:
            log.warning("← response")
        else:
            log.info("← response")

        response.headers["X-Request-ID"] = request_id
        return response
