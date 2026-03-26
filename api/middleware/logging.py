"""
Request/Response Logging Middleware
===================================
Structured logging for all API requests.
"""

import time
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        logger.info(
            f"→ {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}")

        response = await call_next(request)

        # Log response
        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"← {request.method} {request.url.path} "
            f"→ {response.status_code} ({elapsed:.0f}ms)")

        response.headers["X-Process-Time"] = f"{elapsed:.0f}ms"
        return response
