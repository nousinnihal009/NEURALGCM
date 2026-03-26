"""
Shared Pydantic schemas used across all routers.
Defines standard envelope types for pagination and errors.
"""

from pydantic import BaseModel, Field
from typing import Generic, TypeVar, List, Optional, Any

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated list wrapper."""
    total: int = Field(..., description="Total number of matching records")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Records per page")
    pages: int = Field(..., description="Total number of pages")
    items: List[T] = Field(..., description="Records for this page")

    @classmethod
    def build(
        cls,
        items: List[T],
        total: int,
        page: int,
        page_size: int,
    ) -> "PaginatedResponse[T]":
        import math
        return cls(
            total=total,
            page=page,
            page_size=page_size,
            pages=math.ceil(total / page_size) if page_size else 1,
            items=items,
        )


class ErrorDetail(BaseModel):
    """Single error detail item."""
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Human-readable error description")
    code: Optional[str] = Field(None, description="Machine-readable error code")


class ErrorResponse(BaseModel):
    """Standard error envelope returned on 4xx/5xx responses."""
    status: int = Field(..., description="HTTP status code")
    error: str = Field(..., description="Short error type label")
    details: List[ErrorDetail] = Field(
        default_factory=list,
        description="Detailed error list (validation failures etc.)")
    request_id: Optional[str] = Field(
        None, description="Request ID for tracing")

    @classmethod
    def from_exception(
        cls,
        status: int,
        message: str,
        code: str = None,
        field: str = None,
    ) -> "ErrorResponse":
        return cls(
            status=status,
            error=message,
            details=[ErrorDetail(field=field, message=message, code=code)],
        )


class HealthStatus(BaseModel):
    """Health / readiness check response."""
    status: str
    checks: dict
    timestamp: str


class MessageResponse(BaseModel):
    """Simple acknowledgement response."""
    message: str
    success: bool = True
