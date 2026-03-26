from pydantic import BaseModel
from typing import Optional, Any

class ErrorResponse(BaseModel):
    detail: str
    status_code: int
    error_type: Optional[str] = None

class PaginationParams(BaseModel):
    page: int = 1
    page_size: int = 20

class SuccessResponse(BaseModel):
    status: str = "ok"
    message: str
    data: Optional[Any] = None
