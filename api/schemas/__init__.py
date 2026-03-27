"""
NeuralGCM Weather API — Pydantic schemas package.

Import schemas at point-of-use, not here.
Eager imports in __init__.py turn any schema-level syntax error
into a package-level ImportError, making the failure harder to trace.

Usage:
    from api.schemas.forecast import ForecastRequest
    from api.schemas.location import LocationCreate
    from api.schemas.common import PaginatedResponse
"""
