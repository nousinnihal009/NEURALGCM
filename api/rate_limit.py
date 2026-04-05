"""
Rate limiter singleton.
========================
Isolated in its own module to break the circular import that occurs
when routers try to import `limiter` from api.main while main.py is
still in the middle of importing those same routers.

Import pattern:
  api/main.py          → from api.rate_limit import limiter
  api/routers/*.py     → from api.rate_limit import limiter
  Nothing imports from api.main except the ASGI entry point.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
)
