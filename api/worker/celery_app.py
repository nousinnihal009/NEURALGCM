import os
os.environ["JAX_PLATFORMS"]                 = "cpu"
os.environ["XLA_FLAGS"]                     = "--xla_cpu_use_thunk_runtime=false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from celery import Celery
from api.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "neuralgcm_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["api.worker.tasks"],
)

celery_app.conf.update(
    task_serializer        = "json",
    accept_content         = ["json"],
    result_serializer      = "json",
    timezone               = "UTC",
    enable_utc             = True,
    task_soft_time_limit   = settings.celery_task_timeout,
    task_time_limit        = settings.celery_task_timeout + 60,
    worker_prefetch_multiplier = 1,   # one task at a time per worker
    task_acks_late         = True,    # ack after completion, not before
    result_expires         = 86400,   # keep results 24h
)
