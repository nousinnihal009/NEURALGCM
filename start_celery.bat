@echo off
echo ========================================================
echo Starting Celery Worker (Windows Compatible)
echo ========================================================
set PYTHONPATH=%cd%
echo PYTHONPATH set to: %PYTHONPATH%
celery -A api.worker.celery_app worker --loglevel=info -P solo
