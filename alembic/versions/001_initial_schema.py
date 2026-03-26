"""Initial schema: api_keys, locations, forecast_runs

Revision ID: 001
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON
from geoalchemy2 import Geometry
import uuid

revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("api_keys",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  default=uuid.uuid4),
        sa.Column("key_hash", sa.String(255), nullable=False,
                  unique=True, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("rate_limit", sa.String(50), default="60/minute"),
        sa.Column("total_calls", sa.Integer, default=0),
        sa.Column("created_at", sa.DateTime,
                  server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime, nullable=True),
        sa.Column("last_used", sa.DateTime, nullable=True),
    )

    op.create_table("locations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  default=uuid.uuid4),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("lat", sa.Float, nullable=False),
        sa.Column("lon", sa.Float, nullable=False),
        sa.Column("geom", Geometry("POINT", srid=4326), nullable=True),
        sa.Column("country_code", sa.String(3), nullable=True),
        sa.Column("timezone", sa.String(50), nullable=True),
        sa.Column("forecast_count", sa.Integer, default=0),
        sa.Column("created_at", sa.DateTime,
                  server_default=sa.func.now()),
        sa.Column("last_forecast", sa.DateTime, nullable=True),
    )

    op.create_table("forecast_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("celery_id", sa.String(255), nullable=True),
        sa.Column("api_key_id", UUID(as_uuid=True),
                  sa.ForeignKey("api_keys.id"), nullable=True),
        sa.Column("location_name", sa.String(255), nullable=False),
        sa.Column("lat", sa.Float, nullable=False),
        sa.Column("lon", sa.Float, nullable=False),
        sa.Column("geom", Geometry("POINT", srid=4326), nullable=True),
        sa.Column("forecast_days", sa.Integer, default=5),
        sa.Column("init_date", sa.String(50), nullable=True),
        sa.Column("mode", sa.String(20), default="realtime"),
        sa.Column("status", sa.String(20), default="pending",
                  index=True),
        sa.Column("error_msg", sa.Text, nullable=True),
        sa.Column("is_cached", sa.Boolean, default=False),
        sa.Column("cache_key", sa.String(255), nullable=True,
                  index=True),
        sa.Column("created_at", sa.DateTime,
                  server_default=sa.func.now(), index=True),
        sa.Column("started_at", sa.DateTime, nullable=True),
        sa.Column("completed_at", sa.DateTime, nullable=True),
        sa.Column("elapsed_sec", sa.Float, nullable=True),
        sa.Column("result", JSON, nullable=True),
        sa.Column("model_lat", sa.Float, nullable=True),
        sa.Column("model_lon", sa.Float, nullable=True),
        sa.Column("init_time_utc", sa.String(50), nullable=True),
        sa.Column("mode_used", sa.String(20), nullable=True),
        sa.Column("json_path", sa.String(512), nullable=True),
        sa.Column("csv_path", sa.String(512), nullable=True),
        sa.Column("png_path", sa.String(512), nullable=True),
        sa.Column("sanity_ok", sa.Boolean, nullable=True),
        sa.Column("sanity_violations", JSON, nullable=True),
    )


def downgrade():
    op.drop_table("forecast_runs")
    op.drop_table("locations")
    op.drop_table("api_keys")
