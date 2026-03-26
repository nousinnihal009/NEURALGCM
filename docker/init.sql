-- Enable PostGIS extension for spatial queries
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum types
DO $$ BEGIN
    CREATE TYPE forecaststatus AS ENUM
        ('pending','running','complete','failed','cached');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE forecastmode AS ENUM ('realtime','historical');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- The Alembic migration will create the actual tables.
-- This script only sets up extensions.
GRANT ALL PRIVILEGES ON DATABASE neuralgcm_weather TO neuralgcm;
