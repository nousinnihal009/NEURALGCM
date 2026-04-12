resource "google_sql_database_instance" "postgres" {
  name                = "${var.app_name}-postgres"
  database_version    = "POSTGRES_15"
  region              = var.region
  deletion_protection = false   # set true in real production

  settings {
    tier              = var.db_tier
    availability_type = "ZONAL"   # use REGIONAL for HA in production

    backup_configuration {
      enabled    = true
      start_time = "03:00"
      backup_retention_settings {
        retained_backups = 7
      }
    }

    ip_configuration {
      ipv4_enabled    = false   # private IP only
      private_network = var.network_id
      require_ssl     = false   # set true in production
    }

    database_flags {
      name  = "max_connections"
      value = "100"
    }

    insights_config {
      query_insights_enabled  = true
      record_application_tags = true
    }
  }

  depends_on = [var.private_ip_range]

  lifecycle {
    ignore_changes = [settings[0].disk_size]
  }
}

resource "google_sql_database" "neuralgcm" {
  name     = var.db_name
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "neuralgcm" {
  name     = var.db_user
  instance = google_sql_database_instance.postgres.name
  password = var.db_password
}
