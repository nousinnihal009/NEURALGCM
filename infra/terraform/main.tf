terraform {
  required_version = ">= 1.7.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.20"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.20"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Remote state in GCS — run bootstrap.sh first to create the bucket
  backend "gcs" {
    bucket = "neuralgcm-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# ── Enable required APIs ──────────────────────────────────────
resource "google_project_service" "apis" {
  for_each = toset([
    "container.googleapis.com",        # GKE
    "sqladmin.googleapis.com",         # Cloud SQL
    "redis.googleapis.com",            # Memorystore
    "artifactregistry.googleapis.com", # Artifact Registry
    "run.googleapis.com",              # Cloud Run
    "secretmanager.googleapis.com",    # Secret Manager
    "cloudscheduler.googleapis.com",   # Cloud Scheduler
    "monitoring.googleapis.com",       # Cloud Monitoring
    "logging.googleapis.com",          # Cloud Logging
    "cloudresourcemanager.googleapis.com",
    "compute.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "servicenetworking.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# ── Service Account for GKE workloads ────────────────────────
resource "google_service_account" "neuralgcm_sa" {
  account_id   = "${var.app_name}-sa"
  display_name = "NeuralGCM Workload Service Account"
  depends_on   = [google_project_service.apis]
}

# Permissions: read ERA5 GCS, read NeuralGCM checkpoints GCS,
# write forecast-output GCS, access Secret Manager, write metrics
locals {
  sa_roles = [
    "roles/storage.objectViewer",         # read ERA5 + checkpoints
    "roles/storage.objectAdmin",          # write forecast outputs
    "roles/secretmanager.secretAccessor", # read secrets
    "roles/monitoring.metricWriter",      # write custom metrics
    "roles/logging.logWriter",            # write logs
    "roles/cloudtrace.agent",            # distributed tracing
  ]
}
resource "google_project_iam_member" "sa_roles" {
  for_each = toset(local.sa_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.neuralgcm_sa.email}"
}

# ── Workload Identity binding ─────────────────────────────────
# Allows Kubernetes pods to impersonate the GCP service account
# without mounting JSON keys — GCP best practice for security
resource "google_service_account_iam_member" "workload_identity" {
  service_account_id = google_service_account.neuralgcm_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[neuralgcm/neuralgcm-sa]"
  depends_on         = [module.gke]
}

# ── Modules ───────────────────────────────────────────────────
module "networking" {
  source     = "./modules/networking"
  project_id = var.project_id
  region     = var.region
  app_name   = var.app_name
  depends_on = [google_project_service.apis]
}

module "registry" {
  source     = "./modules/registry"
  project_id = var.project_id
  region     = var.region
  app_name   = var.app_name
  depends_on = [google_project_service.apis]
}

module "secrets" {
  source      = "./modules/secrets"
  project_id  = var.project_id
  db_password = var.db_password
  depends_on  = [google_project_service.apis]
}

module "storage" {
  source     = "./modules/storage"
  project_id = var.project_id
  region     = var.region
  app_name   = var.app_name
  sa_email   = google_service_account.neuralgcm_sa.email
  depends_on = [google_project_service.apis]
}

module "database" {
  source           = "./modules/database"
  project_id       = var.project_id
  region           = var.region
  app_name         = var.app_name
  db_tier          = var.db_tier
  db_name          = var.db_name
  db_user          = var.db_user
  db_password      = var.db_password
  network_id       = module.networking.vpc_id
  private_ip_range = module.networking.private_ip_range
  depends_on       = [module.networking, google_project_service.apis]
}

module "cache" {
  source          = "./modules/cache"
  project_id      = var.project_id
  region          = var.region
  app_name        = var.app_name
  redis_tier      = var.redis_tier
  redis_memory_gb = var.redis_memory_gb
  network_id      = module.networking.vpc_id
  depends_on      = [module.networking, google_project_service.apis]
}

module "gke" {
  source        = "./modules/gke"
  project_id    = var.project_id
  region        = var.region
  zone          = var.zone
  app_name      = var.app_name
  node_count    = var.gke_node_count
  machine_type  = var.gke_machine_type
  gpu_type      = var.gpu_type
  gpu_count     = var.gpu_count
  network_id    = module.networking.vpc_id
  subnetwork_id = module.networking.subnet_id
  sa_email      = google_service_account.neuralgcm_sa.email
  depends_on    = [module.networking, google_project_service.apis]
}

# ── Cloud Scheduler: trigger 6h forecast runs ─────────────────
resource "google_cloud_scheduler_job" "forecast_trigger" {
  name     = "${var.app_name}-6h-forecast"
  region   = var.region
  schedule = "0 */6 * * *"   # every 6 hours
  time_zone = "UTC"

  http_target {
    uri         = "https://${var.app_name}-api.${var.region}.run.app/api/v1/internal/run-scheduled"
    http_method = "POST"
    headers     = { "Content-Type" = "application/json" }
    body        = base64encode("{\"trigger\": \"scheduler\"}")

    oidc_token {
      service_account_email = google_service_account.neuralgcm_sa.email
    }
  }
  depends_on = [google_project_service.apis]
}

# ── Monitoring: uptime check ──────────────────────────────────
resource "google_monitoring_uptime_check_config" "api_health" {
  display_name = "NeuralGCM API Health"
  timeout      = "10s"
  period       = "60s"

  http_check {
    path         = "/health"
    port         = 443
    use_ssl      = true
    validate_ssl = true
  }

  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = "${var.app_name}-api.${var.region}.run.app"
    }
  }
}

# Alert if uptime check fails for 2 consecutive minutes
resource "google_monitoring_alert_policy" "api_down" {
  display_name = "NeuralGCM API Down"
  combiner     = "OR"
  conditions {
    display_name = "Uptime check failure"
    condition_threshold {
      filter          = "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND resource.type=\"uptime_url\""
      comparison      = "COMPARISON_LT"
      threshold_value = 1
      duration        = "120s"
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_NEXT_OLDER"
        cross_series_reducer = "REDUCE_COUNT_FALSE"
        group_by_fields      = ["resource.label.*"]
      }
    }
  }

  notification_channels = var.alert_email != "" ? [google_monitoring_notification_channel.email[0].name] : []
}

resource "google_monitoring_notification_channel" "email" {
  count        = var.alert_email != "" ? 1 : 0
  display_name = "NeuralGCM Alert Email"
  type         = "email"
  labels       = { email_address = var.alert_email }
}
