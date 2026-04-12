# GCS bucket for forecast PNG/JSON/CSV outputs
resource "google_storage_bucket" "forecasts" {
  name          = "${var.project_id}-neuralgcm-forecasts"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 90   # delete files older than 90 days
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age                   = 30
      matches_storage_class = ["STANDARD"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"   # cheaper for rarely-accessed data
    }
  }

  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD"]
    response_header = ["Content-Type"]
    max_age_seconds = 3600
  }
}

# Public read access for forecast files (PNG/CSV/JSON are not sensitive)
resource "google_storage_bucket_iam_member" "public_read" {
  bucket = google_storage_bucket.forecasts.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

# Service account write access
resource "google_storage_bucket_iam_member" "sa_write" {
  bucket = google_storage_bucket.forecasts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${var.sa_email}"
}
