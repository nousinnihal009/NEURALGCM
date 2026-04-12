# All application secrets stored in Secret Manager
# Never in environment variables, Docker images, or Kubernetes YAML

resource "google_secret_manager_secret" "db_password" {
  secret_id = "neuralgcm-db-password"
  replication {
    auto {}
  }
}
resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}

resource "google_secret_manager_secret" "secret_key" {
  secret_id = "neuralgcm-secret-key"
  replication {
    auto {}
  }
}
resource "google_secret_manager_secret_version" "secret_key" {
  secret      = google_secret_manager_secret.secret_key.id
  secret_data = random_password.secret_key.result
}

resource "random_password" "secret_key" {
  length  = 64
  special = false
}
