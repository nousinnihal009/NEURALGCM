resource "google_artifact_registry_repository" "neuralgcm" {
  location      = var.region
  repository_id = var.app_name
  format        = "DOCKER"
  description   = "NeuralGCM Weather Platform Docker images"
}
