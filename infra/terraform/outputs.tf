output "registry_url" {
  value       = module.registry.repository_url
  description = "Artifact Registry URL for pushing Docker images"
}
output "gke_cluster_name" {
  value = module.gke.cluster_name
}
output "db_private_ip" {
  value     = module.database.private_ip
  sensitive = true
}
output "redis_host" {
  value     = module.cache.host
  sensitive = true
}
output "redis_port" {
  value = module.cache.port
}
output "forecast_bucket" {
  value = module.storage.bucket_name
}
output "api_url" {
  value       = "https://${var.app_name}-api.${var.region}.run.app"
  description = "Public API URL (update after Cloud Run deploy)"
}
