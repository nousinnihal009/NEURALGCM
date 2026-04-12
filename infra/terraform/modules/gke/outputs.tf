output "cluster_name" {
  value = google_container_cluster.neuralgcm.name
}
output "cluster_endpoint" {
  value = google_container_cluster.neuralgcm.endpoint
}
