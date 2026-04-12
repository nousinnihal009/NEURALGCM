output "vpc_id" {
  value = google_compute_network.vpc.id
}
output "subnet_id" {
  value = google_compute_subnetwork.subnet.id
}
output "private_ip_range" {
  value = google_compute_global_address.private_ip_range.name
}
