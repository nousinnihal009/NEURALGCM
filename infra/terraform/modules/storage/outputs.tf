output "bucket_name" {
  value = google_storage_bucket.forecasts.name
}
output "bucket_url" {
  value = google_storage_bucket.forecasts.url
}
