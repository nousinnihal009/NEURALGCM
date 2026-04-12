output "db_password_secret_id" {
  value = google_secret_manager_secret.db_password.secret_id
}
output "secret_key_secret_id" {
  value = google_secret_manager_secret.secret_key.secret_id
}
