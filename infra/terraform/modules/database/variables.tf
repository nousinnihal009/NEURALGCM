variable "project_id" {
  type = string
}
variable "region" {
  type = string
}
variable "app_name" {
  type = string
}
variable "db_tier" {
  type = string
}
variable "db_name" {
  type = string
}
variable "db_user" {
  type = string
}
variable "db_password" {
  type      = string
  sensitive = true
}
variable "network_id" {
  type = string
}
variable "private_ip_range" {
  type = string
}
