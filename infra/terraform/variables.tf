variable "project_id" {
  description = "GCP project ID"
  type        = string
}
variable "region" {
  description = "Primary GCP region"
  type        = string
  default     = "us-central1"
}
variable "zone" {
  type    = string
  default = "us-central1-a"
}
variable "environment" {
  description = "Deployment environment: dev | staging | prod"
  type        = string
  default     = "prod"
}
variable "app_name" {
  type    = string
  default = "neuralgcm-weather"
}

# Database
variable "db_tier" {
  description = "Cloud SQL machine type"
  type        = string
  default     = "db-g1-small"   # upgrade to db-n1-standard-2 for prod
}
variable "db_name" {
  type    = string
  default = "neuralgcm_weather"
}
variable "db_user" {
  type    = string
  default = "neuralgcm"
}
variable "db_password" {
  type      = string
  sensitive = true
}

# Redis
variable "redis_tier" {
  type    = string
  default = "BASIC"   # use STANDARD_HA for production HA
}
variable "redis_memory_gb" {
  type    = number
  default = 1
}

# GKE
variable "gke_node_count" {
  type    = number
  default = 2
}
variable "gke_machine_type" {
  type    = string
  default = "n1-standard-4"
}
variable "gpu_type" {
  description = "GPU type for NeuralGCM inference workers"
  type        = string
  default     = "nvidia-tesla-t4"   # T4: ~$0.35/hr, 2.5s inference
  # alternatives: nvidia-a100-80gb (~$2.93/hr, <1s inference)
}
variable "gpu_count" {
  type    = number
  default = 1
}

# Container images
variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

# Domain
variable "domain" {
  description = "Custom domain for the API (optional)"
  type        = string
  default     = ""
}

# Alerting
variable "alert_email" {
  description = "Email for monitoring alerts"
  type        = string
  default     = ""
}
