resource "google_redis_instance" "cache" {
  name           = "${var.app_name}-redis"
  tier           = var.redis_tier
  memory_size_gb = var.redis_memory_gb
  region         = var.region
  location_id    = "${var.region}-a"

  authorized_network = var.network_id

  redis_version  = "REDIS_7_0"
  display_name   = "NeuralGCM Forecast Cache"

  redis_configs = {
    "maxmemory-policy"       = "allkeys-lru"
    "notify-keyspace-events" = "Ex"
  }
}
