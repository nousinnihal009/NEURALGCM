resource "google_container_cluster" "neuralgcm" {
  name     = "${var.app_name}-cluster"
  location = var.region

  # Standard mode for GPU node pool control (Autopilot doesn't support custom GPU pools)
  enable_autopilot = false

  network    = var.network_id
  subnetwork = var.subnetwork_id

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }
    http_load_balancing {
      disabled = false
    }
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
  }

  # Remove default node pool — we define custom ones below
  remove_default_node_pool = true
  initial_node_count       = 1

  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS",
      "APISERVER",
      "SCHEDULER",
      "CONTROLLER_MANAGER",
    ]
    managed_prometheus {
      enabled = true
    }
  }

  logging_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS",
      "APISERVER",
    ]
  }
}

# ── CPU node pool for API server ──────────────────────────────
resource "google_container_node_pool" "api_nodes" {
  name       = "api-pool"
  cluster    = google_container_cluster.neuralgcm.id
  node_count = var.node_count

  node_config {
    machine_type    = var.machine_type
    service_account = var.sa_email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    labels = {
      role = "api"
      app  = var.app_name
    }
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 5
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# ── GPU node pool for NeuralGCM Celery workers ────────────────
# This is the key Phase 4 feature: GPU inference = ~2.5s vs ~60s CPU
resource "google_container_node_pool" "gpu_workers" {
  name       = "gpu-worker-pool"
  cluster    = google_container_cluster.neuralgcm.id
  node_count = 1

  node_config {
    machine_type    = "n1-standard-4"
    service_account = var.sa_email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    # GPU accelerator
    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    labels = {
      role                                = "gpu-worker"
      app                                 = var.app_name
      "cloud.google.com/gke-accelerator"  = var.gpu_type
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }
  }

  autoscaling {
    min_node_count = 0   # scale to zero when idle (saves cost!)
    max_node_count = 3
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
