#!/usr/bin/env bash
# Run once before docker-compose up to create Flower basic auth credentials.
# Usage: bash docker/make_htpasswd.sh <username> <password>
set -euo pipefail
USER=${1:-admin}
PASS=${2:-changeme}
htpasswd -cb docker/.htpasswd "$USER" "$PASS"
echo "Created docker/.htpasswd for user '$USER'"
echo "Change the default password before deploying to production."
