#!/usr/bin/env bash
# Generate Flower HTTP basic auth credentials for Nginx.
#
# Usage:
#   bash docker/make_htpasswd.sh <username> <password>
#
# Both arguments are REQUIRED. No defaults are provided intentionally.
# Shipping with a default password is a security vulnerability.
#
# Example:
#   bash docker/make_htpasswd.sh flower_admin "$(openssl rand -hex 16)"
#
# The generated docker/.htpasswd is gitignored.
# Store the credentials in your secrets manager (Vault, AWS Secrets
# Manager, GCP Secret Manager, etc.) — not in this repository.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo ""
    echo "ERROR: Both username and password are required."
    echo ""
    echo "Usage: bash docker/make_htpasswd.sh <username> <password>"
    echo ""
    echo "To generate a strong random password:"
    echo "  bash docker/make_htpasswd.sh flower_admin \"\$(openssl rand -hex 16)\""
    echo ""
    echo "Never use a default or guessable password. The Flower dashboard"
    echo "exposes all task arguments including location coordinates."
    echo ""
    exit 1
fi

USERNAME="$1"
PASSWORD="$2"

# Reject obviously weak passwords
if [[ ${#PASSWORD} -lt 12 ]]; then
    echo ""
    echo "ERROR: Password must be at least 12 characters."
    echo "Generate one with: openssl rand -hex 16"
    echo ""
    exit 1
fi

if [[ "$PASSWORD" == "changeme" || "$PASSWORD" == "password" \
   || "$PASSWORD" == "admin" || "$PASSWORD" == "flower" ]]; then
    echo ""
    echo "ERROR: Password '$PASSWORD' is a known-bad credential."
    echo "Generate one with: openssl rand -hex 16"
    echo ""
    exit 1
fi

HTPASSWD_FILE="$(dirname "$0")/.htpasswd"

htpasswd -cb "$HTPASSWD_FILE" "$USERNAME" "$PASSWORD"

echo ""
echo "Created $HTPASSWD_FILE for user '$USERNAME'."
echo "Store these credentials in your secrets manager."
echo "Do NOT commit docker/.htpasswd to version control."
echo ""
