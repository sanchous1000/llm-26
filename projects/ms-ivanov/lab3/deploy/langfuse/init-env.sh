#!/bin/sh
# Create deploy/langfuse/.env with random secrets and project keys (once), and add the
# client settings (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL) to lab3/.env.
# Existing values are never overwritten. Nothing secret is printed.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ENV_FILE="$HERE/.env"
APP_ENV="$HERE/../../.env"

rand() { openssl rand -hex "$1"; }

if [ ! -f "$ENV_FILE" ]; then
  umask 077
  cat > "$ENV_FILE" <<EOF
POSTGRES_PASSWORD=$(rand 16)
CLICKHOUSE_PASSWORD=$(rand 16)
MINIO_ROOT_PASSWORD=$(rand 16)
REDIS_AUTH=$(rand 16)
SALT=$(rand 16)
ENCRYPTION_KEY=$(rand 32)
NEXTAUTH_SECRET=$(rand 32)
LANGFUSE_PUBLIC_KEY=pk-lf-$(rand 16)
LANGFUSE_SECRET_KEY=sk-lf-$(rand 24)
LANGFUSE_ADMIN_EMAIL=admin@lab3.local
LANGFUSE_ADMIN_PASSWORD=$(rand 12)
EOF
  echo "created $ENV_FILE"
fi

if ! grep -q '^LANGFUSE_SECRET_KEY=' "$APP_ENV" 2>/dev/null; then
  {
    echo ""
    echo "# Langfuse (local, deploy/langfuse) used by the agents"
    grep -E '^LANGFUSE_(PUBLIC|SECRET)_KEY=' "$ENV_FILE"
    echo "LANGFUSE_BASE_URL=http://localhost:3000"
  } >> "$APP_ENV"
  echo "added Langfuse client settings to $APP_ENV"
fi
echo "UI login: admin@lab3.local, password: LANGFUSE_ADMIN_PASSWORD in $ENV_FILE"
