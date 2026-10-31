#!/usr/bin/env bash
# Rebuild and restart the local stack: the API, the worker and the services process (git
# sync and the cron scheduler). Reads .env. Logs go to logs/.
# Usage: scripts/restart.sh [--no-build]
set -Eeuo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
set -a
# shellcheck disable=SC1091
. ./.env
set +a
PORT="${PORT:-3000}"
mkdir -p logs

echo "Restarting Flows on port $PORT..."
pkill -u "$(id -u)" -f "node $ROOT/dist/api/server.js" || true
pkill -u "$(id -u)" -f "node $ROOT/dist/worker/index.js" || true
pkill -u "$(id -u)" -f "node $ROOT/dist/services/index.js" || true
sleep 1

if [ "${1:-}" != "--no-build" ]; then
  echo "Generating Prisma client..."
  if ! npx prisma generate >> logs/api.log 2>&1; then
    echo "ERROR: prisma generate failed. See logs/api.log"
    exit 1
  fi
  echo "Building TypeScript..."
  if ! npm run build >> logs/api.log 2>&1; then
    echo "ERROR: TypeScript build failed. See logs/api.log"
    exit 1
  fi
fi

echo "Waiting for Postgres and RabbitMQ..."
python3 scripts/wait_for.py "$DATABASE_URL" 5432 60 || exit 1
python3 scripts/wait_for.py "$RABBITMQ_URL" 5672 60 || exit 1

nohup node "$ROOT/dist/api/server.js" >> logs/api.log 2>&1 &
API_PID=$!
nohup node "$ROOT/dist/worker/index.js" >> logs/worker.log 2>&1 &
nohup node "$ROOT/dist/services/index.js" >> logs/services.log 2>&1 &

for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "Started: API on port ${PORT}, worker, services"
    exit 0
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "ERROR: the API exited. See logs/api.log"
    exit 1
  fi
  sleep 1
done
echo "ERROR: the API did not answer on /health within 30 s. See logs/api.log"
exit 1
