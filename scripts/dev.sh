#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
echo "Start the host media runtime in another terminal first:"
echo "  APRAG_HOST_DATA_DIR=$PWD/data python3 scripts/host_media_runtime.py"
docker compose --profile vector up --build
