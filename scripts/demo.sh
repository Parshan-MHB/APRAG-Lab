#!/usr/bin/env sh
set -eu

BASE_URL="${BASE_URL:-http://localhost:8000}"

curl -fsS -X POST "$BASE_URL/api/sample-dataset/load" >/tmp/APRAG-Lab-sample.json
PROJECT_ID="$(python - <<'PY'
import json
print(json.load(open('/tmp/APRAG-Lab-sample.json'))['project']['id'])
PY
)"

echo "Loaded sample project: $PROJECT_ID"
curl -fsS -X POST "$BASE_URL/api/acceptance/run"
