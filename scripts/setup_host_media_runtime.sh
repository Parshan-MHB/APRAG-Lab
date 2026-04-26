#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

VENV_DIR="${APRAG_HOST_MEDIA_VENV:-.venv/host-media}"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements-host-media.txt

echo "Host media virtualenv ready: $VENV_DIR"
echo "Run:"
echo "  APRAG_HOST_DATA_DIR=$PWD/data $VENV_DIR/bin/python scripts/host_media_runtime.py"
