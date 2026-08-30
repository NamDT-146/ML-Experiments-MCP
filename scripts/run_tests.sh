#!/usr/bin/env bash
set -euo pipefail
cd /mnt/h/Dev/Lab/AiAuN
PY="$(tr -d '\r' < .aiaun-python)"
echo "=== offline ==="
"$PY" -m pytest tests/unit tests/migration tests/acceptance -q --tb=short -m "not live"
echo "=== live ==="
export AIAUN_LIVE_TEST=1
"$PY" -m pytest tests/acceptance -q --tb=short -m live
