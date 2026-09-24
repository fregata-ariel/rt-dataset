#!/bin/bash
# 単体テスト: runtimeイメージ上 (GPU不要) で tests/ を実行する。push/PR ごとに回す軽量CI。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

docker compose --profile ci run --rm ci \
  python -m pytest -p no:cacheprovider -v tests \
  --junitxml="${CI_REPORT_DIR}/unit-tests.xml" "$@"
