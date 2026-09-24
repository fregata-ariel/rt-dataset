#!/bin/bash
# Lint: ruff check / format --check (CIイメージに固定したバージョンを使う)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

docker compose --profile ci run --rm ci bash -c \
  "ruff check --no-cache src tests scripts && ruff format --check --no-cache src tests scripts"
