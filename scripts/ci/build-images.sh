#!/bin/bash
# CI 用イメージをビルドする (既定: base と prod。例: build-images.sh base prod dev)。
# セルフホストランナーでは Docker のレイヤーキャッシュが残るため、2回目以降は差分のみ再ビルドされる。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [ "$#" -eq 0 ]; then
  set -- base prod
fi
echo "📦 Building $* (${IMAGE_PREFIX})"
docker compose build "$@"
