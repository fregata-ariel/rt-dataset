#!/bin/bash
# base -> builder -> runtime -> ci の順にCI用イメージをビルドする。
# セルフホストランナーではDockerのレイヤーキャッシュが残るため、2回目以降は差分のみ再ビルドされる。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

for service in base builder runtime; do
  echo "📦 Building ${IMAGE_PREFIX}-${service}:${CUDA_ARCH}"
  docker compose build "${service}"
done

echo "📦 Building ${IMAGE_PREFIX}-ci:${CUDA_ARCH}"
docker compose --profile ci build ci
