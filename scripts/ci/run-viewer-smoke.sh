#!/bin/bash
# viewer smoke: 軽量 viewer イメージの内容検査と、起動したコンテナへの API フロー検証。
#   1. イメージ内容の検査 (image-check: 禁止モジュール/配布物・CUDA・viewer import)
#   2. fixture アーカイブの作成 (ci サービスで tests/viewer_bundle_fixtures.py)
#   3. viewer サービスの起動 (独自 compose プロジェクト、healthcheck で --wait)
#   4. ホスト側からの疎通確認 (curl で公開 127.0.0.1 ポート)
#   5. コンテナ内からの API フロー (upload 201 -> overview -> status complete)
#   ログは ci-reports/viewer-smoke.log に保存し、store の read-only 権限は戻して掃除する。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

SMOKE_DIR="${CI_REPORT_DIR}/viewer-smoke"
STORE_DIR="${CI_REPORT_DIR}/viewer-smoke-store"
IMAGE="${IMAGE_PREFIX}-viewer:latest"
PROJECT="${COMPOSE_PROJECT_NAME}-viewer-smoke-$$"

cleanup() {
  docker compose -p "$PROJECT" logs --no-color viewer > "${CI_REPORT_DIR}/viewer-smoke.log" 2>&1 || true
  docker compose -p "$PROJECT" down -v --remove-orphans > /dev/null 2>&1 || true
  if [ -d "${STORE_DIR}" ]; then
    chmod -R u+w "${STORE_DIR}" || true
  fi
}
trap cleanup EXIT

if [ -d "${SMOKE_DIR}" ]; then
  chmod -R u+w "${SMOKE_DIR}"
  rm -rf "${SMOKE_DIR}"
fi
if [ -d "${STORE_DIR}" ]; then
  chmod -R u+w "${STORE_DIR}"
  rm -rf "${STORE_DIR}"
fi
mkdir -p "${SMOKE_DIR}" "${STORE_DIR}"

echo "🔍 Step 1: image contents (${IMAGE})"
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
  echo "image ${IMAGE} not found; build it with scripts/ci/build-images.sh viewer" >&2
  exit 1
fi
if [ "$(docker image inspect -f '{{json .Config.Healthcheck}}' "$IMAGE")" = "null" ]; then
  echo "image ${IMAGE} has no healthcheck" >&2
  exit 1
fi
docker run --rm -i --entrypoint python "$IMAGE" - image-check < scripts/ci/viewer_smoke.py

echo "📦 Step 2: fixture archive -> ${SMOKE_DIR}/bundle.zip"
docker compose --profile ci run --rm ci \
  python tests/viewer_bundle_fixtures.py "${SMOKE_DIR}" --archive zip

echo "🚀 Step 3: start viewer (project ${PROJECT})"
export VIEWER_DATA_DIR="${PWD}/${STORE_DIR}"
export VIEWER_IMPORT_ROOT="${PWD}/${SMOKE_DIR}"
export VIEWER_PORT=0
export VIEWER_UID="${CI_UID}"
export VIEWER_GID="${CI_GID}"
docker compose -p "$PROJECT" up -d --no-build --wait --wait-timeout 120 viewer
HOST_PORT="$(docker compose -p "$PROJECT" port viewer 8000)"
echo "viewer on ${HOST_PORT}"

echo "🌐 Step 4: host-side health check"
curl -fsS "http://${HOST_PORT}/api/health"
echo ""
curl -fsS -o /dev/null "http://${HOST_PORT}/"

echo "🧪 Step 5: API flow in the container"
docker compose -p "$PROJECT" exec -T viewer \
  python - http --base-url http://127.0.0.1:8000 --archive /import/bundle.zip \
  < scripts/ci/viewer_smoke.py

SIZE_BYTES="$(docker image inspect -f '{{.Size}}' "$IMAGE")"
echo "✅ Viewer smoke finished (image ${IMAGE}, $((SIZE_BYTES / 1024 / 1024)) MiB)"
