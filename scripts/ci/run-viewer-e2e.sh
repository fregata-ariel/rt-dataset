#!/bin/bash
# viewer のブラウザ smoke test (V0-8 #43): 合成 bundle を Playwright (Chromium) でアップロードし、
# panel を開いて console error がないことなどを確かめる。
#   1. fixture の作成 (ci サービスで tests/viewer_bundle_fixtures.py): 正常な bundle (v3 / v2)、
#      壊れた manifest の bundle、危険なアーカイブを ci-reports/viewer-e2e/fixtures/ に置く
#   2. viewer サービスの起動 (独自 compose プロジェクト、空いている port、一時的な store、--wait)
#   3. viewer-e2e コンテナで pytest tests/e2e (viewer のネットワーク名前空間を共有し VIEWER_URL で接続)
#   失敗時は Playwright の trace / スクリーンショット (ci-reports/viewer-e2e/test-results/) と
#   viewer のログ (ci-reports/viewer-e2e/viewer.log) が残る。終了時に必ず down -v する。
# 引数はそのまま pytest に渡す (例: run-viewer-e2e.sh -k overview)。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

E2E_DIR="${CI_REPORT_DIR}/viewer-e2e"
FIXTURE_DIR="${E2E_DIR}/fixtures"
PROJECT="${COMPOSE_PROJECT_NAME}-viewer-e2e-$$"
STORE_DIR=""

cleanup() {
  local status=$?
  if [ -d "${E2E_DIR}" ]; then
    docker compose -p "$PROJECT" logs --no-color viewer > "${E2E_DIR}/viewer.log" 2>&1 || true
  fi
  docker compose -p "$PROJECT" --profile e2e down -v --remove-orphans > /dev/null 2>&1 || true
  if [ -n "${STORE_DIR}" ] && [ -d "${STORE_DIR}" ]; then
    chmod -R u+w "${STORE_DIR}" || true
    rm -rf "${STORE_DIR}"
  fi
  # コンテナ・ボリューム・ネットワークが残っていれば失敗にする
  local left
  left="$(docker ps -aq --filter "label=com.docker.compose.project=${PROJECT}")"
  left+="$(docker volume ls -q --filter "label=com.docker.compose.project=${PROJECT}")"
  left+="$(docker network ls -q --filter "label=com.docker.compose.project=${PROJECT}")"
  if [ -n "$left" ]; then
    echo "resources of compose project ${PROJECT} are left behind" >&2
    status=1
  fi
  exit "$status"
}
trap cleanup EXIT

for image in "${IMAGE_PREFIX}-viewer:latest" "${IMAGE_PREFIX}-viewer-e2e:latest"; do
  if ! docker image inspect "$image" > /dev/null 2>&1; then
    echo "image ${image} not found; build it with scripts/ci/build-images.sh" >&2
    exit 1
  fi
done

rm -rf "${E2E_DIR}"
mkdir -p "${FIXTURE_DIR}"
STORE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/viewer-e2e-store.XXXXXX")"

echo "📦 Step 1: fixtures -> ${FIXTURE_DIR}"
docker compose --profile ci run --rm ci bash -euc '
  out="$1"
  python tests/viewer_bundle_fixtures.py "$out" --archive zip
  python tests/viewer_bundle_fixtures.py "$out" --archive zip --schema 2
  python tests/viewer_bundle_fixtures.py "$out" --archive zip --broken bad_schema_version
  python tests/viewer_bundle_fixtures.py "$out" --malicious dotdot
  python -c "$2" "$out/expected.json"
' _ "${FIXTURE_DIR}" '
import json, sys
sys.path.insert(0, "tests")
import viewer_bundle_fixtures as f
expected = {
    "broken_case": "bad_schema_version",
    "broken_message": f.BROKEN_CASE_MESSAGES["bad_schema_version"],
    "malicious_case": "dotdot",
    "malicious_member": f.MALICIOUS_MEMBER_NAMES["dotdot"],
}
with open(sys.argv[1], "w") as handle:
    json.dump(expected, handle, indent=2)
'

echo "🚀 Step 2: start viewer (project ${PROJECT})"
export VIEWER_DATA_DIR="${STORE_DIR}"
export VIEWER_IMPORT_ROOT="${PWD}/${FIXTURE_DIR}"
export VIEWER_PORT=0
export VIEWER_UID="${CI_UID}"
export VIEWER_GID="${CI_GID}"
export VIEWER_E2E_DIR="${PWD}/${E2E_DIR}"
docker compose -p "$PROJECT" up -d --no-build --wait --wait-timeout 120 viewer
echo "viewer on $(docker compose -p "$PROJECT" port viewer 8000)"

echo "🧪 Step 3: browser tests"
docker compose -p "$PROJECT" --profile e2e run --rm --no-deps viewer-e2e \
  python -m pytest -p no:cacheprovider -v tests/e2e \
  --browser chromium --tracing retain-on-failure --screenshot only-on-failure \
  --output ci-reports/viewer-e2e/test-results \
  --junitxml=ci-reports/viewer-e2e/junit.xml "$@"

echo "✅ Viewer e2e finished"
