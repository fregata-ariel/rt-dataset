#!/bin/bash
# 節目 (マイルストーン) に回す重いGPU検証。
#   1. GPU / Mitsuba CUDA バリアントの確認
#   2. インフラ統合テスト (Sionna-RT のCPU/LLVMサブセット, 既存の test サービス)
#   3. Sionna-RT 本体 (submodule) のユニットテスト全体を GPU で実行
#   4. モック建物での End-to-End パイプライン (Makefile の *-mock ターゲット)
#   5. 生成物の検証 (形状・有限性・LoSの角度/遅延の整合性、前後の半球の向き)
#   6. 前面/背面分割パターンが等方性素子を正確に分割していることの確認
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

MOCK_OUT="${CI_REPORT_DIR}/mock_results"

gpu_run() {
  docker compose --profile ci run --rm ci-gpu "$@"
}

echo "🖥️  Step 1: GPU / Mitsuba CUDA variant"
gpu_run bash -c 'nvidia-smi && python -c "
import mitsuba as mi
mi.set_variant(\"cuda_ad_mono_polarized\")
import sionna.rt
print(\"sionna-rt\", sionna.rt.__version__, \"variant:\", mi.variant())
"'

echo "🧪 Step 2: インフラ統合テスト (Sionna-RT CPU/LLVM subset)"
docker compose --profile test run --rm test

echo "🧪 Step 3: Sionna-RT unit tests (GPU)"
docker compose --profile ci run --rm -w /workspace/third_party/sionna-rt/test ci-gpu \
  python -m pytest -p no:cacheprovider -q -rfE --durations=10 unit \
  --junitxml="/workspace/${CI_REPORT_DIR}/sionna-rt-unit-gpu.xml"

echo "🏙️  Step 4: Mock end-to-end pipeline -> ${MOCK_OUT}"
rm -rf "${MOCK_OUT}"
gpu_run make MOCK_OUT="${MOCK_OUT}/" \
  run-all-mock render-mock \
  rf-camera-mock rf-camera-calibrate-mock rf-camera-delay-mock \
  rf-camera-multiview-mock

echo "🔍 Step 5: Validate generated outputs"
# ビュー数は Makefile の rf-camera-multiview-mock (--num-views 8) と一致させる
gpu_run python scripts/ci/check_mock_outputs.py "${MOCK_OUT}" --num-views 8

echo "🧭 Step 6: Front/back hemisphere split reproduces the isotropic element"
gpu_run python scripts/ci/check_hemisphere_split.py "${MOCK_OUT}/mock_building.city.xml"

echo "✅ Heavy CI finished"
