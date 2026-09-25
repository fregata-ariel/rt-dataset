#!/bin/bash
# 節目 (マイルストーン) に回す重いGPU検証。
#   1. GPU / Mitsuba CUDA バリアントの確認
#   2. インフラ統合テスト (Sionna-RT のCPU/LLVMサブセット, 既存の test サービス)
#   3. Sionna-RT 本体 (submodule) のユニットテスト全体を GPU で実行
#   4. モック建物での End-to-End パイプライン (Makefile の *-mock ターゲット)
#   5. 生成物の検証 (形状・有限性・LoSの角度/遅延の整合性、前後の半球の向き、光学参照レンダー)
#   6. パスGTから aperture CFR が再合成できることの確認
#   7. 前面/背面分割パターンが等方性素子を正確に分割していることの確認
#   8. 光学レイレンダラーが mock ボックス形状と一致することの確認
#   9. カバレッジマップUE配置 (#16): 保存ラジオマップ+seed の再現性確認、
#      LoS/NLoS 割当と幾何 LoS マスク (トレースした LoS パスとの一致)、BS ごとのしきい値 (any)
#  10. トモグラフィー用データセットプロファイル (#15 T20): rich mock city の ci プロファイル (8 リング視点 x 2 BS, N=128, los=False オラクル) の manifest 検査
#  11. トモグラフィーGT (T17) と再合成・BSパターン・偏波の検証 (#15 T21): L0f NMSE < 1e-3, 直接波比 0.5 dB 以内
#  12. トモグラフィー heavy smoke (#15 T22): Step 11 の tomography GT を使い、32 ビンのサブバンドで全構成を実行して §6.7 の判定とレポート (ci-reports/tomography_smoke) を出す
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
  rf-camera-multiview-mock rf-camera-optical-mock rf-camera-partial-mock

echo "🔍 Step 5: Validate generated outputs"
# ビュー数・BS数は Makefile の rf-camera-multiview-mock (--num-views 8,
# --bs-position x2) と一致させる
gpu_run python scripts/ci/check_mock_outputs.py "${MOCK_OUT}" --num-views 8 --num-bs 2

echo "🧬 Step 6: Path GT resynthesises the aperture CFRs"
gpu_run python scripts/ci/check_path_gt_resynthesis.py "${MOCK_OUT}/rf_camera_multiview"

echo "🧭 Step 7: Front/back hemisphere split reproduces the isotropic element"
gpu_run python scripts/ci/check_hemisphere_split.py "${MOCK_OUT}/mock_building.city.xml"

echo "🎨 Step 8: Optical ray renderer matches the mock box geometry"
gpu_run python scripts/ci/check_optical_render.py "${MOCK_OUT}/mock_building.city.xml"

echo "📍 Step 9: Coverage-map UE placement (#16) on the mock"
COV="${MOCK_OUT}/coverage"
rm -rf "${COV}"
gpu_run make MOCK_OUT="${COV}/" rf-camera-coverage-mock rf-camera-optical-coverage-mock
gpu_run make MOCK_OUT="${COV}/" RF_CAMERA_COVERAGE_OUT="${COV}/rerun_same_seed/" \
  RADIO_MAP="${COV}/rf_camera_coverage/placement/radio_map.json" rf-camera-coverage-mock
gpu_run make MOCK_OUT="${COV}/" RF_CAMERA_COVERAGE_OUT="${COV}/other_seed/" PLACEMENT_SEED=1 \
  RADIO_MAP="${COV}/rf_camera_coverage/placement/radio_map.json" rf-camera-coverage-mock
gpu_run python scripts/ci/check_coverage_placement.py "${COV}/rf_camera_coverage" \
  --num-views 8 --num-bs 2 --mock-box \
  --same-seed-rerun "${COV}/rerun_same_seed" --other-seed "${COV}/other_seed"

echo "🧊 Step 10: Tomography dataset profile (#15 T20) on the rich mock city"
TOMO="${MOCK_OUT}/tomography"
rm -rf "${TOMO}"
gpu_run make MOCK_OUT="${TOMO}/" rf-tomo-profile-mock-city
gpu_run python scripts/ci/check_tomography_profile.py "${TOMO}/rf_tomo/ci/refraction" \
  --profile ci --num-views 8 --num-bs 2

echo "🔁 Step 11: Tomography GT, L0f resynthesis and BS-pattern check (#15 T17, T21)"
gpu_run python -m plateau_rt.cli.main rf-tomo-gt "${TOMO}/rf_tomo/ci/refraction"
gpu_run python scripts/ci/check_tomography_resynthesis.py "${TOMO}/rf_tomo/ci/refraction" \
  --report "${CI_REPORT_DIR}/tomography_resynthesis.json"

echo "🧪 Step 12: Tomography heavy smoke (#15 T22) on the ci profile"
CI_BLAS_THREADS=2 docker compose --profile ci run --rm ci \
  python scripts/ci/check_tomography_smoke.py "${TOMO}/rf_tomo/ci/refraction" \
  --out "${CI_REPORT_DIR}/tomography_smoke" --workers 8 --overwrite

echo "✅ Heavy CI finished"
