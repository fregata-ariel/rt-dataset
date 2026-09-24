# shellcheck shell=bash
# CIスクリプト共通の環境設定 (source して使う)
#   IMAGE_PREFIX : CI用イメージ名の接頭辞。開発用の plateau-sionna* とは分離する
#   CI_UID/GID   : コンテナ内プロセスをランナーユーザーで実行する

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

export IMAGE_PREFIX="${IMAGE_PREFIX:-plateau-sionna-ci}"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-rt-dataset-ci}"
export CI_UID="${CI_UID:-$(id -u)}"
export CI_GID="${CI_GID:-$(id -g)}"

# CIの成果物 (JUnit XML, ログ, 生成データ) の出力先
export CI_REPORT_DIR="${CI_REPORT_DIR:-ci-reports}"
mkdir -p "${CI_REPORT_DIR}"
