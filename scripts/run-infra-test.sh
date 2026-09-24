#!/bin/bash
set -euo pipefail

# スクリプトの配置ディレクトリからプロジェクトルートに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "🧪 インフラ統合テスト (prod イメージ直接検証)"
echo "=========================================="
echo ""

# 1. イメージのビルド (base -> prod)
echo "📦 Step 1: base / prod イメージをビルド中..."
docker compose build base prod

# 2. テストの実行
echo "🧪 Step 2: prod イメージ上で Sionna-RT ユニットテストを実行中..."
docker compose --profile test run --rm test

echo ""
echo "=========================================="
echo "✅ インフラ統合テストが正常に完了しました！"
echo "=========================================="
