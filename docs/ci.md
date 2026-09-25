# CI (GitHub Actions セルフホストランナー)

GPU ノード (RTX 2080 Ti) 上のセルフホストランナーで、
**本番イメージ (`docker/Dockerfile` の prod ステージ) そのもの**を使って2種類の CI を回します。
本番イメージには test 依存グループ (pytest, ruff) が含まれています。

| ワークフロー | トリガー | ランナーラベル | 内容 | 目安時間 (キャッシュ有) |
|---|---|---|---|---|
| `Unit tests` (`unit-tests.yml`) | 全ブランチへの push (`*.md`, `docs/` のみの変更は除く)、手動 | `self-hosted, docker` | ruff (lint/format) + `tests/` の単体テスト (GPU・Sionna 不要) + viewer smoke (軽量 viewer イメージの内容検査と API フロー) | 1 分未満 |
| `Milestone heavy CI (GPU)` (`milestone-heavy.yml`) | タグ `v*` / `milestone/*` の push、手動 | `self-hosted, docker, gpu` | 単体テスト + Sionna-RT 本体テスト (CPU/GPU) + モック E2E パイプライン + 生成物検証 | 5 分前後 |

## 節目 CI の回し方

```bash
git tag milestone/1bs-multiue-rf-camera-dataset
git push origin milestone/1bs-multiue-rf-camera-dataset
```

または Actions タブ / `gh workflow run milestone-heavy.yml --ref <branch>` で
`Milestone heavy CI (GPU)` を手動実行します。

生成されたモックデータセット (`mock_results/`) と JUnit XML は、
`milestone-heavy-reports` アーティファクトとして 30 日間保存されます。

### 重い処理の内容 (`scripts/ci/run-heavy.sh`)

1. GPU / Mitsuba `cuda_ad_mono_polarized` バリアントの確認
2. 既存のインフラ統合テスト (compose の `test` サービス: Sionna-RT の CPU/LLVM サブセット)
3. Sionna-RT 本体 (`third_party/sionna-rt/test/unit`) の全ユニットテストを GPU で実行
4. Makefile のモックターゲットによる End-to-End 実行
   (`run-all-mock render-mock rf-camera-mock rf-camera-calibrate-mock rf-camera-delay-mock rf-camera-multiview-mock rf-camera-optical-mock`)
5. `scripts/ci/check_mock_outputs.py` による生成物の検証
   - ファイルの有無、配列の形状・有限性
   - 校正後の角度ピークと幾何 LoS 方向の誤差 ≤ 0.05
   - 最強ボクセルの遅延と LoS 遅延の誤差 ≤ 遅延分解能 (10 ns)
   - 光学参照レンダー (pinhole/hemisphere) のファイル・形状・`transforms.json`、
     mock ボックスとの解析的な幾何整合性 (issue #11、詳細は
     [docs/optical_reference.md](optical_reference.md))
6. `scripts/ci/check_hemisphere_split.py` による前面/背面分割パターンの確認
7. `scripts/ci/check_optical_render.py` による光学レイレンダラーと
   mock ボックス形状の一致確認

本番イメージには TensorFlow / JAX / PyTorch を入れていないため、Sionna-RT の
`test_cpx_convert` のうちそれらへの変換ケースは skip されます。

Sionna-RT の GPU パストレースは実行ごとに完全には再現しません (パスの順序や
float の最下位桁が変わることがある)。そのため生成物の検証はビット一致ではなく、
物理的な整合性で判定しています。

## ローカルでの実行

ランナーと同じスクリプトをそのまま実行できます。

```bash
scripts/ci/build-images.sh     # base, prod, viewer (引数で dev も: build-images.sh base prod dev)
scripts/ci/run-lint.sh         # ruff check / format --check
scripts/ci/run-unit-tests.sh   # 単体テスト
scripts/ci/run-viewer-smoke.sh # 軽量 viewer イメージの smoke (要: build-images.sh で viewer をビルド)
scripts/ci/run-heavy.sh        # 節目の重い処理 (GPU)
```

CI 用イメージは `plateau-sionna-ci`, `plateau-sionna-ci-base`, `plateau-sionna-ci-dev`,
`plateau-sionna-ci-viewer`
という名前でビルドされ、開発者の `plateau-sionna*` は上書きしません (`IMAGE_PREFIX` で切り替え)。
節目の CI では dev イメージもビルドし、Devcontainer が壊れていないことも確認します。
コンテナはランナーと同じ UID (既定 1000 = イメージの `app` ユーザー) で実行されるため、
ワークスペースに他ユーザー所有のファイルは残りません。

### Viewer smoke (`scripts/ci/run-viewer-smoke.sh`)

軽量 viewer イメージ (`viewer` ステージ、CUDA・Sionna なし) の内容検査と、
起動したコンテナへの API フローを検証します。手順は以下の通りです。

1. イメージ内容の検査 (`scripts/ci/viewer_smoke.py image-check` をイメージ内で実行)
   - 禁止モジュールが import できないこと (`sionna`, `mitsuba`, `drjit`, `matplotlib`)
   - 禁止配布物が入っていないこと (`sionna`, `sionna-rt`, `mitsuba`, `drjit`, `matplotlib`,
     `nvidia-*` / `cuda-*`)
   - CUDA 環境変数 (`CUDA_VERSION` など) がないこと
   - CUDA ライブラリ (`libcuda*`, `libcudart*`, `libnvidia-*` など) がないこと
   - viewer モジュール (`plateau_rt.viewer.api.app`, `plateau_rt.viewer.__main__`) が
     import できること
   - イメージに healthcheck (`GET /api/health`) があることも確認する
2. fixture アーカイブの作成 (`ci` サービスで `tests/viewer_bundle_fixtures.py OUT --archive zip`)
3. viewer サービスの起動 (独自の compose プロジェクトで `up -d --no-build --wait viewer`;
   healthcheck が通るまで待ち、ホスト側は `VIEWER_PORT=0` で選ばれたランダムな 127.0.0.1 ポート)
4. ホスト側からの疎通確認 (`curl` で `/api/health` と `/`)
5. コンテナ内からの API フロー (`scripts/ci/viewer_smoke.py http`):
   upload (201, `created: true`) -> `GET /api/bundles` に digest が載ること ->
   `derived/overview` (200、なければ 202 のジョブをポーリング) ->
   `overview.json` (200、`ETag` 付き、`num_views` / `num_bs` > 0) ->
   `GET .../status` が `complete: true` になるまで待つ

コンテナのログは `ci-reports/viewer-smoke.log` に保存されます (unit-tests.yml の
アーティファクトにも含まれます)。store の `bundles/<digest>/raw/` は read-only のため、
掃除の前後で `chmod -R u+w` して権限を戻します (そうしないとワークスペースの削除や
次回 checkout の cleanup が壊れます)。

viewer image の大きさ: 約 426 MB (406 MiB、`docker image inspect -f '{{.Size}}'`、2026-09 時点の目安)。
本番イメージ (prod, CUDA + Sionna) は約 1.34 GB です。smoke 全体はキャッシュ有で 10 秒未満です。

## セルフホストランナー

### セットアップ / 削除

```bash
scripts/ci/setup-self-hosted-runner.sh            # ダウンロード(SHA256検証) + 登録 + systemd ユーザーサービス起動
scripts/ci/setup-self-hosted-runner.sh --remove   # サービス停止 + 登録解除
```

- 登録名: `<hostname>-gpu`、ラベル: `gpu, docker, cuda-sm75`
- 配置先: `~/actions-runner/rt-dataset`
- サービス: `systemctl --user status actions-runner-rt-dataset.service`
- ログ: `journalctl --user -u actions-runner-rt-dataset.service -f`

ログアウト/再起動後も常駐させるには linger を有効にします (要 sudo, 1回のみ):

```bash
sudo loginctl enable-linger "$USER"
```

### 公開リポジトリでのセキュリティ

このリポジトリは公開 (public) のため、fork からの PR がセルフホストランナー上で
任意コードを実行するリスクがあります。

- ワークフローは `pull_request` トリガーを使わない (push / タグ / 手動のみ)
- `permissions: contents: read` に限定
- **推奨設定**: Settings → Actions → General →
  "Fork pull request workflows from outside collaborators" を
  **"Require approval for all external contributors"** にする。CLI の場合:

  ```bash
  gh api -X PUT repos/fregata-ariel/rt-dataset/actions/permissions/fork-pr-contributor-approval \
    -f approval_policy=all_external_contributors
  ```
