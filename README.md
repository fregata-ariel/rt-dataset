# PLATEAU × Sionna-RT RF カメラデータセット生成

PLATEAU の 3D 都市モデル (CityJSON) から Sionna-RT のシーンを生成し、電波伝搬
シミュレーションによって次のデータを作るツール群です。

- **カバレッジ**: 2D パスゲインマップと 3D レンダリング
- **RF カメラ**: UE の平面受信開口で観測した複素 CFR と、それを現像した
  角度 / 角度-遅延画像
  - 1 BS / 1 UE の MVP と、その物理校正・角度-遅延展開
  - 1 BS / multi-UE のマルチビューデータセット (Gaussian Splatting 用カメラモデル付き)

```text
CityJSON ──build──▶ PLY + Mitsuba XML + manifest.json
                      │
                      ├─ simulate / run-all ──▶ カバレッジ (npy, PNG, 3D render)
                      │
                      ├─ rf-camera ──▶ aperture_cfr ─ rf-camera-calibrate ─▶ 校正済み角度画像
                      │                                └ rf-camera-delay ───▶ 角度-遅延ボリューム
                      │
                      └─ rf-camera-multiview ──▶ views/*/rf/ + camera_model.npz + dataset_manifest.json
                                                    │
                                                    └─ rf-camera-optical ──▶ views/*/optical/ + transforms.json
```

## 環境構築

GPU (NVIDIA) と Docker / NVIDIA Container Toolkit を前提に、Docker イメージ上で開発・実行します。

```bash
# サブモジュール (third_party/sionna-rt) を含めて取得
git clone --recursive git@github.com:fregata-ariel/rt-dataset.git
cd rt-dataset

# base / prod (本番) / dev (開発) の3イメージを作成
docker compose build
```

| イメージ | 用途 | 中身 |
|---|---|---|
| `plateau-sionna-base` | 土台 | CUDA base (Ubuntu 24.04) + OS ライブラリ + uv + Python 3.12 |
| `plateau-sionna` | 本番・CI | base + `uv.lock` 通りの依存 (test グループ含む) + `src/` |
| `plateau-sionna-dev` | Devcontainer | 本番と同じ依存 + jupyter / ty / git / sudo |

3つとも `docker/Dockerfile` のマルチステージから作られ、非 root ユーザー `app` (UID 1000)
で動きます。Python 依存は `/opt/venv` に入っており、バージョンは `uv.lock` と
`.python-version` で固定しています。Mitsuba / Dr.Jit はホストの NVIDIA ドライバを使うため、
GPU 世代ごとのビルドは不要です。

本番イメージの実行例:

```bash
docker run --rm --gpus all -v "$PWD/data:/app/data" plateau-sionna \
  python -m plateau_rt.cli.main build data/raw/mock_building.city.json data/generated/example
```

VS Code で「Dev Containers: Reopen in Container」を実行すると、dev イメージで開発環境が
起動します (`uv sync --locked` で依存をロックファイルに合わせます)。
コンテナ外でも `uv sync` で同じ依存を用意できます (sionna-rt はサブモジュールからビルド)。

## 使い方

コマンドは `src/` を `PYTHONPATH` に通して実行します (Makefile が設定済み)。

### モックデータでの実行

`data/raw/mock_building.city.json` (10 m 立方の建物1棟) で各段階を試せます。
出力先は `data/generated/mock_results/` です。

| ターゲット | 内容 |
|---|---|
| `make build-mock` | CityJSON → PLY / Mitsuba XML / manifest.json |
| `make sim-mock` / `make render-mock` | カバレッジ計算 / 2D ヒートマップ画像 |
| `make run-all-mock` | build + カバレッジ + パス分解 + 3D レンダリング |
| `make view-mock` | 結果のインタラクティブビューア (要ディスプレイ) |
| `make rf-camera-mock` | 1 BS / 1 UE の RF カメラ画像 |
| `make rf-camera-calibrate-mock` | 角度画像の物理校正 (GPU 不要) |
| `make rf-camera-delay-mock` | 角度-遅延ボリュームへの展開 (GPU 不要) |
| `make rf-camera-multiview-mock` | 1 BS / 8 UE のマルチビューデータセット |
| `make rf-camera-optical-mock` | マルチビューデータセットに光学参照レンダーを追加 |
| `make build-mock-city` | 4棟モックシティ + 200 m 地面プレーン → PLY / Mitsuba XML / manifest.json |
| `make rf-camera-multiview-mock-city` | モックシティの 1 BS / 12 UE マルチビューデータセット |
| `make clean` | 生成物の削除 |

### CLI

```bash
PYTHONPATH=./src python -m plateau_rt.cli.main --help
```

| コマンド | 内容 |
|---|---|
| `build INPUT OUTPUT_DIR [--ground-plane-size-m 200]` | CityJSON から Sionna-RT シーンを生成 (`--ground-plane-size-m` で原点中心・一辺指定 m・z = -0.01 m の正方形地面を追加。0 で無効。地面材 `itu_medium_dry_ground` は 1-10 GHz のみ対応) |
| `simulate XML MANIFEST OUTPUT_DIR` | カバレッジマップを計算 |
| `run-all INPUT OUTPUT_DIR` | build からレンダリングまで一気通貫 |
| `render DIR` / `view DIR` | ヒートマップ画像の生成 / ビューア |
| `rf-camera XML OUTPUT_DIR` | 1 BS / 1 UE の RF カメラ画像 |
| `rf-camera-calibrate DIR` | `rf-camera` の出力を物理座標に校正 |
| `rf-camera-delay DIR` | 校正済み出力を角度-遅延ボリュームに展開 |
| `rf-camera-multiview XML OUTPUT_DIR` | リング配置の multi-UE データセット |
| `rf-camera-optical DATASET_DIR` | マルチビューデータセットに位置合わせ済みの光学参照レンダーを追加 |

各オプションは `--help` を参照してください。RF カメラの観測モデル・座標系・出力形式は
次のドキュメントにまとめています。

- [docs/rf_camera_mvp.md](docs/rf_camera_mvp.md): 1 BS / 1 UE、校正、角度-遅延
- [docs/rf_camera_multiview.md](docs/rf_camera_multiview.md): マルチビューデータセットとカメラモデル
- [docs/optical_reference.md](docs/optical_reference.md): 光学参照レンダー (issue #11)、3DGS 学習との接続

## コード構成

```text
src/plateau_rt/
  domain/
    models.py              建物・面・シーンのドメインモデル
    rf_camera/             RF カメラの数式と座標系 (NumPy のみ・Sionna 非依存)
      imaging.py             周波数グリッド、PlanarArray の並べ替え、空間 FFT
      calibration.py         角度画像の物理校正、方向余弦軸、回転、LoS 方向
      delay.py               角度-遅延 IFFT、伝搬可能方向マスク、支配遅延
      camera.py              視点 (リング配置・look-at) と方向余弦カメラモデル
      optical.py             光学参照レンダーのピンホールカメラ数式 (NumPy のみ)
  application/
    build_scene.py         CityJSON → シーン生成パイプライン
    rf_camera_calibration.py, rf_camera_delay.py
                           RF カメラ出力ディレクトリの後処理 (GPU 不要)
    optical_reference.py   光学参照レンダーの生成 (rf-camera-optical。既定のレンダラーは Mitsuba)
    viewer.py              カバレッジ結果ビューア
  adapters/
    plateau/               CityJSON パーサ
    geometry/              trimesh による PLY 生成
    sionna/                Sionna-RT 連携 (シーン XML、カバレッジ、RF カメラのトレース)
      rf_tracing.py          RF カメラ共通: アレイ設定、PathSolver、開口 CFR 抽出
      rf_camera.py           1 BS / 1 UE MVP
      rf_camera_dataset.py   1 BS / multi-UE データセット
      optical_render.py      Mitsuba によるレイ単位の光学レンダラー
    plotting/              RF カメラ診断画像 (matplotlib)
  cli/main.py              click CLI
```

`domain/rf_camera` と後処理は Sionna を import しないことをテスト
(`tests/test_rf_camera_boundaries.py`) で保証しています。

## テストと CI

```bash
scripts/ci/run-lint.sh        # ruff check / format --check
scripts/ci/run-unit-tests.sh  # tests/ (GPU・Sionna 不要)
scripts/ci/run-heavy.sh       # Sionna-RT テスト + モック E2E + 生成物検証 (GPU)
```

GitHub Actions のセルフホストランナー (GPU ノード) で、push ごとの単体テストと、
`v*` / `milestone/*` タグで起動する重い GPU 検証を回しています。
詳細は [docs/ci.md](docs/ci.md) を参照してください。

本番イメージ上での Sionna-RT 自体の検証は `scripts/run-infra-test.sh` で行えます。
