# syntax=docker/dockerfile:1
#
# このファイル1本から5つのイメージを作る (docker compose build base prod dev viewer viewer-e2e)
#   base : CUDA base + OS ライブラリ + uv + Python。プロジェクトに依存しない土台
#   prod : 本番。uv.lock 通りの依存 (test グループを含む) + src。CI もこのイメージで実行する
#   dev  : Devcontainer。prod と同じ依存 + 開発ツール (jupyter, ty, git, sudo など) + src
#   viewer : Web viewer。CUDA なしの中間ステージ viewer-base (ubuntu:24.04 + uv + Python) に
#            viewer グループだけを入れる (Sionna / Mitsuba / Dr.Jit なし)。CI の viewer smoke が使う
#   viewer-e2e : ブラウザ smoke test。viewer-base に e2e グループ (Playwright) と Chromium と tests/
# prod と dev は中間ステージ deps (依存のみ、src なし) から分岐し、src は最後にコピーする。
# src の変更で dev の apt-get / 依存の層を作り直さないため (ネットワーク依存を減らす)。
#
# 全イメージ共通の非 root ユーザー app (UID 1000) で実行する。

ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04
ARG UV_VERSION=0.12.18

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------------------- base
FROM ${CUDA_BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Mitsuba / Dr.Jit は CUDA ツールキットではなく、コンテナツールキットが注入する
# ホストの NVIDIA ドライバ (CUDA, OptiX) を使うため devel イメージは不要。
#   libatomic1 : Dr.Jit のネイティブ拡張が必要とする
#   libllvm18  : Dr.Jit の CPU (LLVM) バックエンド
#   make       : Makefile のターゲットをコンテナ内で実行する
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libatomic1 libllvm18 make \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

# Python は uv で管理し、バージョンは .python-version で固定する
# (python / python3 は /usr/local/bin に置き、base 単体でも使えるようにする)
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_BIN_DIR=/usr/local/bin
COPY .python-version /tmp/python-version/.python-version
RUN cd /tmp/python-version && uv python install --default && rm -rf /tmp/python-version

# Ubuntu 24.04 既定の ubuntu ユーザー (UID 1000) を共通ユーザー app に置き換える
RUN userdel -r ubuntu 2>/dev/null || true \
    && useradd -m -u 1000 -s /bin/bash app \
    && install -d -o app -g app /opt/venv /app

# uv sync は /opt/venv に入れる (ワークスペースをマウントしても隠れない)
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

USER app
WORKDIR /app

# ---------------------------------------------------------------------------- deps
FROM base AS deps

# uv.lock 通りの依存 (test と viewer グループ) だけを入れる。sionna-rt はサブモジュールのソースからビルドされる。
COPY --chown=app:app pyproject.toml uv.lock .python-version ./
COPY --chown=app:app third_party/sionna-rt third_party/sionna-rt
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-default-groups --group test --group viewer

# ---------------------------------------------------------------------------- prod
FROM deps AS prod

COPY --chown=app:app src/ src/
COPY --chown=app:app configs/ configs/
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "plateau_rt.cli.main"]

# ---------------------------------------------------------------------------- dev
FROM deps AS dev

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget openssh-client neovim sudo \
    && rm -rf /var/lib/apt/lists/* \
    && echo "app ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/app \
    && chmod 0440 /etc/sudoers.d/app

USER app
# dev グループ (test を含む) を追加で入れる (viewer グループも残す)
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --group viewer

COPY --chown=app:app src/ src/
COPY --chown=app:app configs/ configs/
ENV PYTHONPATH=/app/src

# ---------------------------------------------------------------------------- viewer-base
FROM ubuntu:24.04 AS viewer-base

# CUDA を含まない viewer 用の土台。base と同じ手順で uv / Python / app ユーザーを入れる
# (Dr.Jit 用の libatomic1 / libllvm18 は入れない)。V0-8 (#43) の viewer-e2e もこれを使う
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

# Python は uv で管理し、バージョンは .python-version で固定する
# (python / python3 は /usr/local/bin に置き、viewer-base 単体でも使えるようにする)
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_BIN_DIR=/usr/local/bin
COPY .python-version /tmp/python-version/.python-version
RUN cd /tmp/python-version && uv python install --default && rm -rf /tmp/python-version

# Ubuntu 24.04 既定の ubuntu ユーザー (UID 1000) を共通ユーザー app に置き換える
RUN userdel -r ubuntu 2>/dev/null || true \
    && useradd -m -u 1000 -s /bin/bash app \
    && install -d -o app -g app /opt/venv /app

# uv sync は /opt/venv に入れる (ワークスペースをマウントしても隠れない)
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

USER app
WORKDIR /app

# ---------------------------------------------------------------------------- viewer
FROM viewer-base AS viewer

COPY --chown=app:app pyproject.toml uv.lock .python-version ./
# --locked は path 依存 (sionna-rt) のメタデータだけを読むので pyproject.toml だけをコピーする
COPY --chown=app:app third_party/sionna-rt/pyproject.toml third_party/sionna-rt/pyproject.toml
# --only-group viewer はプロジェクトの依存を入れないため、sionna-rt / mitsuba / drjit /
# matplotlib はインストールされない (viewer が import する fastapi, uvicorn, httpx, numpy,
# scipy, trimesh, defusedxml だけが入る)
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --only-group viewer
COPY --chown=app:app src/ src/
ENV PYTHONPATH=/app/src
EXPOSE 8000
# 127.0.0.1 を使うのは既定の VIEWER_ALLOWED_HOSTS (127.0.0.1,localhost) に合わせるため
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --start-interval=1s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]
CMD ["python", "-m", "plateau_rt.viewer", "serve", "--host", "0.0.0.0", "--port", "8000", "--data", "/data"]

# ---------------------------------------------------------------------------- viewer-e2e
FROM viewer-base AS viewer-e2e

# ブラウザ smoke test (V0-8 #43): Playwright の Python 版 + Chromium (headless shell)。
# viewer 本体 (src/) は入れない。試験は別コンテナで起動した viewer に VIEWER_URL で接続する
COPY --chown=app:app pyproject.toml uv.lock .python-version ./
COPY --chown=app:app third_party/sionna-rt/pyproject.toml third_party/sionna-rt/pyproject.toml
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --only-group e2e
# Chromium と OS ライブラリ (--with-deps は apt-get を使うので root で実行する)。
# ブラウザは共通の場所に置き、CI の任意の UID から読めるようにする
USER root
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright
RUN playwright install --with-deps --only-shell chromium \
    && chmod -R a+rX /opt/ms-playwright \
    && rm -rf /var/lib/apt/lists/*
USER app
COPY --chown=app:app tests/ tests/
CMD ["python", "-m", "pytest", "-p", "no:cacheprovider", "-v", "tests/e2e"]
