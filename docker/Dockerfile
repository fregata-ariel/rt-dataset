# syntax=docker/dockerfile:1
#
# このファイル1本から3つのイメージを作る (docker compose build base prod dev)
#   base : CUDA base + OS ライブラリ + uv + Python。プロジェクトに依存しない土台
#   prod : 本番。uv.lock 通りの依存 (test グループを含む) + src。CI もこのイメージで実行する
#   dev  : Devcontainer。prod + 開発ツール (jupyter, ty, git, sudo など)
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

# ---------------------------------------------------------------------------- prod
FROM base AS prod

# 依存関係 (変更頻度が低い) を先に入れてレイヤーキャッシュを効かせる。
# sionna-rt はサブモジュールのソースからビルドされる。
COPY --chown=app:app pyproject.toml uv.lock .python-version ./
COPY --chown=app:app third_party/sionna-rt third_party/sionna-rt
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked --no-default-groups --group test

COPY --chown=app:app src/ src/
COPY --chown=app:app configs/ configs/
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "plateau_rt.cli.main"]

# ---------------------------------------------------------------------------- dev
FROM prod AS dev

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget openssh-client neovim sudo \
    && rm -rf /var/lib/apt/lists/* \
    && echo "app ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/app \
    && chmod 0440 /etc/sudoers.d/app

USER app
# dev グループ (test を含む) を追加で入れる
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    uv sync --locked
