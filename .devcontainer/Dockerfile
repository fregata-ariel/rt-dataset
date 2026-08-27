ARG BASE_IMAGE
ARG BUILDER_IMAGE
FROM ${BUILDER_IMAGE} AS builder

FROM ${BASE_IMAGE}

# 開発むけパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    ssh-client \
    neovim

# UID/GIDの管理 (ホストマシンのパーミッション問題回避)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && apt-get update && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /workspace

# 開発用ツール (uv等) はベースイメージに存在
COPY --from=builder /wheels /wheels

USER $USERNAME

# Bash設定 (uvのパスなどを通す)
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc