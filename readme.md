# 開発環境テンプレート (`Dev Env Template`)

これは、DockerとVS Code Dev Containersを利用した、標準的な開発環境を迅速に構築するためのテンプレートリポジトリです。

このテンプレートの目的は、「開発環境の定義」と「アプリケーションのソースコード」を分離することです。これにより、どんなアプリケーションを開発する場合でも、チームメンバー全員が全く同じクリーンな環境で開発を始めることができます。

## 🎯 特徴

  * **環境の統一**: `Dockerfile`で定義されたOSとツール群により、開発者全員が同じ環境を利用できます。これにより「私の環境では動くのに…」といった問題を撲滅します。
  * **ホストOSの汚染防止**: 開発に必要なツール群はすべてコンテナ内に閉じ込められるため、ローカルマシンをクリーンに保てます。
  * **迅速なセットアップ**: 新しいメンバーや新しいPCでも、いくつかのコマンドを実行するだけで、数分で開発環境が整います。
  * **柔軟な設定**: `project.env`ファイルを変更するだけで、開発対象のアプリケーションを簡単に切り替えることができます。

-----

## ✍️ 事前準備

このテンプレートを利用する前に、お使いのローカルマシンに以下のツールがインストールされていることを確認してください。

  * [Git](https://git-scm.com/)
  * [Visual Studio Code](https://code.visualstudio.com/)
  * [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) (VS Code拡張機能)
  * [Docker Desktop](https://www.docker.com/products/docker-desktop/) または互換性のあるコンテナランタイム

-----

## 🚀 利用手順

以下の手順に従って、新しいプロジェクトのための開発環境を構築します。

### Step 1: このテンプレートから新しいリポジトリを作成する

まず、このリポジトリを元にして、あなたのプロジェクト管理用の新しいリポジトリを作成します。

1.  このページの右上にある緑色の **"Use this template"** ボタンをクリックし、**"Create a new repository"** を選択します。
2.  新しいリポジトリの名前（例: `my-project-dev-env`）などを設定し、リポジトリを作成します。

> **Note**
> このテンプレートを直接`fork`するのではなく、**"Use this template"** を利用してください。これにより、クリーンな履歴で新しいプロジェクトを開始できます。

### Step 2: 新しく作成したリポジトリをクローンする

サブモジュール（Sionna-RT）を含めてローカルマシンにクローンします。

```bash
# "your-account"と"your-new-repository"をあなたのものに置き換えてください
git clone --recursive git@github.com:your-account/your-new-repository.git

# クローンしたディレクトリに移動
cd your-new-repository
```

> **Note**
> すでに通常クローンしている場合は、以下のコマンドでサブモジュールを初期化してください。
> ```bash
> git submodule update --init --recursive
> ```

### Step 3: 環境設定ファイル（.env）を作成・編集する

`.env.example` をコピーして `.env` を作成し、マシンのGPU環境に合わせて設定します。

```bash
cp .env.example .env
```

**`.env`**
```bash
# お使いのGPUのCompute Capability (例: 1660Ti=75, RTX 3080/3090=86, RTX 4090=89, A100=80)
CUDA_ARCH=75

# 利用するベースCUDAイメージ
CUDA_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04
```

### Step 4: プロジェクト設定ファイル（project.env）を作成・編集する

`project.env.example` をコピーして `project.env` を作成し、開発対象のアプリケーション情報を設定します。

```bash
cp project.env.example project.env
```

**`project.env`**

```bash
# 必須：開発対象のアプリケーションリポジトリのSSH URL
APP_REPO_URL=git@github.com:your-org/your-application.git

# 任意：コンテナ内でのGitコミットに利用する名前とメールアドレス
GIT_USER_NAME="Your Name"
GIT_USER_EMAIL="your_email@example.com"
```

### Step 5: SSHキーの準備

コンテナ内から`git clone`を行うために、SSHキーが必要です。

1.  ホストマシン（あなたのPC）にSSHキーペア（`~/.ssh/id_ed25519`など）が設定されていることを確認してください。
2.  そのキーペアの**公開鍵** (`~/.ssh/id_ed25519.pub`) を、GitHubやその他のGitホスティングサービスに登録しておいてください。

### Step 6: ベース・ビルダーイメージを事前にビルドする

Devcontainerが参照するベースイメージおよびSionna-RTのコンパイル済みビルダーイメージをビルドします。

```bash
# 1. 共通ベースイメージのビルド
docker compose build base

# 2. Sionna-RTのコンパイルとWheel生成ビルダーのビルド
docker compose build builder
```

### Step 7: 開発環境を起動する

いよいよ開発環境を起動します。

1.  VS Codeで、Step 2でクローンした**ひな形リポジトリのフォルダ**（`your-new-repository`）を開きます。
2.  VS Codeの右下に「**Reopen in Container**」というポップアップが表示されたら、そのボタンをクリックします。
      * 表示されない場合は、コマンドパレット（`Ctrl+Shift+P` または `Cmd+Shift+P`）を開き、「**Dev Containers: Reopen in Container**」を検索して実行します。
3.  初回起動時は、Dockerイメージのビルドと`post-create.sh`スクリプトの実行に数分かかります。

処理が完了すると、VS Codeのウィンドウがリロードされ、ターミナルが開きます。エクスプローラーには、`project.env`で指定したアプリケーションのソースコードが表示されているはずです。

これで、開発を始める準備が整いました！ 🎉

-----

## 🧪 インフラの統合テスト (CI / 検証)

本テンプレートでビルドされる `runtime` イメージが、Sionna-RT を正しくインストールし実行可能であるかを検証するためのテストスクリプトが用意されています。

```bash
# base -> builder -> runtime をビルドし、Sionna-RTのユニットテストを実行
./scripts/run-infra-test.sh
```

このテストは以下を検証します:
1. `Dockerfile.base`, `Dockerfile.builder`, `Dockerfile.runtime` のビルドが成功すること
2. Sionna-RT のコンパイル済み Wheel が `runtime` イメージに正しくインストールされること
3. Sionna-RT の数値計算・コアロジックのユニットテスト（30件以上）が正常にパスすること

GitHub Actions（GPUノード上のセルフホストランナー）による単体テストと、節目に回す重いGPU検証については [docs/ci.md](docs/ci.md) を参照してください。

-----

## 🔧 環境のカスタマイズ

この環境をさらにカスタマイズしたい場合は、以下のファイルを編集してください。

  * **ツールの追加・変更**: `Dockerfile`を編集し、`apt-get install`の行に必要なパッケージを追加したり、バージョンを変更したりします。
  * **VS Codeの拡張機能や設定の変更**: `.devcontainer/devcontainer.json`を編集し、推奨する拡張機能を追加したり、コンテナのメモリ割り当てを変更したりできます。

ファイルを編集した後は、コマンドパレットから「**Dev Containers: Rebuild Container**」を実行して変更を適用してください。

-----

## oc_ha ブランチ

OpenCodeとHermes Agentが有効化された開発環境のテンプレートです。