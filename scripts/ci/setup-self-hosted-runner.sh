#!/bin/bash
# このノードに GitHub Actions セルフホストランナーを登録し、systemd ユーザーサービスとして常駐させる。
#
# 前提: gh CLI がリポジトリ管理者権限でログイン済み、実行ユーザーが docker グループに所属。
# sudo 不要 (systemd --user を使用)。再起動後も自動起動させるには linger の有効化が必要:
#   sudo loginctl enable-linger "$USER"
#
# 使い方:
#   scripts/ci/setup-self-hosted-runner.sh            # 登録 + サービス起動
#   scripts/ci/setup-self-hosted-runner.sh --remove   # サービス停止 + 登録解除
set -euo pipefail

REPO="${REPO:-fregata-ariel/rt-dataset}"
RUNNER_NAME="${RUNNER_NAME:-$(hostname)-gpu}"
# ワークフローの runs-on はこのラベルで対象ノードを選ぶ
RUNNER_LABELS="${RUNNER_LABELS:-gpu,docker,cuda-sm75}"
RUNNER_DIR="${RUNNER_DIR:-${HOME}/actions-runner/${REPO##*/}}"
SERVICE_NAME="actions-runner-${REPO##*/}.service"
UNIT_DIR="${HOME}/.config/systemd/user"

remove_runner() {
  systemctl --user disable --now "${SERVICE_NAME}" 2>/dev/null || true
  rm -f "${UNIT_DIR}/${SERVICE_NAME}"
  systemctl --user daemon-reload
  if [ -f "${RUNNER_DIR}/.runner" ]; then
    local token
    token="$(gh api -X POST "repos/${REPO}/actions/runners/remove-token" --jq .token)"
    (cd "${RUNNER_DIR}" && ./config.sh remove --token "${token}")
  fi
  echo "✅ Runner '${RUNNER_NAME}' removed (files kept in ${RUNNER_DIR})"
}

if [ "${1:-}" = "--remove" ]; then
  remove_runner
  exit 0
fi

# 1. ランナー本体のダウンロード (リリースノート記載の SHA256 で検証)
mkdir -p "${RUNNER_DIR}"
cd "${RUNNER_DIR}"
if [ ! -x ./config.sh ]; then
  release_json="$(gh api repos/actions/runner/releases/latest)"
  version="$(jq -r .tag_name <<<"${release_json}")"
  version="${version#v}"
  tarball="actions-runner-linux-x64-${version}.tar.gz"
  expected_sha="$(jq -r .body <<<"${release_json}" \
    | sed -n 's/.*<!-- BEGIN SHA linux-x64 -->\([0-9a-f]*\)<!-- END SHA linux-x64 -->.*/\1/p')"

  echo "⬇️  Downloading actions/runner v${version}"
  curl -fsSL -o "${tarball}" \
    "https://github.com/actions/runner/releases/download/v${version}/${tarball}"
  echo "${expected_sha}  ${tarball}" | sha256sum -c -
  tar xzf "${tarball}"
  rm -f "${tarball}"
fi

# 2. リポジトリへの登録 (同名ランナーがあれば置き換え)
if [ ! -f .runner ]; then
  token="$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq .token)"
  ./config.sh --unattended --replace \
    --url "https://github.com/${REPO}" \
    --token "${token}" \
    --name "${RUNNER_NAME}" \
    --labels "${RUNNER_LABELS}" \
    --work _work
fi

# 3. systemd ユーザーサービスとして常駐 (svc.sh 相当の設定。svc.sh は sudo が必要なため使わない)
mkdir -p "${UNIT_DIR}"
cat >"${UNIT_DIR}/${SERVICE_NAME}" <<UNIT
[Unit]
Description=GitHub Actions runner ${RUNNER_NAME} (${REPO})
After=network-online.target

[Service]
ExecStart=${RUNNER_DIR}/runsvc.sh
WorkingDirectory=${RUNNER_DIR}
KillMode=process
KillSignal=SIGTERM
TimeoutStopSec=5min
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
UNIT
cp -f bin/runsvc.sh ./runsvc.sh
systemctl --user daemon-reload
systemctl --user enable --now "${SERVICE_NAME}"

if [ "$(loginctl show-user "${USER}" --property=Linger --value 2>/dev/null)" != "yes" ]; then
  loginctl enable-linger "${USER}" 2>/dev/null \
    || echo "⚠️  linger が無効です。ログアウト/再起動後も常駐させるには: sudo loginctl enable-linger ${USER}"
fi

systemctl --user --no-pager status "${SERVICE_NAME}" | head -5
echo "✅ Runner '${RUNNER_NAME}' [${RUNNER_LABELS}] is running as ${SERVICE_NAME}"
