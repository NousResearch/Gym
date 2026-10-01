#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_portable_python.sh"

: "${DEPS_DIR:?DEPS_DIR must be set}"
: "${NEMO_GYM_ROOT:?NEMO_GYM_ROOT must be set}"

HERMES_REQ="$NEMO_GYM_ROOT/responses_api_agents/hermes_agent/requirements.txt"
HERMES_SPEC="${HERMES_SPEC:-$(sed -n 's/^hermes-agent @ //p' "$HERMES_REQ")}"
: "${HERMES_SPEC:?could not read hermes-agent pin from $HERMES_REQ}"

install_portable_python
install_nemo_gym_deps
"$DEPS_DIR/bin/python3" -m pip install --force-reinstall --no-deps "$HERMES_SPEC"
"$DEPS_DIR/bin/python3" -m pip install "$HERMES_SPEC"
"$DEPS_DIR/bin/python3" -c "import model_tools; from run_agent import AIAgent"

HERMES_SOURCE="${HERMES_SPEC#git+}"
HERMES_REVISION="${HERMES_SOURCE##*@}"
HERMES_REPO="${HERMES_SOURCE%@*}"
CHECKOUT_DIR="$(mktemp -d)"
trap 'rm -rf "$CHECKOUT_DIR"' EXIT
git clone --quiet --filter=blob:none --no-checkout "$HERMES_REPO" "$CHECKOUT_DIR/source"
git -C "$CHECKOUT_DIR/source" checkout --quiet "$HERMES_REVISION"
rm -rf "$DEPS_DIR/skills"
cp -R "$CHECKOUT_DIR/source/skills" "$DEPS_DIR/skills"
