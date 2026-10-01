#!/bin/bash
set -euo pipefail

export PYTHONNOUSERSITE=1

NGHB_PYTHON_VERSION="${NGHB_PYTHON_VERSION:-3.12.8}"
NGHB_PBS_RELEASE="${NGHB_PBS_RELEASE:-20241219}"
NGHB_PYTHON_ARCH="${NGHB_PYTHON_ARCH:-x86_64-unknown-linux-gnu}"

check_portable_python_host() {
    local host
    host="$(uname -s):$(uname -m):$NGHB_PYTHON_ARCH"
    case "$host" in
        Linux:x86_64:x86_64-*|Linux:aarch64:aarch64-*) return 0 ;;
        *)
            echo "portable Python target $NGHB_PYTHON_ARCH cannot run on $(uname -s) $(uname -m)" >&2
            echo "use a matching Linux host or set agent_runtime_source to baked, an archive, or a URL" >&2
            return 2
            ;;
    esac
}

install_portable_python() {
    check_portable_python_host
    if [ -x "$DEPS_DIR/bin/python3" ]; then
        return 0
    fi
    local url="https://github.com/astral-sh/python-build-standalone/releases/download/${NGHB_PBS_RELEASE}/cpython-${NGHB_PYTHON_VERSION}%2B${NGHB_PBS_RELEASE}-${NGHB_PYTHON_ARCH}-install_only.tar.gz"
    curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors "$url" | tar xz -C "$DEPS_DIR" --strip-components=1
    "$DEPS_DIR/bin/python3" -m pip install --upgrade pip
}

install_nemo_gym_deps() {
    "$DEPS_DIR/bin/python3" -m pip install "$NEMO_GYM_ROOT"
}
