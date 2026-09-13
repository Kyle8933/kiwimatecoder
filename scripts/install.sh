#!/bin/sh
# Install KiwiMateCoder from the GitHub repository.
#
# Usage:
#   sh scripts/install.sh [ref]
#
# ref is a branch, tag, or commit SHA and defaults to `main`; set
# KIWIMATECODER_REF to change the default or KIWIMATECODER_REPO_URL to install
# from a fork. The script checks for Python >= 3.10, prefers pipx, falls back
# to `pip install --user`, then verifies the console script.
#
# POSIX sh only: no bashisms and no self-deleting tricks.
set -eu

REPO_URL="${KIWIMATECODER_REPO_URL:-https://github.com/Kyle8933/kiwimatecoder.git}"
REF="${1:-${KIWIMATECODER_REF:-main}}"
TARGET="git+${REPO_URL}@${REF}"

say() {
    printf '%s\n' "$*"
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 \
    || fail "Python 3 not found. Install Python 3.10 or newer and rerun (or set PYTHON=/path/to/python3)."

"$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    || fail "Python 3.10 or newer is required; found $("$PYTHON" --version 2>&1)."

say "Installing kiwimatecoder from ${TARGET}"

if command -v pipx >/dev/null 2>&1; then
    say "Using pipx"
    pipx install --force "git+${REPO_URL}@${REF}"
elif "$PYTHON" -m pip --version >/dev/null 2>&1; then
    say "Using ${PYTHON} -m pip install --user"
    "$PYTHON" -m pip install --user --upgrade --force-reinstall "$TARGET"
else
    fail "neither pipx nor '${PYTHON} -m pip' is available; install pip first."
fi

if command -v kiwimatecoder >/dev/null 2>&1; then
    kiwimatecoder --version
    say "Installed. Run 'kiwimatecoder setup' to add a provider key."
else
    fail "kiwimatecoder was installed but is not on PATH. Add your user bin directory (usually \$HOME/.local/bin) to PATH, then run: kiwimatecoder --version"
fi
