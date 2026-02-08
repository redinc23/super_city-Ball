#!/usr/bin/env bash
# One-command Colab bootstrap for Quantum Seeker 2.0.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash colab_bootstrap.sh [options]

Options:
  --repo <url>         GitHub repo URL to clone (optional if already in repo)
  --branch <name>      Git branch to clone (default: main)
  --dir <path>         Target directory name (defaults to repo name)
  --config <path>      Config JSON path (default: config.json)
  --output-dir <path>  Override output directory
  --seed <int>         Override random seed
  --skip-install       Skip pip install step
  -h, --help           Show help

Examples:
  bash colab_bootstrap.sh --repo https://github.com/OWNER/REPO.git
  bash colab_bootstrap.sh --repo https://github.com/OWNER/REPO.git --branch main
USAGE
}

REPO_URL=""
BRANCH="main"
TARGET_DIR=""
CONFIG_PATH="config.json"
OUTPUT_DIR=""
RANDOM_SEED=""
SKIP_INSTALL="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_URL="${2:?}"; shift 2;;
    --branch) BRANCH="${2:?}"; shift 2;;
    --dir) TARGET_DIR="${2:?}"; shift 2;;
    --config) CONFIG_PATH="${2:?}"; shift 2;;
    --output-dir) OUTPUT_DIR="${2:?}"; shift 2;;
    --seed) RANDOM_SEED="${2:?}"; shift 2;;
    --skip-install) SKIP_INSTALL="true"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

derive_repo_dir() {
  local url="$1"
  local base="${url##*/}"
  base="${base%.git}"
  if [[ -z "$base" ]]; then
    echo "quantum-seeker-2.0"
  else
    echo "$base"
  fi
}

if [[ -n "$REPO_URL" ]]; then
  if [[ -z "$TARGET_DIR" ]]; then
    TARGET_DIR="$(derive_repo_dir "$REPO_URL")"
  fi
  if [[ -d "$TARGET_DIR" ]]; then
    echo "Using existing directory: $TARGET_DIR"
    if [[ -d "$TARGET_DIR/.git" ]]; then
      cd "$TARGET_DIR"
      CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
      if [[ -n "$CURRENT_BRANCH" && "$CURRENT_BRANCH" != "$BRANCH" ]]; then
        echo "Switching from branch '$CURRENT_BRANCH' to '$BRANCH'"
        git fetch origin "$BRANCH" 2>/dev/null || true
        git checkout "$BRANCH" 2>/dev/null || echo "Warning: Could not checkout branch '$BRANCH', using existing branch '$CURRENT_BRANCH'"
      fi
      cd - >/dev/null
    fi
  else
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$TARGET_DIR"
  fi
  cd "$TARGET_DIR"
else
  if [[ -n "$TARGET_DIR" ]]; then
    cd "$TARGET_DIR"
  fi
fi

if [[ ! -f "run_analysis.py" ]]; then
  echo "run_analysis.py not found. Run inside the repo or pass --repo."
  exit 1
fi

PYTHON_BIN=""
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Python is not installed. Install Python 3 and retry."
  exit 1
fi

if [[ "$SKIP_INSTALL" != "true" ]]; then
  if [[ ! -f "requirements.txt" ]]; then
    echo "requirements.txt not found. Skipping install."
  else
    "$PYTHON_BIN" -m pip install -q -r requirements.txt
  fi
fi

ARGS=()
if [[ -n "$CONFIG_PATH" && -f "$CONFIG_PATH" ]]; then
  ARGS+=(--config "$CONFIG_PATH")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$RANDOM_SEED" ]]; then
  ARGS+=(--seed "$RANDOM_SEED")
fi

"$PYTHON_BIN" run_analysis.py "${ARGS[@]}"

resolve_output_dir() {
  if [[ -n "$OUTPUT_DIR" ]]; then
    echo "$OUTPUT_DIR"
    return
  fi
  if [[ -n "$CONFIG_PATH" && -f "$CONFIG_PATH" ]]; then
    "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    print(data.get("output_dir", "output"))
except Exception:
    print("output")
PY
    return
  fi
  echo "output"
}

OUT_DIR="$(resolve_output_dir)"
echo ""
echo "Output directory: $OUT_DIR"
if [[ -d "$OUT_DIR" ]]; then
  echo "Files:"
  ls -1 "$OUT_DIR"
fi
