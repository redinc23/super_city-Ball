#!/usr/bin/env bash
# File: bootstrap_quantum_seeker.sh
set -euo pipefail

PROJECT_NAME="quantum-seeker-2.0"
DEFAULT_BRANCH="main"

usage() {
  cat <<'USAGE'
Usage:
  bash bootstrap_quantum_seeker.sh [options]

Options:
  --dir <path>            Target directory (default: ./quantum-seeker-2.0)
  --github <repo>         Create/push to GitHub via gh. Examples:
                          --github quantum-seeker-2.0
                          --github youruser/quantum-seeker-2.0
  --private               Create GitHub repo as private (default: public)
  --no-push               Create GitHub repo but do not push
  --force                 Overwrite existing directory contents (DANGEROUS)
  -h, --help              Show help

Prereqs:
  - git (required)
  - gh  (optional; required only if you use --github)
USAGE
}

TARGET_DIR="./${PROJECT_NAME}"
GITHUB_REPO=""
VISIBILITY="public"
DO_PUSH="true"
FORCE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) TARGET_DIR="${2:?}"; shift 2;;
    --github) GITHUB_REPO="${2:?}"; shift 2;;
    --private) VISIBILITY="private"; shift 1;;
    --no-push) DO_PUSH="false"; shift 1;;
    --force) FORCE="true"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1"
    exit 1
  }
}

require_cmd git

if [[ -n "$GITHUB_REPO" ]]; then
  require_cmd gh
fi

if [[ -e "$TARGET_DIR" ]]; then
  if [[ "$FORCE" == "true" ]]; then
    rm -rf "$TARGET_DIR"
  else
    echo "Target dir exists: $TARGET_DIR"
    echo "Use --force to overwrite."
    exit 1
  fi
fi

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

mkdir -p tests examples

cat > requirements.txt <<'REQ'
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
pytest>=7.0.0
REQ

cat > config.json <<'CFG'
{
  "years": [2011, 2024],
  "n_simulations": 10000,
  "random_seed": 42,
  "output_dir": "outputs",
  "enable_visuals": true
}
CFG

cat > quantum_seeker_v2.py <<'PY'
"""
Quantum Seeker 2.0

This repo scaffolds a Colab + local runnable framework. Replace placeholders with your actual
9-phase / 13-category engine as you integrate the v1 + v2 PDFs.

Run:
  python run_analysis.py --config config.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class RunConfig:
    years: Tuple[int, int] = (2011, 2024)
    n_simulations: int = 10_000
    random_seed: int = 42
    output_dir: str = "outputs"
    enable_visuals: bool = True


def load_config(path: str | os.PathLike[str]) -> RunConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    years = tuple(data.get("years", [2011, 2024]))
    return RunConfig(
        years=(int(years[0]), int(years[1])),
        n_simulations=int(data.get("n_simulations", 10_000)),
        random_seed=int(data.get("random_seed", 42)),
        output_dir=str(data.get("output_dir", "outputs")),
        enable_visuals=bool(data.get("enable_visuals", True)),
    )


def ensure_output_dir(output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_analysis(cfg: RunConfig) -> Dict[str, Any]:
    """
    Placeholder analysis pipeline.
    Swap this out with your integrated 9-phase engine + models + visuals.
    """
    start_year, end_year = cfg.years
    return {
        "meta": {
            "project": "Quantum Seeker 2.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "years": [start_year, end_year],
            "n_simulations": cfg.n_simulations,
            "random_seed": cfg.random_seed,
        },
        "status": "scaffold_ok",
        "notes": [
            "Replace run_analysis() with the merged v1/v2 pipeline.",
            "Keep JSON output stable for tests + reporting.",
        ],
        "results": {
            "example_metric": 0.0
        },
    }


def save_json(out_dir: Path, payload: Dict[str, Any]) -> Path:
    path = out_dir / "quantum_seeker_output.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantum Seeker 2.0 runner")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg.output_dir)

    payload = run_analysis(cfg)
    out_path = save_json(out_dir, payload)

    print(f"Done. Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

cat > run_analysis.py <<'PY'
"""
Thin wrapper to run quantum_seeker_v2.py with a config file.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    cmd = [sys.executable, "quantum_seeker_v2.py", "--config", args.config]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
PY

cat > README.md <<'MD'
# Quantum Seeker 2.0

Deployable scaffold for the **Quantum Seeker 2.0** Super Bowl betting analysis framework.

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_analysis.py --config config.json
```

Output is written to `outputs/quantum_seeker_output.json`.

## Google Colab

Open `Quantum_Seeker_2.0.ipynb` in Colab and run cells top to bottom.
See: COLAB_SETUP.md

## Project layout

```
quantum-seeker-2.0/
  quantum_seeker_v2.py
  Quantum_Seeker_2.0.ipynb
  run_analysis.py
  requirements.txt
  config.json
  tests/
  examples/
```

## Next integration step

Replace the placeholder `run_analysis()` in `quantum_seeker_v2.py` with your merged 9-phase / 13-category engine.
MD

cat > COLAB_SETUP.md <<'MD'
# Colab Setup

1. Upload this repo to GitHub.
2. In Colab: File → Open notebook → GitHub and paste your repo URL.
3. Run the install cell first, then run the execution cell.

**Tip:** If you want a one-click Colab badge, add this to README once your repo is public:

`[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<OWNER>/<REPO>/blob/main/Quantum_Seeker_2.0.ipynb)`
MD

cat > LOCAL_SETUP.md <<'MD'
# Local Setup

## Requirements

- Python 3.9+ recommended
- macOS / Linux / Windows

## Install + run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py --config config.json
```

## Troubleshooting

- If pip fails on SciPy, upgrade pip/wheel: `pip install -U pip wheel setuptools`
MD

cat > API_REFERENCE.md <<'MD'
# API Reference (initial scaffold)

## quantum_seeker_v2.py

- **RunConfig** – Holds runtime config loaded from config.json.
- **run_analysis(cfg: RunConfig) -> dict** – Main pipeline entrypoint (placeholder). Replace with:
  - 9-phase analysis
  - 13 bet categories
  - sklearn models
  - Monte Carlo + significance
  - visualization + reporting output
MD

cat > tests/test_smoke.py <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_runs_and_outputs_json(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "years": [2011, 2024],
                "n_simulations": 10,
                "random_seed": 1,
                "output_dir": str(tmp_path / "out"),
                "enable_visuals": False,
            }
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        [sys.executable, "run_analysis.py", "--config", str(cfg)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Done. Output:" in r.stdout

    out_json = tmp_path / "out" / "quantum_seeker_output.json"
    assert out_json.exists()
PY

cat > Quantum_Seeker_2.0.ipynb <<'NB'
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Quantum Seeker 2.0\n",
        "\n",
        "Colab runner for the Quantum Seeker 2.0 scaffold.\n",
        "\n",
        "Steps: run install → run analysis → view outputs."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "!pip -q install numpy pandas scikit-learn matplotlib seaborn scipy"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import json\n",
        "from pathlib import Path\n",
        "\n",
        "Path('config.json').write_text(json.dumps({\n",
        "  'years': [2011, 2024],\n",
        "  'n_simulations': 1000,\n",
        "  'random_seed': 42,\n",
        "  'output_dir': 'outputs',\n",
        "  'enable_visuals': True\n",
        "}, indent=2))\n",
        "print('Wrote config.json')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "!python run_analysis.py --config config.json"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from pathlib import Path\n",
        "print(Path('outputs/quantum_seeker_output.json').read_text()[:2000])"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "name": "Quantum_Seeker_2.0.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
NB

cat > .gitignore <<'GI'
.venv/
__pycache__/
*.pyc
outputs/
output/
.ipynb_checkpoints/
*.egg-info/
dist/
build/
GI

git init -b "$DEFAULT_BRANCH" 2>/dev/null || true
git add .
git commit -m "Scaffold Quantum Seeker 2.0 (Colab + local + tests)" 2>/dev/null || true

echo "Repo scaffolded at: $TARGET_DIR"
echo "Initial commit created on branch: $DEFAULT_BRANCH"

if [[ -n "$GITHUB_REPO" ]]; then
  echo "GitHub requested: $GITHUB_REPO"

  if ! gh auth status >/dev/null 2>&1; then
    echo "gh is not authenticated. Run: gh auth login"
    exit 1
  fi

  CREATE_ARGS=()
  if [[ "$VISIBILITY" == "private" ]]; then
    CREATE_ARGS+=(--private)
  else
    CREATE_ARGS+=(--public)
  fi

  if gh repo view "$GITHUB_REPO" >/dev/null 2>&1; then
    echo "Repo exists, using it."
  else
    gh repo create "$GITHUB_REPO" "${CREATE_ARGS[@]}" --source=. --remote=origin
    echo "Created GitHub repo: $GITHUB_REPO"
  fi

  if ! git remote get-url origin >/dev/null 2>&1; then
    gh repo set-default "$GITHUB_REPO" 2>/dev/null || true
    git remote add origin "$(gh repo view "$GITHUB_REPO" --json sshUrl -q .sshUrl)"
  fi

  if [[ "$DO_PUSH" == "true" ]]; then
    git push -u origin "$DEFAULT_BRANCH"
    echo "Pushed to GitHub: $GITHUB_REPO"
  else
    echo "Skipped push (--no-push)."
  fi
fi

echo ""
echo "Next:"
echo "  - Local run: python run_analysis.py --config config.json"
echo "  - Tests: pip install pytest && pytest -q"
