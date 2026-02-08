"""
Smoke test: run the full pipeline and assert expected outputs exist.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# Repo root (parent of tests/)
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_run_analysis_produces_results(tmp_path: Path) -> None:
    """Run run_analysis.py with a minimal config; assert results JSON and report exist."""
    config = {
        "random_seed": 42,
        "num_legs": 80,
        "year_start": 2011,
        "year_end": 2024,
        "parlay_sizes": [2],
        "max_parlays_per_size": 20,
        "max_parlays_analyzed": 15,
        "min_categories_in_parlay": 2,
        "obscure_liquidity_threshold": 0.3,
        "monte_carlo_sims": 30,
        "report_top_n": 5,
        "q_needle_roi_threshold": 25,
        "q_needle_p_value": 0.02,
        "g_needle_roi_threshold": 35,
        "g_needle_p_value": 0.01,
        "g_needle_sharpe": 0.8,
        "output_dir": str(tmp_path / "out"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "run_analysis.py", "--config", str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    out_dir = tmp_path / "out"
    results_json = out_dir / "quantum_seeker_results.json"
    report_txt = out_dir / "quantum_seeker_report.txt"
    assert results_json.exists(), f"Expected {results_json}; stdout: {result.stdout}"
    assert report_txt.exists(), f"Expected {report_txt}"

    data = json.loads(results_json.read_text(encoding="utf-8"))
    assert "metadata" in data or "quantum_needles" in data or "golden_needles" in data
