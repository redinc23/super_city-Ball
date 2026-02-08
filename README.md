# Quantum Seeker 2.0

[![CI](https://github.com/redinc23/super_city-Ball/actions/workflows/ci.yml/badge.svg)](https://github.com/redinc23/super_city-Ball/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redinc23/super_city-Ball/blob/main/Quantum_Seeker_2.0.ipynb)

Quantum Seeker 2.0 is a Super Bowl betting analysis framework that generates synthetic bet legs, evaluates parlay strategies, and produces comprehensive reports with visualizations.

## Highlights
- 9-phase analysis pipeline (temporal, synergies, meta-edges, round robin, correlations)
- Enhanced synthetic data generation with realistic patterns
- Statistical modeling with logistic regression and Monte Carlo significance
- Text, JSON, and HTML reports
- Visualization output (PNG charts)

## Quick Start (Local)
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run the analysis:
   - `python quantum_seeker_v2.py`
3. Results are written to `output/` by default.

See `LOCAL_SETUP.md` for detailed instructions.

## Quick Start (Google Colab)
**Option A (recommended): one-cell bootstrap**

```bash
!curl -sSL https://raw.githubusercontent.com/OWNER/REPO/main/colab_bootstrap.sh | \
  bash -s -- --repo https://github.com/OWNER/REPO.git
```

**Option B: notebook upload flow**
1. Open `Quantum_Seeker_2.0.ipynb` in Colab.
2. Run the install cell.
3. Upload `quantum_seeker_v2.py` and optional `config.json`.
4. Execute the analysis cell.

See `COLAB_SETUP.md` for more details and options.

## Outputs
Default outputs in `output/`:
- `quantum_seeker_report.txt`
- `quantum_seeker_report.html`
- `quantum_seeker_results.json`
- Visualizations (`*.png`)

## Configuration
Edit `config.json` or pass `--config` to `run_analysis.py`.
Key options:
- `num_legs`: Number of synthetic bet legs (default 5000)
- `parlay_sizes`: List of parlay sizes (default [2,3,4])
- `monte_carlo_sims`: Number of Monte Carlo simulations (default 2000)
- `output_dir`: Output directory (default "output")

## File Structure
```
quantum-seeker-2.0/
├── quantum_seeker_v2.py
├── run_analysis.py
├── Quantum_Seeker_2.0.ipynb
├── requirements.txt
├── config.json
├── README.md
├── LOCAL_SETUP.md
├── COLAB_SETUP.md
├── API_REFERENCE.md
└── tests/
```
