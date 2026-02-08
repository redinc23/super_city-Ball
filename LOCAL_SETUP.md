# Local Setup - Quantum Seeker 2.0

## Prerequisites
- Python 3.8 or higher
- pip

## Setup
1. Create and activate a virtual environment (recommended):
   - macOS/Linux:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`
   - Windows:
     - `python -m venv .venv`
     - `.venv\\Scripts\\activate`

2. Install dependencies:
   - `pip install -r requirements.txt`

## Run the analysis
Fire up the app (runs analysis and starts report server):
- `./fire_up.sh`

Basic run:
- `python quantum_seeker_v2.py`

With a config file:
- `python run_analysis.py --config config.json`

Override output directory and seed:
- `python run_analysis.py --output-dir output --seed 123`

## Outputs
Results are saved in the configured output directory (default: `output/`):
- `quantum_seeker_report.txt`
- `quantum_seeker_report.html`
- `quantum_seeker_results.json`
- Visualizations (`*.png`)
