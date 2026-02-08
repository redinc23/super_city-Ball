# Google Colab Setup - Quantum Seeker 2.0

## Option A (recommended): One-cell automated run
In a Colab cell, run the bootstrap script directly from GitHub:

```bash
!curl -sSL https://raw.githubusercontent.com/OWNER/REPO/main/colab_bootstrap.sh | \
  bash -s -- --repo https://github.com/OWNER/REPO.git
```

If you need a specific branch, add `--branch`:

```bash
!curl -sSL https://raw.githubusercontent.com/OWNER/REPO/BRANCH/colab_bootstrap.sh | \
  bash -s -- --repo https://github.com/OWNER/REPO.git --branch BRANCH
```

## Option B: Notebook upload flow
1. Open `Quantum_Seeker_2.0.ipynb` in Google Colab.
2. Run the dependency installation cell.
3. Upload the following files when prompted:
   - `quantum_seeker_v2.py`
   - `config.json` (optional)
4. Run the execution cell to generate results.
5. Use the download cell to retrieve output files.

## Outputs
By default, outputs are saved in `output/`:
- `quantum_seeker_report.txt`
- `quantum_seeker_report.html`
- `quantum_seeker_results.json`
- Visualizations (`*.png`)

## Tips
- Adjust parameters in `config.json` before uploading if needed.
- For faster execution, reduce `num_legs` or `monte_carlo_sims`.
