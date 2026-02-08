# API Reference - Quantum Seeker 2.0

## Module: `quantum_seeker_v2`

### `execute_quantum_seeker(config_path: Optional[str] = None, config_override: Optional[Dict] = None) -> Dict`
Runs the full analysis pipeline and writes outputs to the configured output directory.

**Parameters:**
- `config_path`: Path to a JSON config file (optional)
- `config_override`: Dictionary of config values to override (optional)

**Returns:**
- Results dictionary containing metadata and analysis sections

### `QuantumSeekerFramework`
Main class encapsulating data generation, analysis, reporting, and visualization.

**Key Methods:**
- `execute_full_analysis() -> Dict`
  - Runs the 9-phase analysis pipeline and returns results.
- `generate_full_report() -> str`
  - Returns the text report.
- `generate_html_report(report_text: Optional[str] = None) -> str`
  - Returns the HTML report.
- `generate_visualizations(output_dir: Optional[str] = None) -> Dict[str, str]`
  - Generates PNG charts and returns file paths.

## Configuration Options
Config values in `config.json`:
- `random_seed`
- `num_legs`
- `year_start`
- `year_end`
- `parlay_sizes`
- `max_parlays_per_size`
- `max_parlays_analyzed`
- `min_categories_in_parlay`
- `obscure_liquidity_threshold`
- `monte_carlo_sims`
- `q_needle_roi_threshold`
- `q_needle_p_value`
- `g_needle_roi_threshold`
- `g_needle_p_value`
- `g_needle_sharpe`
- `output_dir`
