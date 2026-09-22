# Numerical Simulation

Simulation and plotting code for the paper.

## Files

- `current_theory_simulation.py` — Simulation library.
  - Model and constants: `Plant` (velocity/current block parameters), `DBAR`, `ROUTE_II_KAPPA`, data-sampling parameters.
  - Data and synthesis: `collect_snapshots` (direct-derivative snapshots), `synthesize_block` (null-space synthesis), `monte_carlo_margin`.
  - Reference and outer loop: `reference`, `reference_components`, `outer_command`, and related command functions.
  - Closed loop: `closed_loop_rhs`, `simulate`.
  - Run-folder helpers: `new_run_dir`, `update_latest`, `latest_run_dir`.
- `try_preview_terminal_capture_simulation.py` — Main simulation entry.
  - `main`: synthesize weights → baseline/preview closed loops → the four preview horizons of Table 3 (official JSON key `preview_horizon_table`) → all products written to `runs/<timestamp>_preview/`.
  - `run_full_mimo`: full MIMO closed loop; `trajectory_metrics`: error/voltage metrics; `plot_comparison`: tracking comparison figure.
- `run_multi_initial_position.py` — Batch experiment over 16 initial poses.
  - `main`: per-trial closed loops, results written to `runs/<timestamp>_multi_initial/`; `make_figure`: multi-initial trajectory figure.
- `plot_figures.py` — Standalone plotting entry.
  - `plot_tracking` / `plot_multi_initial`: load data from a run folder, save the figure PDF into that folder, then copy it to `../submission/figures/`.
- `simulation_results/runs/<timestamp>_<tag>/` — All products of one run; `latest_preview.txt` and `latest_multi_initial.txt` are pointers to the latest runs.

## Usage

```bash
python try_preview_terminal_capture_simulation.py    # main simulation + Table 3 data
python run_multi_initial_position.py                 # multi-initial experiment
python plot_figures.py tracking       [--run folder] [--out path.pdf]
python plot_figures.py multi_initial  [--run folder] [--out path.pdf]
```
