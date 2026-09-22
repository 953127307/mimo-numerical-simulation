"""Independent plotting entry: draw the paper figures from a chosen run folder.

Usage:
    python plot_figures.py tracking       [--run RUN_NAME_OR_PATH] [--out PDF_PATH]
    python plot_figures.py multi_initial  [--run RUN_NAME_OR_PATH] [--out PDF_PATH]

The figure PDF is always archived inside the run folder next to its data, then
copied to the destination: ../submission/figures/ by default, or --out if given.
Without --run, the newest run folder of that kind is used (pointer file in
simulation_results/). Simulation scripts never plot; this is the only entry.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import current_theory_simulation as sim
import run_multi_initial_position as multi
import try_preview_terminal_capture_simulation as prev

FIG_DIR = sim.ROOT.parent / "submission" / "figures"
RUN_KIND = {"tracking": "preview", "multi_initial": "multi_initial"}
FIG_NAME = {
    "tracking": "simulation_tracking.pdf",
    "multi_initial": "simulation_multi_initial.pdf",
}


def resolve_run(kind: str, run_arg: str | None) -> Path:
    if run_arg is None:
        return sim.latest_run_dir(kind)
    path = Path(run_arg)
    return path if path.exists() else sim.RUNS_DIR / path


def plot_tracking(run_dir: Path, out_pdf: Path) -> None:
    baseline = dict(np.load(run_dir / "no_preview_trace.npz"))
    candidate = dict(np.load(run_dir / "preview_terminal_capture_trace.npz"))
    prev.plot_comparison(baseline, candidate, out_pdf)
    print(f"Figure written to {out_pdf}", flush=True)


def plot_multi_initial(run_dir: Path, out_pdf: Path) -> None:
    report = json.loads(
        (run_dir / "multi_initial_position_results.json").read_text(encoding="utf-8"))
    offsets = np.asarray([row["pose_offset"] for row in report["trials"]])
    traces = sorted(run_dir.glob("trial_*.npz"))
    if len(traces) != multi.N_TRIALS:
        raise FileNotFoundError(
            f"Expected {multi.N_TRIALS} trial traces in {run_dir}, found {len(traces)}")
    runs = [dict(np.load(path)) for path in traces]
    multi.make_figure(offsets[:, :2], runs, out_pdf)
    print(f"Figure written to {out_pdf}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw paper simulation figures from a run folder.")
    parser.add_argument("figure", choices=("tracking", "multi_initial"))
    parser.add_argument(
        "--run", default=None,
        help="run folder name under simulation_results/runs/ or a folder path")
    parser.add_argument(
        "--out", default=None,
        help="copy destination PDF (default: ../submission/figures/<figure name>)")
    args = parser.parse_args()

    run_dir = resolve_run(RUN_KIND[args.figure], args.run)
    run_pdf = run_dir / FIG_NAME[args.figure]
    out_pdf = Path(args.out) if args.out else FIG_DIR / FIG_NAME[args.figure]
    print(f"Run folder: {run_dir}", flush=True)
    if args.figure == "tracking":
        plot_tracking(run_dir, run_pdf)
    else:
        plot_multi_initial(run_dir, run_pdf)
    shutil.copyfile(run_pdf, out_pdf)
    print(f"Archived: {run_pdf}", flush=True)
    print(f"Copied to: {out_pdf}", flush=True)


if __name__ == "__main__":
    main()
