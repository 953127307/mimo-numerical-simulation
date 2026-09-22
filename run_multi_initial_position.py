# -*- coding: utf-8 -*-
"""Multi-initial-position runs under the current paper configuration.

Sixteen randomized initial poses/currents around the reference start are
driven by the same previewed terminal-capture loop as Subsection IV-B
(T_p = 1.4 s, k_rho = 0.8, k_alpha = 0.7, Route II weights, 4-V hard limit,
bounded online disturbances) and produce one trajectory figure for the
simulation section.  Disturbance draws are identical across trials (simulate
re-seeds), mirroring the archived single-run protocol.
"""

from __future__ import annotations

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np

import current_theory_simulation as sim
import try_preview_terminal_capture_simulation as prev

N_TRIALS = 16
SETTLED_START_S = 5.0
FIG_NAME = "simulation_multi_initial"


def main() -> None:
    run_dir = sim.new_run_dir("multi_initial")
    print(f"Run folder: {run_dir}", flush=True)
    archived = json.loads(
        (sim.latest_run_dir("preview") / "current_theory_results.json").read_text(
            encoding="utf-8")
    )

    print("regenerating direct snapshots and Route II synthesis", flush=True)
    data = sim.collect_snapshots(K=800)
    epsilon = {2: 0.80, 3: 3.00}
    lambdas = {2: -8.0 * np.eye(2), 3: -30.0 * np.eye(2)}
    synthesis = {
        k: sim.synthesize_block(k, data[k], lambdas[k], sim.DBAR[k], epsilon[k])
        for k in (2, 3)
    }
    for k in (2, 3):
        ref = archived["synthesis"][str(k)]
        for key in ("route_ii_p_norm", "route_ii_certificate_max"):
            if abs(float(synthesis[k][key]) - ref[key]) > 1.0e-9:
                raise AssertionError(f"Block {k}: {key} disagrees with the archive")
    W2 = np.asarray(synthesis[2]["W"])
    W3 = np.asarray(synthesis[3]["W"])

    sim.reference = prev.stopped_reference
    sim.outer_command = prev.make_preview_outer_command(
        prev.PAPER_PREVIEW_HORIZON,
        k_rho=prev.PAPER_K_RHO,
        k_alpha=prev.PAPER_K_ALPHA,
    )

    rng = np.random.default_rng(20260731)
    offsets = np.column_stack(
        [
            rng.uniform(-0.30, 0.30, N_TRIALS),
            rng.uniform(-0.20, 0.20, N_TRIALS),
            rng.uniform(-0.30, 0.30, N_TRIALS),
        ]
    )
    currents = rng.uniform(-0.75, 0.75, size=(N_TRIALS, 2))

    horizon = prev.PATH_LENGTH / 2.0 / prev.REFERENCE_CRUISE_SPEED
    trials = []
    for index, (offset, current) in enumerate(zip(offsets, currents)):
        run = sim.simulate(
            W2, W3, adapt=True, disturbance_scale=1.0,
            horizon=horizon, dt=2.0e-3,
            pose_offset=offset, initial_current=current,
            voltage_limit=sim.P.Umax,
            outer_mode="preview_terminal_capture",
            actuator_map="hard_clip",
        )
        for value in run.values():
            if isinstance(value, (np.ndarray, float)) and not np.all(np.isfinite(value)):
                raise FloatingPointError(f"Nonfinite result in trial {index}")
        t = run["t"]
        position = np.linalg.norm(run["pose"][:, :2] - run["qr"][:, :2], axis=1)
        settled = t >= SETTLED_START_S
        summary = {
            "trial": index,
            "pose_offset": offset.tolist(),
            "initial_current": current.tolist(),
            "final_error_m": float(position[-1]),
            "settled_rmse_m": float(np.sqrt(np.mean(position[settled] ** 2))),
            "voltage_peak_v": float(run["u_peak"]),
        }
        trials.append(summary)
        np.savez_compressed(run_dir / f"trial_{index:02d}.npz", **run)
        print(
            f"trial {index:02d}: final={1e3 * summary['final_error_m']:.1f} mm, "
            f"settled RMSE={1e3 * summary['settled_rmse_m']:.1f} mm, "
            f"Upeak={summary['voltage_peak_v']:.3f} V",
            flush=True,
        )

    report = {
        "status": "completed",
        "configuration": {
            "model": "previewed terminal-capture closed loop, paper configuration",
            "preview_horizon_s": prev.PAPER_PREVIEW_HORIZON,
            "k_rho": prev.PAPER_K_RHO,
            "k_alpha": prev.PAPER_K_ALPHA,
            "voltage_limit_v": sim.P.Umax,
            "scope": "half-loop traversal (right lobe, phi in [0, pi])",
            "horizon_s": horizon,
            "integration_step_s": 2.0e-3,
            "disturbance_scale": 1.0,
            "offset_distribution_m_rad": [-0.30, 0.30, -0.20, 0.20, -0.30, 0.30],
            "current_distribution_a": [-0.75, 0.75],
            "rng_seed": 20260731,
            "n_trials": N_TRIALS,
        },
        "trials": trials,
    }
    with (run_dir / "multi_initial_position_results.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=True, indent=2)
    sim.update_latest("multi_initial", run_dir)

    for trial in trials:
        if not (trial["voltage_peak_v"] <= sim.P.Umax + 1.0e-9):
            raise AssertionError(f"Trial {trial['trial']} exceeded the voltage limit")
    print(f"Results written to {run_dir / 'multi_initial_position_results.json'}", flush=True)
    print(
        "Plotting (independent): python plot_figures.py multi_initial "
        f"[--run {run_dir.name}]", flush=True,
    )


def make_figure(initial_xy: np.ndarray, runs: list[dict[str, np.ndarray]], out_pdf) -> None:
    """Draw the paper figure from per-trial traces."""
    sim.set_plot_style()
    plt.rcParams.update(
        {
            "font.size": 12.9,
            "axes.labelsize": 12.9,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
        }
    )
    fig, ax = plt.subplots(figsize=(2.55, 2.68))
    phi = np.linspace(0.0, np.pi, 1001)
    reference = np.column_stack(
        [
            sim.REFERENCE_A * np.sin(phi),
            sim.REFERENCE_B * np.sin(2.0 * phi),
        ]
    )
    ax.plot(
        reference[:, 0], reference[:, 1],
        linestyle=(0, (5, 4)), color="#222222", linewidth=0.9,
        label="reference", zorder=3,
    )
    for index in range(N_TRIALS):
        run = runs[index]
        ax.plot(
            run["pose"][:, 0], run["pose"][:, 1],
            color="#0072B2", alpha=0.45, linewidth=0.7, zorder=2,
        )
    ax.scatter(
        initial_xy[:, 0], initial_xy[:, 1],
        s=9, facecolor="#0072B2", edgecolor="white", linewidth=0.4,
        label="initial poses", zorder=5,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.5, 3.2)
    ax.set_ylim(-1.7, 1.7)
    ax.set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0])
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.tick_params(pad=1.5)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(
        frameon=False, loc="lower left", fontsize=9.3,
        handlelength=1.2, borderpad=0.2, labelspacing=0.18,
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
