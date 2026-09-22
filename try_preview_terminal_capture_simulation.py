# -*- coding: utf-8 -*-
"""Preview point capture with a smooth terminal polar-capture phase.

This simulation keeps the current data-driven MIMO inner loop,
offline synthesis, bounded online disturbances, DSC filters, adaptation, and
hard voltage constraint unchanged.  The figure-eight is parameterized by arc
length, traversed at constant Cartesian speed, and brought smoothly to rest by
a terminal half-cosine speed profile.  During motion, the outer loop captures
``q_r(min(t + T_p, T_f))``.  Once the preview reaches the terminal time, the
target remains fixed, so the same polar law becomes a terminal point regulator.
"""

from __future__ import annotations

import json
import sys
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import current_theory_simulation as sim


REFERENCE_CRUISE_SPEED = 0.24
TERMINAL_DECEL_TIME = 8.0
POST_STOP_TIME = 15.0
SCREEN_DT = 2.0e-3
FULL_DT = 1.0e-3
MOVING_METRIC_START = 10.0
TERMINAL_STOP_RADIUS = 0.010
TERMINAL_BLEND_RADIUS = 0.030
PAPER_K_RHO = 0.8
PAPER_K_ALPHA = 0.7
PAPER_PREVIEW_HORIZON = 1.4
TABLE_HORIZONS = (1.0, 1.2, 1.4, 1.6)

ARC_PARAMETER = np.linspace(0.0, 2.0 * np.pi, 200_001)
ARC_DX = sim.REFERENCE_A * np.cos(ARC_PARAMETER)
ARC_DY = 2.0 * sim.REFERENCE_B * np.cos(2.0 * ARC_PARAMETER)
ARC_DS_DU = np.hypot(ARC_DX, ARC_DY)
ARC_LENGTH = np.concatenate(
    (
        np.zeros(1),
        np.cumsum(
            0.5
            * (ARC_DS_DU[1:] + ARC_DS_DU[:-1])
            * np.diff(ARC_PARAMETER)
        ),
    )
)
PATH_LENGTH = float(ARC_LENGTH[-1])
CRUISE_END_TIME = PATH_LENGTH / REFERENCE_CRUISE_SPEED - 0.5 * TERMINAL_DECEL_TIME
REFERENCE_END_TIME = CRUISE_END_TIME + TERMINAL_DECEL_TIME
HORIZON = REFERENCE_END_TIME + POST_STOP_TIME


def reference_progress(t: float) -> tuple[float, float]:
    """Return traveled arc length and its C1 terminal speed profile."""
    time = max(float(t), 0.0)
    if time <= CRUISE_END_TIME:
        return REFERENCE_CRUISE_SPEED * time, REFERENCE_CRUISE_SPEED
    if time < REFERENCE_END_TIME:
        tau = time - CRUISE_END_TIME
        phase = np.pi * tau / TERMINAL_DECEL_TIME
        speed = 0.5 * REFERENCE_CRUISE_SPEED * (1.0 + np.cos(phase))
        distance = REFERENCE_CRUISE_SPEED * (
            CRUISE_END_TIME
            + 0.5 * tau
            + TERMINAL_DECEL_TIME * np.sin(phase) / (2.0 * np.pi)
        )
        return min(float(distance), PATH_LENGTH), float(speed)
    return PATH_LENGTH, 0.0


def stopped_reference(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Traverse the figure-eight at constant speed, then stop smoothly."""
    distance, speed = reference_progress(t)
    parameter = float(np.interp(distance, ARC_LENGTH, ARC_PARAMETER))
    dx_du = sim.REFERENCE_A * np.cos(parameter)
    dy_du = 2.0 * sim.REFERENCE_B * np.cos(2.0 * parameter)
    ddx_du2 = -sim.REFERENCE_A * np.sin(parameter)
    ddy_du2 = -4.0 * sim.REFERENCE_B * np.sin(2.0 * parameter)
    tangent_norm = float(np.hypot(dx_du, dy_du))
    curvature = (
        dx_du * ddy_du2 - dy_du * ddx_du2
    ) / tangent_norm**3
    qr = np.array(
        [
            sim.REFERENCE_A * np.sin(parameter),
            sim.REFERENCE_B * np.sin(2.0 * parameter),
            np.arctan2(dy_du, dx_du),
        ]
    )
    return qr, np.array([speed, curvature * speed])


def make_preview_outer_command(
    preview_horizon: float,
    *,
    k_rho: float = sim.OUTER_K_RHO,
    k_alpha: float = sim.OUTER_K_ALPHA,
    capture_speed: float = sim.OUTER_CAPTURE_SPEED,
) -> Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Create the outer command used by ``sim.simulate``."""
    if preview_horizon < 0.0 or not np.isfinite(preview_horizon):
        raise ValueError("preview_horizon must be finite and nonnegative")
    if k_rho <= 0.0 or not np.isfinite(k_rho):
        raise ValueError("k_rho must be finite and strictly positive")
    if k_alpha <= 0.0 or not np.isfinite(k_alpha):
        raise ValueError("k_alpha must be finite and strictly positive")
    if capture_speed <= 0.0 or not np.isfinite(capture_speed):
        raise ValueError("capture_speed must be finite and strictly positive")

    def command(
        pose: np.ndarray,
        t: float,
        kx: float = 4.0,
        ky: float = 12.0,
        kth: float = 5.0,
        mode: str = "preview_terminal_capture",
        capture_gain: float = 4.0,
        lookahead: float = 0.04,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del kx, ky, kth, mode, capture_gain, lookahead
        qr, _ = stopped_reference(t)
        preview_time = min(max(t, 0.0) + preview_horizon, REFERENCE_END_TIME)
        preview_target, _ = stopped_reference(preview_time)
        alpha1, polar_error, preview_error = (
            sim.polar_capture_command_from_reference(
                pose,
                preview_target,
                k_rho=k_rho,
                k_alpha=k_alpha,
                capture_speed=capture_speed,
            )
        )
        preview_at_terminal = preview_time >= REFERENCE_END_TIME - 1.0e-12
        preview_distance = float(np.linalg.norm(preview_error))
        if preview_at_terminal:
            terminal_scale = float(
                np.clip(
                    (preview_distance - TERMINAL_STOP_RADIUS)
                    / (TERMINAL_BLEND_RADIUS - TERMINAL_STOP_RADIUS),
                    0.0,
                    1.0,
                )
            )
            alpha1 = terminal_scale * alpha1
        synchronized_error = qr[:2] - pose[:2]
        error = np.array(
            [synchronized_error[0], synchronized_error[1], polar_error[2]]
        )
        return alpha1, error, qr, preview_error

    return command


def simulate_ideal_kinematics(
    preview_horizon: float,
    dt: float = SCREEN_DT,
) -> dict[str, np.ndarray]:
    """Screen one preview horizon using the ideal unicycle kinematics."""
    command = make_preview_outer_command(preview_horizon)
    q0, _ = stopped_reference(0.0)
    pose = q0 + np.array([1.0, 0.0, 0.15])
    stride = max(1, int(round(0.01 / dt)))
    t_log: list[float] = []
    pose_log: list[np.ndarray] = []
    qr_log: list[np.ndarray] = []
    alpha_log: list[np.ndarray] = []

    def derivative(state: np.ndarray, tt: float) -> np.ndarray:
        alpha1, _, _, _ = command(state, tt)
        return np.array(
            [
                alpha1[0] * np.cos(state[2]),
                alpha1[0] * np.sin(state[2]),
                alpha1[1],
            ]
        )

    nsteps = int(np.ceil(HORIZON / dt))
    for index in range(nsteps + 1):
        t = min(index * dt, HORIZON)
        if index % stride == 0 or index == nsteps:
            alpha1, _, qr, _ = command(pose, t)
            t_log.append(t)
            pose_log.append(pose.copy())
            qr_log.append(qr.copy())
            alpha_log.append(alpha1.copy())
        if index == nsteps:
            break
        step = min(dt, HORIZON - t)
        k1 = derivative(pose, t)
        k2 = derivative(pose + 0.5 * step * k1, t + 0.5 * step)
        k3 = derivative(pose + 0.5 * step * k2, t + 0.5 * step)
        k4 = derivative(pose + step * k3, t + step)
        pose = pose + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        pose[2] = sim.wrap_angle(float(pose[2]))

    return {
        "t": np.asarray(t_log),
        "pose": np.asarray(pose_log),
        "qr": np.asarray(qr_log),
        "alpha1": np.asarray(alpha_log),
    }


def trajectory_metrics(run: dict[str, np.ndarray | float]) -> dict[str, float]:
    t = np.asarray(run["t"])
    pose = np.asarray(run["pose"])
    qr = np.asarray(run["qr"])
    error = np.linalg.norm(pose[:, :2] - qr[:, :2], axis=1)
    moving = (t >= MOVING_METRIC_START) & (t < REFERENCE_END_TIME)
    late_moving = (t >= REFERENCE_END_TIME - 20.0) & (t < REFERENCE_END_TIME)
    stopped = t >= REFERENCE_END_TIME
    result = {
        "moving_rmse_m": float(np.sqrt(np.mean(error[moving] ** 2))),
        "late_moving_rmse_m": float(np.sqrt(np.mean(error[late_moving] ** 2))),
        "post_stop_rmse_m": float(np.sqrt(np.mean(error[stopped] ** 2))),
        "final_error_m": float(error[-1]),
        "maximum_error_m": float(np.max(error)),
    }
    if "U" in run:
        result.update(
            {
                "voltage_peak_v": float(run["u_peak"]),
                "commanded_voltage_peak_v": float(run["uc_peak"]),
                "saturation_residual_peak_v": float(run["du_peak"]),
                "voltage_saturation_fraction": float(run["sat_fraction"]),
                "z2_rms_tail": float(run["z2_rms_tail"]),
                "z3_rms_tail": float(run["z3_rms_tail"]),
            }
        )
    return result


def run_full_mimo(
    W2: np.ndarray,
    W3: np.ndarray,
    preview_horizon: float,
    *,
    k_rho: float = sim.OUTER_K_RHO,
    k_alpha: float = sim.OUTER_K_ALPHA,
    capture_speed: float = sim.OUTER_CAPTURE_SPEED,
    dt: float = FULL_DT,
) -> dict[str, np.ndarray | float]:
    sim.reference = stopped_reference
    sim.outer_command = make_preview_outer_command(
        preview_horizon,
        k_rho=k_rho,
        k_alpha=k_alpha,
        capture_speed=capture_speed,
    )
    return sim.simulate(
        W2,
        W3,
        adapt=True,
        disturbance_scale=1.0,
        horizon=HORIZON,
        dt=dt,
        voltage_limit=sim.P.Umax,
        outer_mode="preview_terminal_capture",
        actuator_map="hard_clip",
    )


def plot_comparison(
    baseline: dict[str, np.ndarray | float],
    candidate: dict[str, np.ndarray | float],
    out_pdf,
) -> None:
    sim.set_plot_style()
    plt.rcParams.update(
        {
            "font.size": 10.8,
            "axes.labelsize": 10.8,
            "legend.fontsize": 9.6,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
        }
    )
    blue, orange, green, gray = "#0072B2", "#D55E00", "#009E73", "#666666"
    panel_titles = ("(a) Trajectories", "(b) Position error",
                    "(c) Velocities", "(d) Voltages")
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.15))
    axes = axes.ravel()
    for sub_ax, title in zip(axes, panel_titles):
        sub_ax.set_title(title, fontsize=9.6, pad=8)

    t1 = np.asarray(candidate["t"])
    q1 = np.asarray(candidate["qr"])
    p1 = np.asarray(candidate["pose"])
    e1 = np.linalg.norm(q1[:, :2] - p1[:, :2], axis=1)

    ax = axes[0]
    # Identical data coordinates: distinguish near-coincident curves by layering.
    ax.plot(p1[:, 0], p1[:, 1], color=blue, linewidth=1.1,
            label="trajectory", zorder=2)
    # Direction arrows along the vehicle trajectory in time order.
    for frac in (0.12, 0.30, 0.52, 0.75):
        base = int(frac * (p1.shape[0] - 1))
        span = min(120, p1.shape[0] - 1 - base)
        ax.annotate(
            "", xy=p1[base + span, :2], xytext=p1[base, :2],
            arrowprops={"arrowstyle": "-|>", "color": blue, "lw": 0.9,
                        "mutation_scale": 6.0},
            zorder=4,
        )
    ax.plot(q1[:, 0], q1[:, 1], linestyle=(0, (5, 4)), color="#222222",
            linewidth=0.85, label="reference", zorder=3)
    # The hollow diamond marks the actual vehicle start pose and the hollow
    # square the terminal reference point; distinct hollow shapes avoid
    # single-channel color coding against the blue/black line palette.
    ax.scatter(p1[0, 0], p1[0, 1], marker="D", facecolors="none",
               edgecolors=green, linewidths=0.9, s=9,
               label="start", zorder=6)
    ax.scatter(q1[-1, 0], q1[-1, 1], marker="s", facecolors="none",
               edgecolors="red", linewidths=0.9, s=12,
               label="terminal", zorder=6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-2.1, 4.5)
    ax.set_xticks([-2.0, 0.0, 2.0])
    ax.set_yticks([-2.0, 0.0, 2.0, 4.0])
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)", labelpad=2)
    ax.tick_params(pad=1.5)
    ax.legend(
        frameon=False, loc="upper right", fontsize=7.8, ncol=1,
        handlelength=1.2, borderpad=0.2, labelspacing=0.18,
        borderaxespad=0.2,
    )
    ax.grid(True, linestyle=":", linewidth=0.5)

    ax = axes[1]
    ax.plot(t1, e1, color=blue, label=r"$\|\boldsymbol{e}_r\|$")
    ax.set_xlim(0.0, REFERENCE_END_TIME)
    ax.set_xticks([0.0, 20.0, 40.0, 60.0, 80.0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("position error (m)", labelpad=2)
    ax.tick_params(pad=1.5)
    ax.legend(
        frameon=False, loc="upper right", fontsize=7.8,
        handlelength=1.2, borderpad=0.2, labelspacing=0.18,
    )
    ax.set_box_aspect(1)
    ax.grid(True, linestyle=":", linewidth=0.5)

    ax = axes[2]
    alpha1 = np.asarray(candidate["alpha1"])
    velocity = np.asarray(candidate["V"])
    v_scale = float(max(np.abs(alpha1[:, 0]).max(), np.abs(velocity[:, 0]).max()))
    w_scale = float(max(np.abs(alpha1[:, 1]).max(), np.abs(velocity[:, 1]).max()))
    ax.plot(t1, velocity[:, 0] / v_scale, color=blue, linewidth=1.7, alpha=0.7, label=r"$v$")
    ax.plot(t1, velocity[:, 1] / w_scale, color=orange, linewidth=1.7, alpha=0.7, label=r"$\omega$")
    ax.plot(t1, alpha1[:, 0] / v_scale, linestyle=(0, (5, 4)), color="#123D68",
            linewidth=0.9, zorder=4, label=r"$v_c$")
    ax.plot(t1, alpha1[:, 1] / w_scale, linestyle=(0, (3, 3)), color="#81380D",
            linewidth=0.9, zorder=4, label=r"$\omega_c$")
    ax.set_xlim(0.0, REFERENCE_END_TIME)
    ax.set_xticks([0.0, 20.0, 40.0, 60.0, 80.0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("normalized velocity", labelpad=2)
    ax.set_ylim(-1.0, 2.0)
    ax.set_box_aspect(1)
    ax.tick_params(pad=1.5)
    ax.legend(frameon=False, loc="upper right", fontsize=7.8, ncol=2,
              handlelength=1.2, columnspacing=0.7, borderpad=0.2,
              labelspacing=0.18)
    ax.grid(True, linestyle=":", linewidth=0.5)

    ax = axes[3]
    voltage = np.asarray(candidate["U"])
    ax.plot(t1, voltage[:, 0], color=blue, linewidth=1.1, label=r"$u_R$")
    ax.plot(t1, voltage[:, 1], color=orange, linewidth=1.0, linestyle=(0, (5, 3)),
            label=r"$u_L$")
    # The dash-dotted lines show the hard input constraint used by the run.
    display_voltage_limit = 4.0
    ax.axhline(display_voltage_limit, color="black", linestyle=(0, (4, 3)),
               linewidth=0.8, zorder=1)
    ax.axhline(-display_voltage_limit, color="black", linestyle=(0, (4, 3)),
               linewidth=0.8, zorder=1)
    ax.set_ylim(-6.0, 10.0)
    ax.set_yticks([-4.0, 0.0, 4.0, 8.0])
    ax.set_xlim(0.0, REFERENCE_END_TIME)
    ax.set_xticks([0.0, 20.0, 40.0, 60.0, 80.0])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("voltage (V)", labelpad=2)
    ax.tick_params(pad=1.5)
    ax.set_box_aspect(1)
    ax.grid(True, linestyle=":", linewidth=0.5)

    fig.tight_layout(w_pad=0.9, h_pad=0.4, pad=0.4)

    # Panel (d) uses a manually laid out two-row frameless legend: row 1 holds
    # u_R and u_L at normal spacing, row 2 holds the voltage limit aligned with
    # the left edge of row 1; rows are pinned to the upper right of the panel.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(renderer)
    pt_px = fig.dpi / 72.0
    fs = 7.8
    sample_px = 1.6 * fs * pt_px
    hpad_px = 0.5 * fs * pt_px
    col_px = 0.8 * fs * pt_px
    row_h_px = 1.35 * fs * pt_px
    gap_px = 0.4 * fs * pt_px
    border_px = 0.45 * fs * pt_px

    def _text_w(label_text: str) -> float:
        probe = ax.text(0.0, 0.0, label_text, fontsize=fs, transform=ax.transAxes)
        width = probe.get_window_extent(renderer).width
        probe.remove()
        return width

    label_w = [_text_w(s) for s in (r"$u_R$", r"$u_L$", "voltage limit")]
    row1_px = (sample_px + hpad_px + label_w[0] + col_px
               + sample_px + hpad_px + label_w[1])
    row2_px = sample_px + hpad_px + label_w[2]
    box_w = max(row1_px, row2_px) + 2.0 * border_px
    box_h = 2.0 * row_h_px + gap_px + 2.0 * border_px
    x1 = axbb.x1 - 3.0 * pt_px
    y1 = axbb.y1 - 0.5 * pt_px
    x0 = x1 - box_w
    y0 = y1 - box_h
    fx = lambda px_: (px_ - axbb.x0) / axbb.width
    fy = lambda px_: (px_ - axbb.y0) / axbb.height
    left = x0 + border_px
    y_row1 = y1 - 0.2 * fs * pt_px - 0.5 * row_h_px
    y_row2 = y_row1 - (row_h_px + gap_px)
    layout = (
        (left, y_row1,
         dict(color=blue, linewidth=1.1, linestyle="-"), r"$u_R$", label_w[0]),
        (left + sample_px + hpad_px + label_w[0] + col_px, y_row1,
         dict(color=orange, linewidth=1.0, linestyle=(0, (5, 3))), r"$u_L$", label_w[1]),
        (left, y_row2,
         dict(color="black", linewidth=0.8, linestyle=(0, (4, 3))), "voltage limit", label_w[2]),
    )
    for sample_x, sample_y, style, label_text, width in layout:
        ax.plot([fx(sample_x), fx(sample_x + sample_px)], [fy(sample_y), fy(sample_y)],
                transform=ax.transAxes, clip_on=False, zorder=7, **style)
        ax.text(fx(sample_x + sample_px + hpad_px), fy(sample_y), label_text,
                fontsize=fs, transform=ax.transAxes, clip_on=False, zorder=7,
                ha="left", va="center")

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    run_dir = sim.new_run_dir("preview")
    print(f"Run folder: {run_dir}", flush=True)

    print("regenerating direct snapshots and synthesis certificates", flush=True)
    data = sim.collect_snapshots(K=800)
    epsilon = {2: 0.80, 3: 3.00}
    lambdas = {2: -8.0 * np.eye(2), 3: -30.0 * np.eye(2)}
    synthesis = {
        k: sim.synthesize_block(k, data[k], lambdas[k], sim.DBAR[k], epsilon[k])
        for k in (2, 3)
    }
    W2 = np.asarray(synthesis[2]["W"])
    W3 = np.asarray(synthesis[3]["W"])
    mc = {
        k: sim.monte_carlo_margin(
            synthesis[k]["Psol"], synthesis[k]["nominal_matrix"], sim.DBAR[k], 800,
            draws=2000, seed=20 + k,
        )
        for k in (2, 3)
    }
    for k, block in synthesis.items():
        if (block["rank"] != block["rows"]
                or block["match_residual"] > 1.0e-7
                or block["route_i_phi_min"] < 0.0
                or block["route_ii_spectral_margin"] <= 0.0
                or block["route_ii_certificate_max"] > 0.0
                or block["xi_max"] > 0.0
                or mc[k].max() > -epsilon[k] + 1.0e-8):
            raise AssertionError(f"Block {k} failed the synthesis checks")

    print("running full MIMO baseline", flush=True)
    baseline = run_full_mimo(
        W2, W3, preview_horizon=0.0, k_rho=PAPER_K_RHO, k_alpha=PAPER_K_ALPHA,
    )
    print("running full MIMO preview candidate", flush=True)
    candidate = run_full_mimo(
        W2, W3, preview_horizon=PAPER_PREVIEW_HORIZON,
        k_rho=PAPER_K_RHO, k_alpha=PAPER_K_ALPHA,
    )
    baseline_metrics = trajectory_metrics(baseline)
    candidate_metrics = trajectory_metrics(candidate)
    for run in (baseline, candidate):
        for value in run.values():
            if isinstance(value, (np.ndarray, float)) and not np.all(np.isfinite(value)):
                raise FloatingPointError("Nonfinite closed-loop result")
        if run["u_peak"] > sim.P.Umax + 1.0e-9:
            raise AssertionError("Applied voltage exceeds the hard constraint")

    print("running preview-horizon table (T_p = 1.0/1.2/1.4/1.6 s)", flush=True)
    table_rows = []
    for horizon in TABLE_HORIZONS:
        if np.isclose(horizon, PAPER_PREVIEW_HORIZON):
            run = candidate
        else:
            run = run_full_mimo(
                W2, W3, preview_horizon=horizon,
                k_rho=PAPER_K_RHO, k_alpha=PAPER_K_ALPHA,
            )
            for value in run.values():
                if isinstance(value, (np.ndarray, float)) and not np.all(np.isfinite(value)):
                    raise FloatingPointError(f"Nonfinite closed-loop result at T_p = {horizon} s")
            if run["u_peak"] > sim.P.Umax + 1.0e-9:
                raise AssertionError("Applied voltage exceeds the hard constraint")
        position = np.linalg.norm(run["pose"][:, :2] - run["qr"][:, :2], axis=1)
        t = run["t"]
        full_mask = t <= REFERENCE_END_TIME
        ss_mask = (t >= 5.0) & (t <= REFERENCE_END_TIME)
        row = {
            "preview_horizon_s": horizon,
            "p_rmse_m": float(np.sqrt(np.mean(position[full_mask] ** 2))),
            "ss_rmse_m": float(np.sqrt(np.mean(position[ss_mask] ** 2))),
        }
        table_rows.append(row)
        print(
            f"T_p={horizon}: P-RMSE={1e3 * row['p_rmse_m']:.2f} mm, "
            f"SS-RMSE={1e3 * row['ss_rmse_m']:.2f} mm",
            flush=True,
        )

    report = {
        "status": "completed",
        "scope": "official previewed terminal-capture simulation",
        "configuration": {
            "reference": "arc-length-uniform figure-eight with smooth terminal deceleration and stationary hold",
            "path_length_m": PATH_LENGTH,
            "cruise_speed_mps": REFERENCE_CRUISE_SPEED,
            "cruise_end_time_s": CRUISE_END_TIME,
            "terminal_deceleration_time_s": TERMINAL_DECEL_TIME,
            "reference_end_time_s": REFERENCE_END_TIME,
            "post_stop_time_s": POST_STOP_TIME,
            "horizon_s": HORIZON,
            "full_integration_step_s": FULL_DT,
            "online_disturbance_scale": 1.0,
            "offline_data_disturbance": "bounded process disturbances ||d_k|| <= dbar_k active during data generation; D_k remains unknown to the synthesis and enters only through measured Z_k",
            "voltage_limit_v": sim.P.Umax,
            "k_rho": PAPER_K_RHO,
            "k_alpha": PAPER_K_ALPHA,
            "capture_speed_limit_mps": sim.OUTER_CAPTURE_SPEED,
            "terminal_stop_radius_m": TERMINAL_STOP_RADIUS,
            "terminal_blend_radius_m": TERMINAL_BLEND_RADIUS,
            "selected_preview_horizon_s": PAPER_PREVIEW_HORIZON,
            "initial_current_a": [0.0, 0.0],
            "selection_rule": "fixed paper outer-loop parameters with a 4-V hard limit and zero initial current",
        },
    }
    synthesis_keys = (
        "selected_route", "rank", "rows", "cond", "match_residual",
        "right_inverse_residual",
        "xi_max", "actual_sym_max", "data_equation_residual",
        "route_i_phi_min", "route_i_closed_xi_max",
        "route_i_closed_match", "route_i_closed_p_norm",
        "route_ii_sigma_min", "route_ii_spectral_margin",
        "route_ii_kappa_lower", "route_ii_kappa",
        "route_ii_certificate_max", "route_ii_match", "route_ii_p_norm",
    )
    official_report = {
        "status": "completed",
        "configuration": {
            "K": 800,
            "data_step_s": sim.DATA_DT,
            "sampling_interval_s": sim.DATA_SAMPLE_PERIOD,
            "snapshot_definition": "direct_derivative_samples",
            "offline_data_disturbance": "bounded process disturbances ||d_k|| <= dbar_k active during data generation; D_k remains unknown to the synthesis and enters only through measured Z_k",
            "selected_synthesis_route": "II",
            "initial_transient_s": sim.DATA_INITIAL_TRANSIENT,
            "friction_basis": "sgn",
            "dbar2": sim.DBAR[2], "dbar3": sim.DBAR[3],
            "epsilon2": epsilon[2], "epsilon3": epsilon[3],
            "route_ii_kappa2": sim.ROUTE_II_KAPPA[2],
            "route_ii_kappa3": sim.ROUTE_II_KAPPA[3],
            "Umax": sim.P.Umax,
            "actuator_map": "hard_clip",
            "outer_loop": "previewed_polar_terminal_capture",
            "outer_k_rho": PAPER_K_RHO,
            "outer_k_alpha": PAPER_K_ALPHA,
            "outer_capture_speed_mps": sim.OUTER_CAPTURE_SPEED,
            "polar_capture_epsilon_m": sim.POLAR_CAPTURE_EPS,
            "preview_horizon_s": PAPER_PREVIEW_HORIZON,
            **report["configuration"],
        },
        "synthesis": {
            str(k): {key: sim.scalarize(synthesis[k][key]) for key in synthesis_keys}
            | {"mc_max": float(mc[k].max()), "mc_mean": float(mc[k].mean())}
            for k in (2, 3)
        },
        "preview_horizon_table": table_rows,
        "closed_loop": {
            "no_preview": baseline_metrics,
            "preview_terminal_capture": candidate_metrics,
        },
    }
    official_path = run_dir / "current_theory_results.json"
    with official_path.open("w", encoding="utf-8") as stream:
        json.dump(official_report, stream, ensure_ascii=True, indent=2)

    for name, run in (("no_preview", baseline), ("preview_terminal_capture", candidate)):
        np.savez_compressed(run_dir / f"{name}_trace.npz", **run)
    sim.update_latest("preview", run_dir)

    print("Full MIMO comparison", flush=True)
    for name, metrics in (("no preview", baseline_metrics), ("preview", candidate_metrics)):
        print(
            f"{name:10s}: moving={1e3 * metrics['moving_rmse_m']:.2f} mm, "
            f"post-stop={1e3 * metrics['post_stop_rmse_m']:.2f} mm, "
            f"final={1e3 * metrics['final_error_m']:.3f} mm, "
            f"Upeak={metrics['voltage_peak_v']:.3f} V, "
            f"sat={100 * metrics['voltage_saturation_fraction']:.2f}%",
            flush=True,
        )
    print(f"Official results: {official_path}", flush=True)
    print(
        "Plotting (independent): python plot_figures.py tracking "
        f"[--run {run_dir.name}]", flush=True,
    )


if __name__ == "__main__":
    main()
