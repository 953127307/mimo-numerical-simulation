# -*- coding: utf-8 -*-
"""Numerical validation for the current data-driven MIMO theory in main.tex.

This script intentionally does not reuse the legacy ``mimo_dbc.py`` synthesis,
which implements an earlier kernel-space pole-placement formula.  The routines
below follow the current manuscript literally:

    Y_k P_k = C_k,   G_k Q_k = [0; R_k],
    W_k = X_{k+1} [P_k Q_k],
    A_cl,k = (Z_k - D_k) P_k.

The off-line dataset contains direct samples of ``dot(Z_k)``, ``X_{k+1}``,
and ``Y_k`` at the selected instants.  No finite-difference or interval-average
surrogate is used for the derivative matrix.  The closed-loop weights are
synthesized with the constructive null-space selector (Route II).  The legacy
Route I data-LMI feasibility quantities are still evaluated as an internal
cross-check of data consistency only; the manuscript no longer references
them.

It also implements the current adaptive law with the known direction matrices
T_2 and T_3, a bounded polar point-capture outer loop for the moving Cartesian
target, and the componentwise hard voltage constraint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT.parent / "submission" / "figures"
RESULT_DIR = ROOT / "simulation_results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = RESULT_DIR / "runs"
LATEST_POINTER = {
    "preview": RESULT_DIR / "latest_preview.txt",
    "multi_initial": RESULT_DIR / "latest_multi_initial.txt",
}


def new_run_dir(tag: str) -> Path:
    """Create a timestamped archive folder for one simulation run."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def update_latest(kind: str, run_dir: Path) -> None:
    """Point `kind`-type plotting at the newest run folder."""
    LATEST_POINTER[kind].write_text(run_dir.name + "\n", encoding="utf-8")


def latest_run_dir(kind: str) -> Path:
    name = LATEST_POINTER[kind].read_text(encoding="utf-8").strip()
    return RUNS_DIR / name


@dataclass(frozen=True)
class Plant:
    M: float = 1.50
    J: float = 0.020
    r: float = 0.0325
    track: float = 0.150
    N: float = 30.0
    Kt: float = 0.0115
    Ke: float = 0.0115
    Ra: float = 2.0
    La: float = 0.050
    Bv: float = 0.50
    Bw: float = 0.050
    Fc: float = 0.30
    tauc: float = 0.020
    Umax: float = 4.0

    @property
    def H2(self) -> np.ndarray:
        kf = self.Kt * self.N / self.r
        return np.array(
            [
                [kf / self.M, kf / self.M],
                [kf * self.track / (2.0 * self.J), -kf * self.track / (2.0 * self.J)],
            ]
        )

    @property
    def H3(self) -> np.ndarray:
        return np.eye(2) / self.La


P = Plant()
T2 = 0.5 * np.array([[1.0, 1.0], [1.0, -1.0]])
T3 = np.eye(2)
# Amplitudes of the bounded process disturbances present during both offline
# data generation and online operation; they satisfy ||d_k(t)|| <= DBAR[k],
# i.e., Assumption 2.  The synthesis never sees a disturbance matrix: the
# sampled disturbances enter only through the measured Z_k.
DBAR = {2: 0.10, 3: 0.10}
ROUTE_II_KAPPA = {2: 6.0, 3: 10.0}
DATA_D2_PHASE = 0.3
DATA_D3_PHASE = 1.1
DATA_DT = 5.0e-4
DATA_SAMPLE_PERIOD = 0.20
DATA_INITIAL_TRANSIENT = 1.0
REFERENCE_A = 3.0
REFERENCE_B = 1.5
REFERENCE_NU = 0.08
OUTER_K_RHO = 0.80
OUTER_K_ALPHA = 1.50
OUTER_CAPTURE_SPEED = 0.50
POLAR_CAPTURE_EPS = 1.0e-9


def wheel_speeds(V: np.ndarray) -> np.ndarray:
    v, w = V
    return np.array([v + 0.5 * P.track * w, v - 0.5 * P.track * w])


def sign_selection(value: float | np.ndarray) -> float | np.ndarray:
    """Paper Coulomb-friction sign basis shared by the plant and regressor."""
    return np.sign(np.asarray(value))


def sigma2(V: np.ndarray, dbeta1: np.ndarray) -> np.ndarray:
    v, w = V
    return np.array(
        [v, w, sign_selection(v), sign_selection(w), dbeta1[0], dbeta1[1]]
    )


def sigma3(V: np.ndarray, I: np.ndarray, dbeta2: np.ndarray) -> np.ndarray:
    vr, vl = wheel_speeds(V)
    return np.array([I[0], I[1], vr, vl, dbeta2[0], dbeta2[1]])


def F2(V: np.ndarray) -> np.ndarray:
    v, w = V
    return np.array(
        [
            -(P.Bv * v + P.Fc * sign_selection(v)) / P.M,
            -(P.Bw * w + P.tauc * sign_selection(w)) / P.J,
        ]
    )


def F3(V: np.ndarray, I: np.ndarray) -> np.ndarray:
    return -(P.Ra / P.La) * I - (P.Ke * P.N / (P.La * P.r)) * wheel_speeds(V)


def known_rho(block: int) -> np.ndarray:
    """Plant coefficient matrix used only to verify the simulated data equation."""
    if block == 2:
        return np.array(
            [
                [0.0, 0.0, -P.Bv / P.M, 0.0, -P.Fc / P.M, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, -P.Bw / P.J, 0.0, -P.tauc / P.J, 0.0, -1.0],
            ]
        )
    if block == 3:
        bemf = P.Ke * P.N / (P.La * P.r)
        return np.array(
            [
                [0.0, 0.0, -P.Ra / P.La, 0.0, -bemf, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, -P.Ra / P.La, 0.0, -bemf, 0.0, -1.0],
            ]
        )
    raise ValueError(f"Unsupported block: {block}")


def bounded_disturbance(t: float, bound: float, phase: float) -> np.ndarray:
    """A deterministic vector signal satisfying ||d(t)||_2 <= bound."""
    raw = np.array(
        [
            np.sin(0.73 * t + phase) + 0.35 * np.sin(1.91 * t - 0.4),
            np.cos(0.61 * t - 0.7 * phase) + 0.30 * np.sin(1.37 * t + 0.2),
        ]
    )
    scale = max(1.0, np.linalg.norm(raw))
    return bound * raw / scale


def excitation(t: float) -> np.ndarray:
    uR = 3.0 * np.sin(1.1 * t) + 2.2 * np.sin(2.7 * t + 0.3) + 1.4 * np.sin(5.3 * t)
    uL = 2.7 * np.sin(0.9 * t + 1.0) + 2.0 * np.sin(3.1 * t) + 1.6 * np.sin(4.7 * t + 0.6)
    uc = np.array([uR, uL])
    return np.clip(uc, -P.Umax, P.Umax)


def synthetic_filter(t: float, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Bounded synthetic filtered signal and its derivative for data collection."""
    if block == 2:
        beta = np.array(
            [0.18 * np.sin(0.47 * t) + 0.06 * np.sin(1.31 * t),
             0.42 * np.sin(0.59 * t + 0.4) + 0.10 * np.sin(1.73 * t)]
        )
        dbeta = np.array(
            [0.18 * 0.47 * np.cos(0.47 * t) + 0.06 * 1.31 * np.cos(1.31 * t),
             0.42 * 0.59 * np.cos(0.59 * t + 0.4) + 0.10 * 1.73 * np.cos(1.73 * t)]
        )
    else:
        beta = np.array(
            [0.25 * np.sin(0.83 * t + 0.2) + 0.08 * np.sin(2.11 * t),
             0.23 * np.sin(0.71 * t - 0.5) + 0.09 * np.sin(1.89 * t + 0.7)]
        )
        dbeta = np.array(
            [0.25 * 0.83 * np.cos(0.83 * t + 0.2) + 0.08 * 2.11 * np.cos(2.11 * t),
             0.23 * 0.71 * np.cos(0.71 * t - 0.5) + 0.09 * 1.89 * np.cos(1.89 * t + 0.7)]
        )
    return beta, dbeta


def plant_rhs(V: np.ndarray, I: np.ndarray, U: np.ndarray, d2: np.ndarray, d3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return P.H2 @ I + F2(V) + d2, P.H3 @ U + F3(V, I) + d3


def rk4_plant(V: np.ndarray, I: np.ndarray, t: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Advance the data-generation plant under the bounded disturbances d_k."""
    def f(tt: float, vv: np.ndarray, ii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return plant_rhs(
            vv,
            ii,
            excitation(tt),
            bounded_disturbance(tt, DBAR[2], DATA_D2_PHASE),
            bounded_disturbance(tt, DBAR[3], DATA_D3_PHASE),
        )

    k1v, k1i = f(t, V, I)
    k2v, k2i = f(t + dt / 2.0, V + dt * k1v / 2.0, I + dt * k1i / 2.0)
    k3v, k3i = f(t + dt / 2.0, V + dt * k2v / 2.0, I + dt * k2i / 2.0)
    k4v, k4i = f(t + dt, V + dt * k3v, I + dt * k3i)
    return (
        V + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
        I + dt * (k1i + 2.0 * k2i + 2.0 * k3i + k4i) / 6.0,
    )


def snapshot_signals(t: float, V: np.ndarray, I: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """Record measured offline signals without constructing a disturbance matrix.

    The bounded disturbances act on the true plant, so their sampled values
    are embedded in the measured derivatives ``Z`` exactly as in physical
    data collection; the sampler still records only ``Z``, ``Y``, and ``X``
    and never constructs ``D``.
    """
    U = excitation(t)
    d2 = bounded_disturbance(t, DBAR[2], DATA_D2_PHASE)
    d3 = bounded_disturbance(t, DBAR[3], DATA_D3_PHASE)
    b1, db1 = synthetic_filter(t, 2)
    b2, db2 = synthetic_filter(t, 3)
    z2 = V - b1
    z3 = I - b2
    Vdot, Idot = plant_rhs(V, I, U, d2, d3)
    return {
        2: {
            "Z": Vdot - db1,
            "Y": np.concatenate([z2, sigma2(V, db1)]),
            "X": I.copy(),
        },
        3: {
            "Z": Idot - db2,
            "Y": np.concatenate([z3, sigma3(V, I, db2)]),
            "X": U,
        },
    }


def collect_snapshots(K: int = 800, dt: float = DATA_DT,
                      sample_period: float = DATA_SAMPLE_PERIOD,
                      initial_transient: float = DATA_INITIAL_TRANSIENT) -> dict[int, dict[str, np.ndarray]]:
    """Collect offline measurements under bounded process disturbances.

    The disturbances satisfy Assumption 2 and remain unknown to the
    synthesis: their sampled effect is embedded in the measured derivative
    matrix ``Z``, and the sampler records only ``Z``, ``Y``, and ``X``.
    """
    intervals = int(round(sample_period / dt))
    if intervals < 2 or not np.isclose(intervals * dt, sample_period):
        raise ValueError("sample_period must be an integer multiple of dt")
    transient_steps = int(round(initial_transient / dt))
    if not np.isclose(transient_steps * dt, initial_transient):
        raise ValueError("initial_transient must be an integer multiple of dt")

    V = np.zeros(2)
    I = np.zeros(2)
    rows = {2: {key: [] for key in ("Z", "Y", "X")},
            3: {key: [] for key in ("Z", "Y", "X")}}

    t = 0.0
    for _ in range(transient_steps):
        V, I = rk4_plant(V, I, t, dt)
        t += dt

    for sample_index in range(K):
        signals = snapshot_signals(t, V, I)
        for k in (2, 3):
            for key in ("Z", "Y", "X"):
                rows[k][key].append(signals[k][key])

        if sample_index == K - 1:
            break
        for _ in range(intervals):
            V, I = rk4_plant(V, I, t, dt)
            t += dt

    result: dict[int, dict[str, np.ndarray]] = {}
    for k in (2, 3):
        result[k] = {key: np.asarray(value).T for key, value in rows[k].items()}
        if result[k]["Z"].shape[1] != K:
            raise RuntimeError(f"Block {k}: expected {K} snapshots, obtained {result[k]['Z'].shape[1]}")
    return result


def synthesize_block(block: int, data: dict[str, np.ndarray], Lambda: np.ndarray, dbar: float,
                     epsilon: float) -> dict[str, np.ndarray | float | int]:
    Z, Y, X = data["Z"], data["Y"], data["X"]
    n, K = Z.shape
    m = Y.shape[0]
    C = np.vstack([np.eye(n), np.zeros((m - n, n))])
    Rsel = np.vstack([np.zeros((n, m - n)), np.eye(m - n)])
    S = np.vstack([np.zeros((n, m - n)), Rsel])
    G = np.vstack([Z, Y])
    gram = G @ G.T
    yy = Y @ Y.T
    ydag = Y.T @ np.linalg.inv(yy)
    projector = np.eye(K) - ydag @ Y
    P0 = ydag @ C
    chi = np.sqrt(K) * dbar
    mu = epsilon + chi**2

    # Route I: exact data-only feasibility test and its closed-form feasible point.
    E = C + Y @ Z.T
    Phi = Z @ Z.T - mu * np.eye(n) - E.T @ np.linalg.solve(yy, E)
    P_route_i = ydag @ E - Z.T
    Xi_route_i = (
        Z @ P_route_i + (Z @ P_route_i).T
        + mu * np.eye(n) + P_route_i.T @ P_route_i
    )

    # Route II: null-space feasibility test and a strict constructive selector.
    N = Z @ projector
    n_singular_values = np.linalg.svd(N, compute_uv=False)
    n_sigma_min = float(n_singular_values[-1])
    if n_sigma_min <= chi:
        raise RuntimeError(
            f"Route II infeasible: sigma_min(N)={n_sigma_min:.6g} <= chi={chi:.6g}"
        )
    Vsel = projector @ N.T @ np.linalg.inv(N @ N.T)
    numerator = (
        np.linalg.eigvalsh(Z @ P0 + (Z @ P0).T).max()
        + 2.0 * chi * np.linalg.norm(P0, 2) + epsilon
    )
    kappa_lower = max(0.0, float(numerator / (2.0 * (1.0 - chi / n_sigma_min))))
    kappa = ROUTE_II_KAPPA[block]
    if kappa <= kappa_lower:
        raise RuntimeError(
            f"Selected Route-II kappa_{block}={kappa:.6g} must exceed "
            f"kappa_min={kappa_lower:.6g}"
        )
    P_route_ii = P0 - kappa * Vsel
    route_ii_certificate = (
        Z @ P_route_ii + (Z @ P_route_ii).T
        + (2.0 * chi * np.linalg.norm(P_route_ii, 2) + epsilon) * np.eye(n)
    )

    # Route II is the selected synthesis route for all closed-loop simulations.
    # ``Lambda`` is retained in the function signature only so older diagnostic
    # callers remain compatible; it does not constrain the selected solution.
    _ = Lambda
    Psol = P_route_ii
    Qsol = G.T @ np.linalg.solve(gram, S)
    W = X @ np.hstack([Psol, Qsol])
    Xi = route_ii_certificate
    # Simulation-only truth diagnostics use the known plant coefficients.  The
    # synthesis above uses no D_k realization; in physical data D_k remains an
    # unknown matrix embedded in Z and is handled only through dbar.
    Theta = {2: P.H2, 3: P.H3}[block] @ W + known_rho(block)
    Acl = Theta[:, :n]
    M = Theta[:, n:]
    p_match = Y @ Psol - C
    q_match = G @ Qsol - S
    match_residual = float(
        np.sqrt(
            np.linalg.norm(p_match, ord="fro") ** 2
            + np.linalg.norm(q_match, ord="fro") ** 2
        )
    )
    return {
        "selected_route": "II",
        "W": W,
        "Psol": Psol,
        "Qsol": Qsol,
        "nominal_matrix": Z @ Psol,
        "Acl": Acl,
        "M": M,
        "Xi": Xi,
        "rank": int(np.linalg.matrix_rank(G, tol=1e-9)),
        "rows": int(G.shape[0]),
        "cond": float(np.linalg.cond(gram)),
        "match_residual": match_residual,
        "right_inverse_residual": float(np.linalg.norm(N @ Vsel - np.eye(n), ord="fro")),
        "xi_max": float(np.linalg.eigvalsh(0.5 * (Xi + Xi.T)).max()),
        "actual_sym_max": float(np.linalg.eigvalsh(Acl + Acl.T).max()),
        "data_equation_residual": float(
            np.linalg.norm(
                Z - {2: P.H2, 3: P.H3}[block] @ X - known_rho(block) @ Y,
                ord="fro",
            )
        ),
        "route_i_phi_min": float(np.linalg.eigvalsh(0.5 * (Phi + Phi.T)).min()),
        "route_i_closed_xi_max": float(
            np.linalg.eigvalsh(0.5 * (Xi_route_i + Xi_route_i.T)).max()
        ),
        "route_i_closed_match": float(np.linalg.norm(Y @ P_route_i - C, ord="fro")),
        "route_i_closed_p_norm": float(np.linalg.norm(P_route_i, 2)),
        "route_ii_sigma_min": n_sigma_min,
        "route_ii_spectral_margin": n_sigma_min - chi,
        "route_ii_kappa_lower": kappa_lower,
        "route_ii_kappa": kappa,
        "route_ii_Psol": P_route_ii,
        "route_ii_Qsol": Qsol,
        "route_ii_W": X @ np.hstack([P_route_ii, Qsol]),
        "route_ii_certificate_max": float(
            np.linalg.eigvalsh(0.5 * (route_ii_certificate + route_ii_certificate.T)).max()
        ),
        "route_ii_match": float(np.linalg.norm(Y @ P_route_ii - C, ord="fro")),
        "route_ii_p_norm": float(np.linalg.norm(P_route_ii, 2)),
    }


def monte_carlo_margin(Psol: np.ndarray, nominal_matrix: np.ndarray, dbar: float, K: int,
                       draws: int = 2000, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    margins = np.empty(draws)
    radius = np.sqrt(K) * dbar
    for j in range(draws):
        D = rng.standard_normal((2, K))
        D *= radius / np.linalg.norm(D, ord="fro")
        Acl = nominal_matrix - D @ Psol
        margins[j] = np.linalg.eigvalsh(Acl + Acl.T).max()
    return margins


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def reference_components(
    t: float,
    a: float = REFERENCE_A,
    b: float = REFERENCE_B,
    nu: float = REFERENCE_NU,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = a * np.sin(nu * t)
    y = b * np.sin(2.0 * nu * t)
    dx = a * nu * np.cos(nu * t)
    dy = 2.0 * b * nu * np.cos(2.0 * nu * t)
    ddx = -a * nu**2 * np.sin(nu * t)
    ddy = -4.0 * b * nu**2 * np.sin(2.0 * nu * t)
    vr = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    omega = (dx * ddy - dy * ddx) / (vr**2)
    return (
        np.array([x, y, theta]),
        np.array([vr, omega]),
        np.array([dx, dy]),
        np.array([ddx, ddy]),
    )


def reference(t: float) -> tuple[np.ndarray, np.ndarray]:
    qr, Vr, _, _ = reference_components(t)
    return qr, Vr


def reference_cartesian_velocity(
    t: float,
    a: float = REFERENCE_A,
    b: float = REFERENCE_B,
    nu: float = REFERENCE_NU,
) -> np.ndarray:
    """Analytic Cartesian feedforward velocity of the simulated reference."""
    return np.array(
        [
            a * nu * np.cos(nu * t),
            2.0 * b * nu * np.cos(2.0 * nu * t),
        ]
    )




def outer_command_from_reference(
    pose: np.ndarray,
    qr: np.ndarray,
    Vr: np.ndarray,
    kx: float = 4.0,
    ky: float = 12.0,
    kth: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    dx, dy = qr[0] - pose[0], qr[1] - pose[1]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    e = np.array([c * dx + s * dy, -s * dx + c * dy, wrap_angle(qr[2] - pose[2])])
    alpha1 = np.array(
        [
            Vr[0] * np.cos(e[2]) + kx * e[0],
            Vr[1] + ky * Vr[0] * e[1] + kth * abs(Vr[0]) * np.sin(e[2]),
        ]
    )
    return alpha1, e


def direct_capture_command_from_reference(
    pose: np.ndarray,
    qr: np.ndarray,
    qrdot: np.ndarray,
    gain: float = 4.0,
    lookahead: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ``[v_c, omega_c]`` by Cartesian capture of a look-ahead point.

    The commanded look-ahead-point velocity is ``qrdot + gain * e_l``.  This
    avoids reconstructing a reference yaw rate from the curvature quotient.
    ``capture_error`` is the error of the controlled look-ahead point, whereas
    ``pose_error`` is retained only for direct comparison with the manuscript's
    existing body-frame tracking metrics.
    """
    if not np.isfinite(lookahead) or lookahead <= 0.0:
        raise ValueError("lookahead must be finite and strictly positive")
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("direct-capture gain must be finite and strictly positive")

    c, s = np.cos(pose[2]), np.sin(pose[2])
    heading = np.array([c, s])
    normal = np.array([-s, c])
    controlled_point = pose[:2] + lookahead * heading
    capture_error = qr[:2] - controlled_point
    desired_point_velocity = qrdot + gain * capture_error
    alpha1 = np.array(
        [
            heading @ desired_point_velocity,
            (normal @ desired_point_velocity) / lookahead,
        ]
    )

    dx, dy = qr[0] - pose[0], qr[1] - pose[1]
    pose_error = np.array(
        [
            c * dx + s * dy,
            -s * dx + c * dy,
            wrap_angle(qr[2] - pose[2]),
        ]
    )
    return alpha1, pose_error, capture_error


def polar_capture_command_from_reference(
    pose: np.ndarray,
    qr: np.ndarray,
    k_rho: float = OUTER_K_RHO,
    k_alpha: float = OUTER_K_ALPHA,
    capture_speed: float = OUTER_CAPTURE_SPEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounded polar point-capture command for a moving Cartesian target.

    Only the instantaneous target position is used.  The target velocity is
    therefore an exogenous input to the position-error dynamics, as required
    by the moving-target UUB interpretation tested here.
    """
    if not np.isfinite(k_rho) or k_rho <= 0.0:
        raise ValueError("k_rho must be finite and strictly positive")
    if not np.isfinite(k_alpha) or k_alpha <= 0.0:
        raise ValueError("k_alpha must be finite and strictly positive")
    if not np.isfinite(capture_speed) or capture_speed <= 0.0:
        raise ValueError("capture_speed must be finite and strictly positive")

    position_error = qr[:2] - pose[:2]
    rho = float(np.linalg.norm(position_error))
    if rho <= POLAR_CAPTURE_EPS:
        alpha = 0.0
        radial_speed = 0.0
        curvature_term = 0.0
    else:
        target_bearing = float(np.arctan2(position_error[1], position_error[0]))
        alpha = wrap_angle(target_bearing - pose[2])
        radial_speed = float(capture_speed * np.tanh(k_rho * rho / capture_speed))
        curvature_term = radial_speed * np.sin(alpha) * np.cos(alpha) / rho

    alpha1 = np.array(
        [
            radial_speed * np.cos(alpha),
            k_alpha * alpha + curvature_term,
        ]
    )
    error = np.array([position_error[0], position_error[1], alpha])
    return alpha1, error, position_error


def continuous_guidance_command(
    pose: np.ndarray,
    t: float,
    kp: float = 0.80,
    correction_limit: float = 0.12,
    ktheta: float = 1.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Implement (guidance)--(kanayama) in the current manuscript."""
    qr, _, qrdot, qrddot = reference_components(t)
    position_error = qr[:2] - pose[:2]
    scaled_error = kp * position_error / correction_limit
    tanh_error = np.tanh(scaled_error)
    correction = correction_limit * tanh_error
    guidance = qrdot + correction
    guidance_norm_sq = float(guidance @ guidance)
    if guidance_norm_sq <= 1.0e-12:
        raise FloatingPointError(
            f"The guidance field violates the nonvanishing condition at t={t:.6f} s"
        )

    c, s = np.cos(pose[2]), np.sin(pose[2])
    heading = np.array([c, s])
    theta_guidance = float(np.arctan2(guidance[1], guidance[0]))
    heading_error = wrap_angle(theta_guidance - pose[2])
    v_command = float(heading @ guidance)
    correction_jacobian = np.diag(kp * (1.0 - tanh_error**2))
    guidance_derivative = qrddot + correction_jacobian @ (
        qrdot - v_command * heading
    )
    guidance_perp = np.array([-guidance[1], guidance[0]])
    omega_guidance = float(guidance_perp @ guidance_derivative / guidance_norm_sq)
    omega_command = omega_guidance + ktheta * heading_error
    error = np.array([position_error[0], position_error[1], heading_error])
    return np.array([v_command, omega_command]), error, qr, position_error


def outer_command(pose: np.ndarray, t: float, kx: float = 4.0,
                  ky: float = 12.0, kth: float = 5.0,
                  mode: str = "polar_capture", capture_gain: float = 4.0,
                  lookahead: float = 0.04) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qr, Vr = reference(t)
    if mode == "polar_capture":
        alpha1, e, capture_error = polar_capture_command_from_reference(pose, qr)
    elif mode == "continuous_guidance":
        alpha1, e, qr, capture_error = continuous_guidance_command(pose, t)
    elif mode == "kanayama":
        alpha1, e = outer_command_from_reference(pose, qr, Vr, kx, ky, kth)
        capture_error = qr[:2] - pose[:2]
    elif mode == "direct_capture":
        alpha1, e, capture_error = direct_capture_command_from_reference(
            pose,
            qr,
            reference_cartesian_velocity(t),
            gain=capture_gain,
            lookahead=lookahead,
        )
    else:
        raise ValueError(f"Unsupported outer-loop mode: {mode}")
    return alpha1, e, qr, capture_error


def unpack_state(state: np.ndarray) -> tuple[np.ndarray, ...]:
    pose = state[0:3]
    V = state[3:5]
    I = state[5:7]
    beta1 = state[7:9]
    beta2 = state[9:11]
    Mh2 = state[11:23].reshape(2, 6)
    Mh3 = state[23:35].reshape(2, 6)
    return pose, V, I, beta1, beta2, Mh2, Mh3


def closed_loop_rhs(t: float, state: np.ndarray, W2: np.ndarray, W3: np.ndarray,
                    adapt: bool, disturbance_scale: float, tau1: float, tau2: float,
                    r2: float, r3: float, delta2: float, delta3: float,
                    voltage_limit: float, outer_mode: str = "polar_capture",
                    capture_gain: float = 4.0,
                    lookahead: float = 0.04,
                    actuator_map: str = "hard_clip") -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pose, V, I, beta1, beta2, Mh2, Mh3 = unpack_state(state)
    alpha1, e, qr, capture_error = outer_command(
        pose,
        t,
        mode=outer_mode,
        capture_gain=capture_gain,
        lookahead=lookahead,
    )
    dbeta1 = (alpha1 - beta1) / tau1
    Z2 = V - beta1
    sig2 = sigma2(V, dbeta1)
    alpha2 = W2 @ np.concatenate([Z2, sig2]) - Mh2 @ sig2
    dbeta2 = (alpha2 - beta2) / tau2
    Z3 = I - beta2
    sig3 = sigma3(V, I, dbeta2)
    Uc = W3 @ np.concatenate([Z3, sig3]) - Mh3 @ sig3
    if actuator_map == "smooth_tanh":
        U = voltage_limit * np.tanh(Uc / voltage_limit)
    elif actuator_map == "hard_clip":
        U = np.clip(Uc, -voltage_limit, voltage_limit)
    else:
        raise ValueError(f"Unsupported actuator map: {actuator_map}")
    d2 = bounded_disturbance(t, disturbance_scale * DBAR[2], 0.9)
    d3 = bounded_disturbance(t, disturbance_scale * DBAR[3], 1.7)
    Vdot, Idot = plant_rhs(V, I, U, d2, d3)
    pose_dot = np.array([V[0] * np.cos(pose[2]), V[0] * np.sin(pose[2]), V[1]])
    if adapt:
        dMh2 = r2 * (np.outer(T2 @ Z2, sig2) - delta2 * Mh2)
        dMh3 = r3 * (np.outer(T3 @ Z3, sig3) - delta3 * Mh3)
    else:
        dMh2 = np.zeros_like(Mh2)
        dMh3 = np.zeros_like(Mh3)
    derivative = np.concatenate([pose_dot, Vdot, Idot, dbeta1, dbeta2, dMh2.ravel(), dMh3.ravel()])
    aux = {
        "pose": pose,
        "qr": qr,
        "e": e,
        "capture_error": capture_error,
        "V": V,
        "alpha1": alpha1,
        "I": I,
        "Z2": Z2,
        "Z3": Z3,
        "U": U,
        "Uc": Uc,
        "Mh": np.array([np.linalg.norm(Mh2, ord="fro"), np.linalg.norm(Mh3, ord="fro")]),
    }
    return derivative, aux


def simulate(W2: np.ndarray, W3: np.ndarray, adapt: bool, disturbance_scale: float,
             horizon: float = 80.0, dt: float = 1.0e-3,
             pose_offset: np.ndarray | None = None,
             initial_current: np.ndarray | None = None,
             voltage_limit: float | None = None,
             outer_mode: str = "polar_capture",
             capture_gain: float = 4.0,
             lookahead: float = 0.04,
             actuator_map: str = "hard_clip") -> dict[str, np.ndarray | float]:
    tau1, tau2 = 0.04, 0.015
    r2, r3 = 0.50, 0.12
    delta2, delta3 = 0.08, 0.10
    if pose_offset is None:
        # Closed-loop runs start from (1.0, 0.0) m relative to the reference
        # start pose, matching the red start marker in the tracking figure.
        pose_offset = np.array([1.0, 0.0, 0.15])
    if initial_current is None:
        initial_current = np.zeros(2)
    if voltage_limit is None:
        voltage_limit = P.Umax
    q0, _ = reference(0.0)
    pose0 = q0 + np.asarray(pose_offset, dtype=float)
    alpha10, _, _, _ = outer_command(
        pose0,
        0.0,
        mode=outer_mode,
        capture_gain=capture_gain,
        lookahead=lookahead,
    )
    V0 = np.zeros(2)
    I0 = np.asarray(initial_current, dtype=float)
    beta10 = alpha10.copy()
    z20 = V0 - beta10
    sig20 = sigma2(V0, np.zeros(2))
    alpha20 = W2 @ np.concatenate([z20, sig20])
    state = np.concatenate([pose0, V0, I0, beta10, alpha20, np.zeros(24)])

    keys = (
        "t", "pose", "qr", "e", "capture_error", "V", "alpha1",
        "I", "Z2", "Z3", "U", "Uc", "Mh",
    )
    log = {key: [] for key in keys}
    stride = 10
    nsteps = int(horizon / dt)

    def f(tt: float, xx: np.ndarray) -> np.ndarray:
        return closed_loop_rhs(
            tt, xx, W2, W3, adapt, disturbance_scale, tau1, tau2,
            r2, r3, delta2, delta3, voltage_limit,
            outer_mode, capture_gain, lookahead, actuator_map,
        )[0]

    for j in range(nsteps + 1):
        t = j * dt
        if j % stride == 0:
            _, aux = closed_loop_rhs(
                t, state, W2, W3, adapt, disturbance_scale, tau1, tau2,
                r2, r3, delta2, delta3, voltage_limit,
                outer_mode, capture_gain, lookahead, actuator_map,
            )
            log["t"].append(t)
            for key in keys[1:]:
                log[key].append(aux[key].copy())
        if j == nsteps:
            break
        k1 = f(t, state)
        k2 = f(t + dt / 2.0, state + dt * k1 / 2.0)
        k3 = f(t + dt / 2.0, state + dt * k2 / 2.0)
        k4 = f(t + dt, state + dt * k3)
        state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(f"Nonfinite closed-loop state at t={t:.3f} s")

    out: dict[str, np.ndarray | float] = {key: np.asarray(value) for key, value in log.items()}
    t = out["t"]
    tail = t >= 0.60 * horizon
    pos = np.linalg.norm(out["pose"][:, :2] - out["qr"][:, :2], axis=1)
    out["position_rmse"] = float(np.sqrt(np.mean(pos**2)))
    out["position_rmse_tail"] = float(np.sqrt(np.mean(pos[tail] ** 2)))
    capture_norm = np.linalg.norm(out["capture_error"], axis=1)
    out["capture_rmse"] = float(np.sqrt(np.mean(capture_norm**2)))
    out["capture_rmse_tail"] = float(np.sqrt(np.mean(capture_norm[tail] ** 2)))
    out["bearing_rmse_tail"] = float(np.sqrt(np.mean(out["e"][tail, 2] ** 2)))
    # Retained as a compatibility alias for earlier result-processing scripts.
    out["heading_rmse_tail"] = out["bearing_rmse_tail"]
    out["z2_rms_tail"] = float(np.sqrt(np.mean(np.sum(out["Z2"][tail] ** 2, axis=1))))
    out["z3_rms_tail"] = float(np.sqrt(np.mean(np.sum(out["Z3"][tail] ** 2, axis=1))))
    out["u_peak"] = float(np.abs(out["U"]).max())
    out["uc_peak"] = float(np.abs(out["Uc"]).max())
    out["du_peak"] = float(np.abs(out["U"] - out["Uc"]).max())
    out["sat_fraction"] = float(np.mean(np.abs(out["Uc"]) > voltage_limit))
    out["voltage_limit"] = float(voltage_limit)
    out["mhat_peak"] = float(out["Mh"].max())
    out["mhat2_peak"] = float(out["Mh"][:, 0].max())
    out["mhat3_peak"] = float(out["Mh"][:, 1].max())
    return out




def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.1,
            "savefig.bbox": "tight",
        }
    )








def scalarize(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value




