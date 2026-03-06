"""
zone_b.py - Zone B Processing: v_x, R, x_0 Estimation (v5)
============================================================

Target moves in x-direction: x_T(t) = x_0 + v_x * t, constant z_T.
All distances normalised to lambda (lambda = 1).

DOA: ℓ_x^(k)(t) ≈ (x_T(t) - X_k) / R (far field)
Doppler: f_D(t) ≈ -v_x * (x_T(t) - X_k) / R

From linear fits:
    DOA rate β = v_x / R
    Doppler rate f_dot = -v_x² / R
    → v_x = -f_dot / β,  R = f_dot / β²
"""

import numpy as np
from typing import Tuple, List

from .geometry import ArrayNode, Target, SystemConfig


def generate_received_signal_zone_b(
    node: ArrayNode,
    target: Target,
    t_values: np.ndarray,
    snr_db: float = 20.0,
    A_D: float = 1.0,
) -> np.ndarray:
    """
    Generate N x M received signal matrix at one node in Zone B.

    Target moves in x-direction with constant z_T.
    """
    N = node.N
    M = len(t_values)
    R = node.R

    n_idx = np.arange(N)

    # Target area (normalised units²)
    A_sigma = float(np.sum(target.silhouette)) * target.pixel_size ** 2
    G_T_amp = 2 * np.sqrt(np.pi) * A_sigma / R

    X = np.zeros((N, M), dtype=complex)

    for m, t in enumerate(t_values):
        x_T = target.x_T(t)
        dx = x_T - node.x_centre   # Δx_k(t)
        z_T = target.z_T
        D_k = np.sqrt(dx**2 + R**2 + z_T**2)

        # x-direction cosine
        ell_x = dx / D_k
        a_T = np.exp(1j * np.pi * n_idx * ell_x)

        # Propagation phase
        psi_k = np.pi * (dx**2 + z_T**2) / R

        X[:, m] = A_D * np.ones(N) + G_T_amp * np.exp(1j * psi_k) * a_T

    # Complex AWGN referenced to direct path power
    snr_linear = 10 ** (snr_db / 10)
    sigma2 = 1.0 / snr_linear
    noise = np.sqrt(sigma2 / 2) * (
        np.random.randn(N, M) + 1j * np.random.randn(N, M)
    )
    X += noise

    return X


def _parabolic_interpolation(spectrum: np.ndarray, peak_idx: int,
                              axis_values: np.ndarray) -> float:
    """Sub-bin peak estimation via parabolic interpolation."""
    N = len(spectrum)
    if peak_idx <= 0 or peak_idx >= N - 1:
        return axis_values[peak_idx]

    alpha = np.log(spectrum[peak_idx - 1] + 1e-30)
    beta = np.log(spectrum[peak_idx] + 1e-30)
    gamma = np.log(spectrum[peak_idx + 1] + 1e-30)

    delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma + 1e-30)
    delta = np.clip(delta, -0.5, 0.5)

    bin_spacing = axis_values[1] - axis_values[0] if len(axis_values) > 1 else 1.0
    return axis_values[peak_idx] + delta * bin_spacing


def estimate_ell_x_timeseries(
    X: np.ndarray,
    N_angle: int = 8192,
    u_min: float = 0.125,
    window_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate time-varying DOA ℓ_x(t) from sliding-window spatial DFT.

    Returns time centres and ℓ_x estimates.
    """
    N, M_total = X.shape
    hop = window_size // 2
    n_windows = max(1, (M_total - window_size) // hop + 1)

    ell_x = np.zeros(n_windows)
    t_centres = np.zeros(n_windows)
    u_axis = np.linspace(-1, 1, N_angle, endpoint=False)

    for w_idx in range(n_windows):
        start = w_idx * hop
        end = start + window_size
        if end > M_total:
            break

        X_win = X[:, start:end]
        # Remove direct path
        X_clean = X_win - np.mean(X_win, axis=0, keepdims=True)

        S = np.fft.fftshift(np.fft.fft(X_clean, n=N_angle, axis=0), axes=0)
        P = np.mean(np.abs(S) ** 2, axis=1)

        mask = np.abs(u_axis) > u_min
        P_masked = P.copy()
        P_masked[~mask] = 0

        idx_peak = np.argmax(P_masked)
        ell_x[w_idx] = _parabolic_interpolation(P_masked, idx_peak, u_axis)
        t_centres[w_idx] = (start + window_size / 2)

    return t_centres[:n_windows], ell_x[:n_windows]


def estimate_ell_x_single(X: np.ndarray, N_angle: int = 8192,
                           u_min: float = 0.125) -> float:
    """Estimate single (averaged) DOA from all snapshots."""
    X_clean = X - np.mean(X, axis=0, keepdims=True)
    S = np.fft.fftshift(np.fft.fft(X_clean, n=N_angle, axis=0), axes=0)
    P = np.mean(np.abs(S) ** 2, axis=1)

    u_axis = np.linspace(-1, 1, N_angle, endpoint=False)
    mask = np.abs(u_axis) > u_min
    P_masked = P.copy()
    P_masked[~mask] = 0

    idx_peak = np.argmax(P_masked)
    return _parabolic_interpolation(P_masked, idx_peak, u_axis)


def estimate_doppler_from_steered_beam(
    X: np.ndarray,
    ell_x: float,
    dt: float,
    window_size: int = 256,
    N_fft: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate instantaneous Doppler by steering a beam toward the target.

    In v5, f_D(t) = -v_x * (x_T(t) - X_k) / R, which is linear in t.
    """
    N, M_total = X.shape

    n_idx = np.arange(N)
    w = np.exp(-1j * np.pi * n_idx * ell_x) / N
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        y = w @ X
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    y_ac = y - np.mean(y)

    hop = window_size // 2
    n_windows = max(1, (M_total - window_size) // hop + 1)
    t_centres = np.zeros(n_windows)
    f_D = np.zeros(n_windows)

    f_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, d=dt))
    window = np.hanning(window_size)

    for w_idx in range(n_windows):
        start = w_idx * hop
        end = start + window_size
        if end > M_total:
            break

        segment = y_ac[start:end] * window
        spec = np.fft.fftshift(np.abs(np.fft.fft(segment, n=N_fft)) ** 2)

        dc_idx = N_fft // 2
        exclude = max(3, N_fft // 256)
        spec[dc_idx - exclude:dc_idx + exclude] = 0

        idx_peak = np.argmax(spec)
        f_D[w_idx] = _parabolic_interpolation(spec, idx_peak, f_axis)
        t_centres[w_idx] = (start + window_size / 2) * dt

    return t_centres[:n_windows], f_D[:n_windows]


def estimate_parameters_zone_b(
    nodes: List[ArrayNode],
    signals: List[np.ndarray],
    dt: float,
    R_nominal: float,
    verbose: bool = True,
) -> dict:
    """
    Estimate v_x, R, and x_0 from Zone B DOA + Doppler analysis.

    Uses the relationships:
        DOA rate β = v_x / R
        Doppler rate f_dot = -v_x² / R
        → v_x = -f_dot / β
        → R = f_dot / β²

    Parameters
    ----------
    nodes : list of ArrayNode
    signals : list of np.ndarray (N, M)
    dt : float
        Time step between snapshots.
    R_nominal : float
        Nominal R for initial processing.
    verbose : bool

    Returns
    -------
    dict with keys: v_x, R, x_0, doa_rates, doppler_rates
    """
    K = len(nodes)
    doa_rates = []
    doa_intercepts = []
    doppler_rates = []
    doppler_intercepts = []

    for k, (node, X) in enumerate(zip(nodes, signals)):
        u_min = 2.0 / node.N

        # DOA time series
        win_size = min(128, X.shape[1] // 4)
        t_doa, ell_x = estimate_ell_x_timeseries(
            X, u_min=u_min, window_size=max(16, win_size)
        )
        t_doa_sec = t_doa * dt

        if len(t_doa_sec) >= 3:
            coeffs_doa = np.polyfit(t_doa_sec, ell_x, 1)
            doa_rates.append(coeffs_doa[0])       # β = v_x / R
            doa_intercepts.append(coeffs_doa[1])   # α = (x_0 - X_k) / R

        # Doppler time series
        ell_x_avg = estimate_ell_x_single(X, u_min=u_min)
        t_dop, f_dop = estimate_doppler_from_steered_beam(
            X, ell_x_avg, dt,
            window_size=min(256, X.shape[1] // 4)
        )

        if len(t_dop) >= 3:
            coeffs_dop = np.polyfit(t_dop, f_dop, 1)
            doppler_rates.append(coeffs_dop[0])       # f_dot = -v_x² / R
            doppler_intercepts.append(coeffs_dop[1])

    # Combine estimates
    beta_avg = np.mean(doa_rates) if doa_rates else 1e-10
    f_dot_avg = np.mean(doppler_rates) if doppler_rates else 0.0

    if abs(beta_avg) > 1e-15 and abs(f_dot_avg) > 1e-15:
        # v_x = -f_dot / β, R = f_dot / β²
        v_x_est = -f_dot_avg / beta_avg
        R_est = abs(f_dot_avg) / (beta_avg ** 2)
    else:
        # Fallback: use DOA rate only with nominal R
        v_x_est = beta_avg * R_nominal
        R_est = R_nominal

    # x_0 from DOA intercepts: α = (x_0 - X_k) / R → x_0 = α * R + X_k
    x_0_estimates = []
    for k, alpha in enumerate(doa_intercepts):
        x_0_k = alpha * R_est + nodes[k].x_centre
        x_0_estimates.append(x_0_k)
    x_0_est = np.mean(x_0_estimates) if x_0_estimates else 0.0

    if verbose:
        print(f"  DOA rates (β = v_x/R): {doa_rates}")
        print(f"  Doppler rates (f_dot = -v_x²/R): {doppler_rates}")
        print(f"  v_x estimate: {v_x_est:.4f}")
        print(f"  R estimate: {R_est:.1f}")
        print(f"  x_0 estimate: {x_0_est:.1f}")

    return {
        'v_x': float(v_x_est),
        'R': float(R_est),
        'x_0': float(x_0_est),
        'doa_rates': doa_rates,
        'doppler_rates': doppler_rates,
    }


def beamform(X: np.ndarray, beta: float = 0.0) -> np.ndarray:
    """Matched beamformer steered toward angle beta."""
    N = X.shape[0]
    n_idx = np.arange(N)
    w = np.exp(-1j * np.pi * n_idx * np.sin(beta)) / N
    return w @ X
