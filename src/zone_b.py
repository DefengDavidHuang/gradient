"""
zone_b.py — Zone B Processing (v9)
====================================

Implements the Zone B parameter estimation pipeline (Section 6):

1. Phase-compensated covariance (Section 6.5)
2. Separable 2-D DOA estimation (Section 6.6)
   - Azimuth:  R_x from eq. (17), 1-D MUSIC/ESPRIT → ℓ̂_x
   - Elevation: R_z from eq. (18), 1-D MUSIC/ESPRIT → ℓ̂_z
3. Altitude estimation: ẑ_T = ℓ̂_z · R̂_0  (eq. 21)
4. STFT Doppler estimation (Section 6.8)
5. Joint parameter estimation via closed-form init + NLS (Section 6.9)

Five estimated parameters: v_x, v_y, R_0, x_0, z_T.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple

from .geometry import ArrayNode, Target, SystemConfig, node_positions
from .signals import (
    generate_zone_b_signal, steering_1d, steering_2d,
    direction_cosines, beamform_2d,
)


# =====================================================================
# Phase-compensated covariance (Section 6.5)
# =====================================================================

def compensated_covariance(X: np.ndarray, N: int,
                           beta_x: float, dt: float,
                           t_ref: float = 0.0) -> np.ndarray:
    r"""Compute phase-compensated covariance R̃_k (eq. 26).

    Parameters
    ----------
    X : (N², M) — received snapshots
    N : int     — elements per dimension
    beta_x : float — estimated azimuth DOA rate
    dt : float  — snapshot interval
    t_ref : float — reference time

    Returns
    -------
    R_tilde : (N², N²), complex
    """
    N2, M = X.shape
    X_comp = X.copy()
    for m in range(M):
        t_m = m * dt
        for nx in range(N):
            phase = -np.pi * nx * beta_x * (t_m - t_ref)
            idx_start = nx * N
            idx_end = idx_start + N
            X_comp[idx_start:idx_end, m] *= np.exp(1j * phase)

    R_tilde = (X_comp @ X_comp.conj().T) / M
    # Forward-backward averaging
    J = np.fliplr(np.eye(N2))
    R_tilde = 0.5 * (R_tilde + J @ R_tilde.conj() @ J)
    return R_tilde


# =====================================================================
# Separable 2-D DOA (Section 6.6.2)
# =====================================================================

def extract_Rx(R_tilde: np.ndarray, N: int) -> np.ndarray:
    """Extract azimuth covariance R_x from R̃, eq. (17)."""
    R_x = np.zeros((N, N), dtype=complex)
    for nx1 in range(N):
        for nx2 in range(N):
            s = 0.0
            for m in range(N):
                s += R_tilde[nx1 * N + m, nx2 * N + m]
            R_x[nx1, nx2] = s / N
    return R_x


def extract_Rz(R_tilde: np.ndarray, N: int) -> np.ndarray:
    """Extract elevation covariance R_z from R̃, eq. (18)."""
    R_z = np.zeros((N, N), dtype=complex)
    for nz1 in range(N):
        for nz2 in range(N):
            s = 0.0
            for m in range(N):
                s += R_tilde[m * N + nz1, m * N + nz2]
            R_z[nz1, nz2] = s / N
    return R_z


def _music_1d(R_mat: np.ndarray, n_sources: int = 1,
              N_grid: int = 4096,
              u_min: float | None = None) -> float:
    """1-D MUSIC on an N×N covariance matrix.  Returns the strongest DOA.

    Parameters
    ----------
    u_min : float or None
        Minimum |u| to consider (to skip DC peak).  Default: 0.5/N.
    """
    N = R_mat.shape[0]
    # Guard against NaN
    R_clean = np.nan_to_num(R_mat, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals, eigvecs = np.linalg.eigh(R_clean)
    E_n = eigvecs[:, :N - n_sources]

    u_axis = np.linspace(-1, 1, N_grid, endpoint=False)
    P = np.zeros(N_grid)
    for i, u in enumerate(u_axis):
        a = steering_1d(u, N)
        proj = a.conj() @ E_n
        P[i] = 1.0 / (np.real(proj @ proj.conj()) + 1e-30)

    if u_min is None:
        u_min = 0.5 / N
    mask = np.abs(u_axis) > u_min
    P_m = P.copy()
    P_m[~mask] = 0.0
    idx = int(np.argmax(P_m))
    return float(u_axis[idx])


def _esprit_1d(R_mat: np.ndarray, n_sources: int = 1) -> float:
    """1-D LS-ESPRIT.  Returns the strongest DOA."""
    N = R_mat.shape[0]
    eigvals, eigvecs = np.linalg.eigh(R_mat)
    E_s = eigvecs[:, N - n_sources:]
    E1 = E_s[:N - 1, :]
    E2 = E_s[1:N, :]
    Phi = np.linalg.pinv(E1) @ E2
    evals = np.linalg.eigvals(Phi)
    u_est = np.angle(evals) / np.pi
    # Pick the largest-magnitude DOA
    return float(u_est[np.argmax(np.abs(u_est))])


def estimate_2d_doa(R_tilde: np.ndarray, N: int,
                    method: str = "music") -> Tuple[float, float]:
    """Separable 2-D DOA estimation.

    Returns (ℓ̂_x, ℓ̂_z).
    """
    R_x = extract_Rx(R_tilde, N)
    R_z = extract_Rz(R_tilde, N)
    if method == "music":
        # Azimuth: exclude DC (large DOA expected)
        ell_x = _music_1d(R_x, n_sources=1, u_min=0.5 / N)
        # Elevation: very small DOA expected, use tiny u_min
        ell_z = _music_1d(R_z, n_sources=1, u_min=1e-4)
    else:
        ell_x = _esprit_1d(R_x, n_sources=1)
        ell_z = _esprit_1d(R_z, n_sources=1)
    return ell_x, ell_z


# =====================================================================
# DOA time series (sliding-window)
# =====================================================================

def doa_time_series(X: np.ndarray, N: int,
                    t_actual: np.ndarray,
                    window: int = 64,
                    method: str = "music") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate ℓ_x(t) and ℓ_z(t) over sliding windows.

    Parameters
    ----------
    X : (N², M) received snapshots
    N : int
    t_actual : (M,) actual simulation time values
    window : int — sliding window size in samples
    method : str

    Returns
    -------
    t_centres : (W,)
    ell_x_series : (W,)
    ell_z_series : (W,)
    """
    N2, M_total = X.shape
    hop = window // 2
    n_win = max(1, (M_total - window) // hop + 1)
    t_c = np.zeros(n_win)
    lx = np.zeros(n_win)
    lz = np.zeros(n_win)

    for w in range(n_win):
        s = w * hop
        e = s + window
        if e > M_total:
            break
        Xw = X[:, s:e]
        # Uncompensated covariance (with nan protection)
        Rw = (Xw @ Xw.conj().T) / (e - s)
        Rw = np.nan_to_num(Rw, nan=0.0, posinf=0.0, neginf=0.0)
        J = np.fliplr(np.eye(N2))
        Rw = 0.5 * (Rw + J @ Rw.conj() @ J)
        Rw = np.nan_to_num(Rw, nan=0.0, posinf=0.0, neginf=0.0)
        ex, ez = estimate_2d_doa(Rw, N, method=method)
        lx[w] = ex
        lz[w] = ez
        # Use actual time for the window centre
        mid_idx = min(s + window // 2, M_total - 1)
        t_c[w] = t_actual[mid_idx]

    return t_c[:n_win], lx[:n_win], lz[:n_win]


# =====================================================================
# STFT Doppler (Section 6.8)
# =====================================================================

def estimate_doppler_stft(X: np.ndarray, N: int,
                          ell_x: float, ell_z: float,
                          dt: float,
                          t_actual: np.ndarray | None = None,
                          W: int = 64, H: int | None = None,
                          N_fft: int = 2048
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """STFT-based Doppler extraction (eqs. 22–24).

    Steers the 2-D beam, removes DC, applies STFT, and picks argmax.

    Returns
    -------
    t_centres : (n_win,)
    f_D : (n_win,) — instantaneous Doppler estimates
    """
    if H is None:
        H = W // 2
    N2, M_total = X.shape

    # Steered beam output, eq. (22)
    y = beamform_2d(X, ell_x, ell_z, N)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y_tilde = y - np.mean(y)  # remove DC (direct path ≈ 1)

    n_win = max(1, (M_total - W) // H + 1)
    t_c = np.zeros(n_win)
    f_D = np.zeros(n_win)

    f_axis = np.fft.fftshift(np.fft.fftfreq(N_fft, d=dt))
    hann = np.hanning(W)

    for w_idx in range(n_win):
        s = w_idx * H
        e = s + W
        if e > M_total:
            break
        seg = y_tilde[s:e] * hann
        spec = np.fft.fftshift(np.abs(np.fft.fft(seg, n=N_fft)) ** 2)

        # Zero out DC region
        dc = N_fft // 2
        exc = max(3, N_fft // 256)
        spec[dc - exc:dc + exc] = 0.0

        idx = int(np.argmax(spec))
        f_D[w_idx] = f_axis[idx]
        # Use actual times if provided
        mid_idx = min(s + W // 2, M_total - 1)
        if t_actual is not None:
            t_c[w_idx] = t_actual[mid_idx]
        else:
            t_c[w_idx] = (s + W / 2) * dt

    return t_c[:n_win], f_D[:n_win]


# =====================================================================
# Closed-form initialisation (Section 6.9, eq. 31)
# =====================================================================

def _closed_form_init(beta_k: np.ndarray, X_bar_k: np.ndarray,
                      alpha_k: np.ndarray,
                      f_dot_avg: float,
                      beta_avg: float) -> dict:
    r"""Closed-form parameter initialisation.

    From eq. (31):
        R̂_0 ≈ |f̈_avg| / β_avg²
        v̂_y = −b̂ · R̂_0²
        x̂_0 = ᾱ · R̂_0 + X̄_avg
        v̂_x = (â − b̂·x̂_0) · R̂_0
    """
    # β_k vs X̄_k linear fit → slope b, intercept a
    # β_k = a − b·X̄_k  (eq. 29)
    coeffs = np.polyfit(X_bar_k, beta_k, 1)
    b_hat = -coeffs[0]   # slope is −b
    a_hat = coeffs[1]

    # R_0
    if abs(beta_avg) > 1e-15 and abs(f_dot_avg) > 1e-15:
        R_0_hat = abs(f_dot_avg) / (beta_avg ** 2)
    else:
        R_0_hat = 1e4  # fallback

    # v_y = −b · R_0²
    v_y_hat = -b_hat * R_0_hat ** 2

    # x_0 = ᾱ · R_0 + X̄_avg
    alpha_avg = np.mean(alpha_k)
    X_bar_avg = np.mean(X_bar_k)
    x_0_hat = alpha_avg * R_0_hat + X_bar_avg

    # v_x = (a − b·x_0)·R_0
    v_x_hat = (a_hat - b_hat * x_0_hat) * R_0_hat

    return {
        'v_x': float(v_x_hat),
        'v_y': float(v_y_hat),
        'R_0': float(R_0_hat),
        'x_0': float(x_0_hat),
        'a_hat': float(a_hat),
        'b_hat': float(b_hat),
    }


# =====================================================================
# NLS refinement (Section 6.9, eq. 32–37)
# =====================================================================

def _nls_residuals(phi, X_bar_k, beta_k_obs, f_D_obs, f_dot_obs,
                   w_beta, w_f, w_fdot, t_ref, z_T):
    """Residual vector for NLS, eq. (32)."""
    v_x, v_y, R_0, x_0 = phi
    K = len(X_bar_k)
    res = []
    for k in range(K):
        dx0 = x_0 - X_bar_k[k]
        # model β_k (eq. 33)
        beta_model = (v_x * R_0 + v_y * dx0) / R_0 ** 2
        res.append(np.sqrt(w_beta) * (beta_k_obs[k] - beta_model))

        # model f_D at t_ref (eq. 34)
        dx_t = x_0 + v_x * t_ref - X_bar_k[k]
        Rt = R_0 - v_y * t_ref
        f_D_model = v_x * dx_t / Rt + v_y * (dx_t ** 2 + z_T ** 2) / (2 * Rt ** 2)
        res.append(np.sqrt(w_f) * (f_D_obs[k] - f_D_model))

        # model ḟ_D (numerical derivative at t_ref)
        eps_t = 0.01
        dx_p = x_0 + v_x * (t_ref + eps_t) - X_bar_k[k]
        Rp = R_0 - v_y * (t_ref + eps_t)
        f_p = v_x * dx_p / Rp + v_y * (dx_p ** 2 + z_T ** 2) / (2 * Rp ** 2)
        dx_m = x_0 + v_x * (t_ref - eps_t) - X_bar_k[k]
        Rm = R_0 - v_y * (t_ref - eps_t)
        f_m = v_x * dx_m / Rm + v_y * (dx_m ** 2 + z_T ** 2) / (2 * Rm ** 2)
        fdot_model = (f_p - f_m) / (2 * eps_t)
        res.append(np.sqrt(w_fdot) * (f_dot_obs[k] - fdot_model))

    return np.array(res)


def _compute_nls_weights(phi0, X_bar_k, beta_k_obs, f_D_obs, f_dot_obs,
                         t_ref, z_T):
    """Compute NLS weights from initial residual variances (eqs. 36–38)."""
    v_x, v_y, R_0, x_0 = phi0
    K = len(X_bar_k)

    # σ_β² (eq. 36) — from β linear fit residuals
    a_hat = v_x / R_0 + v_y * x_0 / R_0 ** 2
    b_hat = v_y / R_0 ** 2
    beta_pred = np.array([a_hat - b_hat * Xk for Xk in X_bar_k])
    res_beta = beta_k_obs - beta_pred
    sig_beta2 = np.sum(res_beta ** 2) / max(K - 2, 1)

    # σ_f² (eq. 37)
    f_D_pred = np.zeros(K)
    f_dot_pred = np.zeros(K)
    eps_t = 0.01
    for k in range(K):
        dx_t = x_0 + v_x * t_ref - X_bar_k[k]
        Rt = R_0 - v_y * t_ref
        f_D_pred[k] = v_x * dx_t / Rt + v_y * (dx_t ** 2 + z_T ** 2) / (2 * Rt ** 2)
        dx_p = x_0 + v_x * (t_ref + eps_t) - X_bar_k[k]
        Rp = R_0 - v_y * (t_ref + eps_t)
        f_p = v_x * dx_p / Rp + v_y * (dx_p ** 2 + z_T ** 2) / (2 * Rp ** 2)
        dx_m = x_0 + v_x * (t_ref - eps_t) - X_bar_k[k]
        Rm = R_0 - v_y * (t_ref - eps_t)
        f_m = v_x * dx_m / Rm + v_y * (dx_m ** 2 + z_T ** 2) / (2 * Rm ** 2)
        f_dot_pred[k] = (f_p - f_m) / (2 * eps_t)

    sig_f2 = np.sum((f_D_obs - f_D_pred) ** 2) / max(K - 1, 1)
    sig_fdot2 = np.sum((f_dot_obs - f_dot_pred) ** 2) / max(K - 1, 1)

    w_beta = 1.0 / max(sig_beta2, 1e-30)
    w_f = 1.0 / max(sig_f2, 1e-30)
    w_fdot = 1.0 / max(sig_fdot2, 1e-30)

    return w_beta, w_f, w_fdot


# =====================================================================
# Full Zone B estimation pipeline
# =====================================================================

def estimate_parameters_zone_b(
    nodes: List[ArrayNode],
    signals: List[np.ndarray],
    dt: float,
    R_0_nominal: float,
    t_actual: np.ndarray | None = None,
    z_T_init: float = 0.0,
    doa_method: str = "music",
    verbose: bool = True,
) -> dict:
    """Run the complete Zone B estimation pipeline.

    Parameters
    ----------
    nodes : list of ArrayNode
    signals : list of np.ndarray, each (N², M)
    dt : float
    R_0_nominal : float
    t_actual : (M,) or None — actual simulation times for the Zone B segment
    z_T_init : float
    doa_method : str — "music" or "esprit"
    verbose : bool

    Returns
    -------
    dict with v_x, v_y, R_0, x_0, z_T, ell_z_avg, etc.
    """
    K = len(nodes)
    N = nodes[0].N
    M = signals[0].shape[1]
    X_bar = np.array([n.x_centre for n in nodes])

    # If no actual times provided, generate from 0
    if t_actual is None:
        t_actual = np.arange(M) * dt

    # ------------------------------------------------------------------
    # Pass 1: uncompensated DOA time series
    # ------------------------------------------------------------------
    all_beta = []
    all_alpha = []
    all_ell_z = []
    win = min(64, M // 4)
    win = max(16, win)

    for k in range(K):
        t_c, lx, lz = doa_time_series(
            signals[k], N, t_actual, window=win, method=doa_method)
        if len(t_c) >= 3:
            c = np.polyfit(t_c, lx, 1)
            all_beta.append(c[0])
            all_alpha.append(c[1])
        all_ell_z.extend(lz.tolist())

    beta_k = np.array(all_beta)
    alpha_k = np.array(all_alpha)
    ell_z_avg = float(np.mean(all_ell_z)) if all_ell_z else 0.0

    if verbose:
        print(f"  Pass 1 — DOA rates β_k: {np.round(beta_k, 8)}")
        print(f"  Pass 1 — mean ℓ_z: {ell_z_avg:.6f}")

    beta_avg = float(np.mean(beta_k)) if len(beta_k) > 0 else 1e-10

    # ------------------------------------------------------------------
    # Doppler via STFT
    # ------------------------------------------------------------------
    all_fD = []
    all_fdot = []
    ell_x_avg_per_node = []

    for k in range(K):
        # Use average ℓ_x for steering
        if k < len(alpha_k):
            lx_steer = alpha_k[k]
        else:
            lx_steer = 0.0
        ell_x_avg_per_node.append(lx_steer)

        t_d, fD = estimate_doppler_stft(
            signals[k], N, lx_steer, ell_z_avg, dt,
            t_actual=t_actual,
            W=min(64, M // 4),
        )
        if len(t_d) >= 3:
            c = np.polyfit(t_d, fD, 1)
            all_fdot.append(c[0])
            # Evaluate f_D at t=0 using the polynomial
            all_fD.append(float(np.polyval(c, 0.0)))
        elif len(t_d) >= 1:
            all_fdot.append(0.0)
            all_fD.append(float(fD[0]))

    f_dot_avg = float(np.mean(all_fdot)) if all_fdot else 0.0
    f_D_array = np.array(all_fD) if all_fD else np.zeros(K)
    f_dot_array = np.array(all_fdot) if all_fdot else np.zeros(K)

    if verbose:
        print(f"  Doppler rates ḟ_D: {np.round(f_dot_array, 8)}")

    # ------------------------------------------------------------------
    # Closed-form initialisation
    # ------------------------------------------------------------------
    init = _closed_form_init(beta_k, X_bar[:len(beta_k)],
                             alpha_k, f_dot_avg, beta_avg)

    if verbose:
        print(f"  Closed-form init: v_x={init['v_x']:.4f}, "
              f"v_y={init['v_y']:.6f}, R_0={init['R_0']:.1f}, "
              f"x_0={init['x_0']:.1f}")

    # z_T from elevation DOA
    z_T_est = ell_z_avg * init['R_0']

    # ------------------------------------------------------------------
    # NLS refinement (general θ)
    # ------------------------------------------------------------------
    phi0 = [init['v_x'], init['v_y'], init['R_0'], init['x_0']]
    t_ref = 0.0

    # Pad observation arrays to K if needed
    beta_obs = np.zeros(K)
    beta_obs[:len(beta_k)] = beta_k
    fD_obs = np.zeros(K)
    fD_obs[:len(f_D_array)] = f_D_array
    fdot_obs = np.zeros(K)
    fdot_obs[:len(f_dot_array)] = f_dot_array

    w_beta, w_f, w_fdot = _compute_nls_weights(
        phi0, X_bar, beta_obs, fD_obs, fdot_obs, t_ref, z_T_est)

    try:
        result = least_squares(
            _nls_residuals, phi0,
            args=(X_bar, beta_obs, fD_obs, fdot_obs,
                  w_beta, w_f, w_fdot, t_ref, z_T_est),
            method='lm', max_nfev=200,
        )
        v_x_nls, v_y_nls, R_0_nls, x_0_nls = result.x
        R_0_nls = max(R_0_nls, 100.0)
    except Exception:
        v_x_nls, v_y_nls, R_0_nls, x_0_nls = phi0

    # Update z_T with refined R_0
    z_T_nls = ell_z_avg * R_0_nls

    if verbose:
        print(f"  NLS refined: v_x={v_x_nls:.4f}, v_y={v_y_nls:.6f}, "
              f"R_0={R_0_nls:.1f}, x_0={x_0_nls:.1f}")
        print(f"  Altitude: ℓ_z={ell_z_avg:.6f} → z_T={z_T_nls:.2f}")

    return {
        'v_x': float(v_x_nls),
        'v_y': float(v_y_nls),
        'R_0': float(R_0_nls),
        'x_0': float(x_0_nls),
        'z_T': float(z_T_nls),
        'ell_z_avg': float(ell_z_avg),
        'beta_k': beta_k.tolist(),
        'alpha_k': alpha_k.tolist(),
        'f_D': f_D_array.tolist() if isinstance(f_D_array, np.ndarray) else f_D_array,
        'f_dot': f_dot_array.tolist() if isinstance(f_dot_array, np.ndarray) else f_dot_array,
    }
