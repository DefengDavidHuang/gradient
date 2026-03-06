"""
doa.py - High-Resolution DOA Estimation: MUSIC and ESPRIT (v5)
================================================================

For a ULA with N elements, d=lambda/2, two signals (direct path at u=0
and target at u=ell_x), MUSIC and ESPRIT provide super-resolution
estimation of ell_x.

In v5, the target moves in x so DOA is large initially and decreases
to zero as target crosses the baseline. This means DOA estimation is
most effective early in Zone B.
"""

import warnings
import numpy as np
from typing import Tuple


def _spatial_covariance(X: np.ndarray, fb_avg: bool = True) -> np.ndarray:
    """Estimate the spatial covariance matrix with optional FB averaging."""
    N, M = X.shape

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if M > 2000:
            R = np.zeros((N, N), dtype=complex)
            chunk = 1000
            for i in range(0, M, chunk):
                Xi = X[:, i:i+chunk]
                R += Xi @ Xi.conj().T
            R /= M
        else:
            R = (X @ X.conj().T) / M

        if fb_avg:
            J = np.fliplr(np.eye(N))
            R = 0.5 * (R + J @ R.conj() @ J)

    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    return R


def _noise_subspace(R: np.ndarray, n_sources: int) -> np.ndarray:
    """Extract noise subspace eigenvectors."""
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    E_n = eigenvectors[:, :R.shape[0] - n_sources]
    return E_n


def music_spectrum(X: np.ndarray, n_sources: int = 2,
                   N_grid: int = 8192,
                   u_range: Tuple[float, float] = (-1.0, 1.0),
                   fb_avg: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the MUSIC pseudo-spectrum."""
    N = X.shape[0]
    R = _spatial_covariance(X, fb_avg=fb_avg)
    E_n = _noise_subspace(R, n_sources)

    u_axis = np.linspace(u_range[0], u_range[1], N_grid)
    P_music = np.zeros(N_grid)
    n_idx = np.arange(N)

    for i, u in enumerate(u_axis):
        a = np.exp(1j * np.pi * n_idx * u)
        proj = a.conj() @ E_n
        denom = np.real(proj @ proj.conj())
        P_music[i] = 1.0 / (denom + 1e-30)

    return u_axis, P_music


def music_estimate(X: np.ndarray, n_sources: int = 2,
                   N_grid: int = 8192,
                   u_min: float = 0.02,
                   fb_avg: bool = True) -> float:
    """Estimate the target's x-direction cosine using MUSIC."""
    u_axis, P_music = music_spectrum(X, n_sources, N_grid, fb_avg=fb_avg)

    mask = np.abs(u_axis) > u_min
    P_masked = P_music.copy()
    P_masked[~mask] = 0

    idx_peak = np.argmax(P_masked)

    if 0 < idx_peak < len(u_axis) - 1:
        alpha = np.log(P_masked[idx_peak - 1] + 1e-30)
        beta = np.log(P_masked[idx_peak] + 1e-30)
        gamma = np.log(P_masked[idx_peak + 1] + 1e-30)
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-20:
            delta = 0.5 * (alpha - gamma) / denom
            delta = np.clip(delta, -0.5, 0.5)
            du = u_axis[1] - u_axis[0]
            return float(u_axis[idx_peak] + delta * du)

    return float(u_axis[idx_peak])


def esprit_estimate(X: np.ndarray, n_sources: int = 2,
                    fb_avg: bool = True) -> np.ndarray:
    """Estimate direction cosines using LS-ESPRIT."""
    N = X.shape[0]
    R_mat = _spatial_covariance(X, fb_avg=fb_avg)

    if np.any(~np.isfinite(R_mat)):
        M = X.shape[1]
        chunk = min(1000, M)
        R_mat = np.zeros((N, N), dtype=complex)
        for i in range(0, M, chunk):
            Xi = X[:, i:i+chunk]
            if np.all(np.isfinite(Xi)):
                R_mat += Xi @ Xi.conj().T
        R_mat /= M
        if fb_avg:
            J = np.fliplr(np.eye(N))
            R_mat = 0.5 * (R_mat + J @ R_mat.conj() @ J)

    eigenvalues, eigenvectors = np.linalg.eigh(R_mat)
    E_s = eigenvectors[:, N - n_sources:]

    E1 = E_s[:N - 1, :]
    E2 = E_s[1:N, :]

    Phi = np.linalg.pinv(E1) @ E2
    eigenvalues_phi = np.linalg.eigvals(Phi)

    u_estimates = np.angle(eigenvalues_phi) / np.pi
    idx_sort = np.argsort(np.abs(u_estimates))
    return u_estimates[idx_sort]


def estimate_xT_music(
    nodes: list,
    signals: list,
    R: float,
    n_sources: int = 2,
    N_grid: int = 16384,
) -> float:
    """Estimate x_T using MUSIC across multiple nodes."""
    xT_estimates = []

    for k, (node, X) in enumerate(zip(nodes, signals)):
        u_min = 0.5 / node.N
        ell_x = music_estimate(X, n_sources=n_sources, N_grid=N_grid,
                               u_min=u_min)
        if abs(ell_x) < 1.0:
            dx = R * ell_x / np.sqrt(1.0 - ell_x**2)
        else:
            dx = R * ell_x
        xT_k = node.x_centre + dx  # v5: x_T = X_k + dx (target offset from node)
        xT_estimates.append(xT_k)

    return float(np.mean(xT_estimates))


def estimate_xT_esprit(
    nodes: list,
    signals: list,
    R: float,
    n_sources: int = 2,
) -> float:
    """Estimate x_T using ESPRIT across multiple nodes."""
    xT_estimates = []

    for k, (node, X) in enumerate(zip(nodes, signals)):
        u_est = esprit_estimate(X, n_sources=n_sources)
        ell_x = u_est[np.argmax(np.abs(u_est))]
        if abs(ell_x) < 1.0:
            dx = R * ell_x / np.sqrt(1.0 - ell_x**2)
        else:
            dx = R * ell_x
        xT_k = node.x_centre + dx
        xT_estimates.append(xT_k)

    return float(np.mean(xT_estimates))
