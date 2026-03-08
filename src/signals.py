"""
signal.py — Element-Level Signal Generation and Beamforming (v9)
=================================================================

Zone B received signal model (eq. 12):

    r_{k,n_x,n_z}(t) = 1 + G_T(t) exp(jψ_k(t))
                          · exp(jπ n_x ℓ_x^{(k)}(t))
                          · exp(jπ n_z ℓ_z^{(k)}(t))
                        + η_{k,n_x,n_z}(t)

Direction cosines (eqs. 14–15):

    ℓ_x^{(k)}(t) = Δx_k(t) / D_{xy}^{(k)}(t)
    ℓ_z^{(k)}(t) = z_T / D_{yz}(t)

with D_{xy} = √(Δx_k² + R²), D_{yz} = √(R² + z_T²).

The 2-D steering vector is a(ℓ_x,ℓ_z) = a_x(ℓ_x) ⊗ a_z(ℓ_z)
with element (n_x, n_z) at vector index n_x·N + n_z.
"""

from __future__ import annotations

import numpy as np
from typing import List

from .geometry import ArrayNode, Target, SystemConfig


# =====================================================================
# Steering vectors
# =====================================================================

def steering_1d(u: float, N: int) -> np.ndarray:
    """1-D steering vector a(u) = [1, e^{jπu}, …, e^{jπ(N-1)u}]^T."""
    return np.exp(1j * np.pi * np.arange(N) * u)


def steering_2d(ell_x: float, ell_z: float, N: int) -> np.ndarray:
    """2-D steering vector a(ℓ_x,ℓ_z) = a_x ⊗ a_z, length N²."""
    return np.kron(steering_1d(ell_x, N), steering_1d(ell_z, N))


# =====================================================================
# Direction cosines
# =====================================================================

def direction_cosines(dx_k: float | np.ndarray,
                      R: float | np.ndarray,
                      z_T: float):
    r"""Compute azimuth and elevation direction cosines.

    Parameters
    ----------
    dx_k : Δx_k(t)
    R    : R(t)
    z_T  : target altitude

    Returns
    -------
    ell_x, ell_z : same shape as inputs
    """
    D_xy = np.sqrt(dx_k ** 2 + R ** 2)
    D_yz = np.sqrt(R ** 2 + z_T ** 2)
    ell_x = dx_k / D_xy
    ell_z = z_T / D_yz
    return ell_x, ell_z


# =====================================================================
# Zone B signal generation (eq. 12)
# =====================================================================

def generate_zone_b_signal(node: ArrayNode, target: Target,
                           t_values: np.ndarray, R_0: float,
                           snr_db: float) -> np.ndarray:
    r"""Generate the N²×M received signal matrix at one node in Zone B.

    Parameters
    ----------
    node : ArrayNode
    target : Target
    t_values : (M,)
    R_0 : float
    snr_db : float

    Returns
    -------
    X : np.ndarray, shape (N², M), complex
    """
    N = node.N
    N2 = N * N
    M = len(t_values)

    # Target shadow area (normalised)
    A_sigma = float(np.sum(target.silhouette)) * target.pixel_size ** 2

    sigma2 = 10.0 ** (-snr_db / 10.0)

    X = np.zeros((N2, M), dtype=complex)

    for m, t in enumerate(t_values):
        Rt = R_0 - target.v_y * t
        dx_k = target.x_T(t) - node.x_centre
        ell_x, ell_z = direction_cosines(dx_k, Rt, target.z_T)

        # Scattered amplitude and phase (with overflow protection)
        if abs(Rt) < 1.0:
            Rt = max(Rt, 1.0)
        G_T = 2.0 * np.sqrt(np.pi) * A_sigma / Rt
        psi_k = np.pi * (dx_k ** 2 + target.z_T ** 2) / Rt
        # Clamp phase to prevent overflow in exp
        psi_k = np.clip(psi_k, -1e6, 1e6)

        # Build the N² element signal
        a_vec = steering_2d(ell_x, ell_z, N)
        scatter = G_T * np.exp(1j * psi_k) * a_vec
        scatter = np.nan_to_num(scatter, nan=0.0, posinf=0.0, neginf=0.0)
        signal_m = np.ones(N2, dtype=complex) + scatter

        # Add noise
        noise = np.sqrt(sigma2 / 2) * (
            np.random.randn(N2) + 1j * np.random.randn(N2)
        )
        X[:, m] = signal_m + noise

    return X


# =====================================================================
# 2-D Beamforming (eq. 20)
# =====================================================================

def beamform_2d(X: np.ndarray, ell_x: float, ell_z: float,
                N: int) -> np.ndarray:
    """Steer the N²-element array toward (ℓ_x, ℓ_z) and return scalar per snapshot.

    Parameters
    ----------
    X : (N², M)
    ell_x, ell_z : direction cosines
    N : int

    Returns
    -------
    y : (M,), complex — steered beam output
    """
    w = np.conj(steering_2d(ell_x, ell_z, N)) / (N * N)
    result = w @ X
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def beamform_broadside(X: np.ndarray, N: int) -> np.ndarray:
    """Broadside beamforming (ℓ_x = ℓ_z = 0)."""
    return beamform_2d(X, 0.0, 0.0, N)


# =====================================================================
# FSSR from element signals
# =====================================================================

def measure_fssr_from_signal(X: np.ndarray, N: int) -> np.ndarray:
    """Beamform broadside and return |ỹ_k(t)|², eq. (19)."""
    y = beamform_broadside(X, N)
    return np.abs(y) ** 2
