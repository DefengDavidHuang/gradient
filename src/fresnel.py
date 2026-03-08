"""
fresnel.py — Fresnel Diffraction Integrals and FSSR Computation (v9)
=====================================================================

Implements the separable FSSR model from Section 3 of the v9 document.

Key equations
-------------
- Eq. (6):  F_q^{(k)}(t)  — x-direction Fresnel coefficient
- Eq. (7):  G_p(t)        — z-direction Fresnel coefficient
- Eq. (9):  S_k(t) = (i/R(t)) G(t)^T P F^{(k)}(t)
- Eq. (10): ε_k(t) = |1 + S_k(t)|²
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.special import erfi, erf
from typing import Tuple, List

from .geometry import ArrayNode, Target, SystemConfig


# =====================================================================
# Low-level Fresnel helpers
# =====================================================================

def _erfi_safe(z: complex | np.ndarray) -> complex | np.ndarray:
    """Compute erfi(z) with overflow fallback via erf(iz)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val = erfi(z)
    if np.all(np.isfinite(val)):
        return val
    # Fallback: erfi(z) = -i·erf(i·z)
    return -1j * erf(1j * z)


def _fresnel_coeff(centre: float, delta: float, offset: float,
                   R: float) -> complex:
    r"""Compute one Fresnel coefficient (eq. 6 or 7).

    .. math::

        \sqrt{R/(4i)}\,\bigl[\mathrm{erfi}(\sqrt{i\pi/R}\,(c+\delta+o))
        - \mathrm{erfi}(\sqrt{i\pi/R}\,(c-\delta+o))\bigr]

    Parameters
    ----------
    centre : float
        Pixel centre (X'_q or Z'_p).
    delta : float
        Pixel half-width δ.
    offset : float
        Δx_k(t) (for x) or z_T (for z).
    R : float
        Propagation distance R(t).
    """
    sqrt_fac = np.sqrt(1j * np.pi / R)
    prefactor = np.sqrt(R / 4j)
    a = sqrt_fac * (centre + delta + offset)
    b = sqrt_fac * (centre - delta + offset)
    result = prefactor * (_erfi_safe(a) - _erfi_safe(b))
    return complex(np.nan_to_num(result, nan=0.0))


def fresnel_coeff_x(xp: float, delta: float, dx_k: float,
                    R: float) -> complex:
    """x-direction coefficient F_q^{(k)}(t), eq. (6)."""
    return _fresnel_coeff(xp, delta, dx_k, R)


def fresnel_coeff_z(zp: float, delta: float, z_T: float,
                    R: float) -> complex:
    """z-direction coefficient G_p(t), eq. (7)."""
    return _fresnel_coeff(zp, delta, z_T, R)


# =====================================================================
# Vectorised coefficient builders
# =====================================================================

def build_F_vectors(node: ArrayNode, target: Target,
                    t_values: np.ndarray, R_0: float,
                    x_0: float | None = None,
                    v_x: float | None = None,
                    v_y: float | None = None) -> np.ndarray:
    r"""Build x-direction Fresnel vectors for every time step.

    Returns F_q^{(k)}(t_m) as shape ``(M, M_x)``.

    Parameters
    ----------
    node : ArrayNode
    target : Target
        Provides pixel grid (pixel_centres_x, pixel_size).
    t_values : np.ndarray, shape (M,)
    R_0 : float
    x_0, v_x, v_y : float or None
        Override target kinematics if given.
    """
    _x0 = x_0 if x_0 is not None else target.x_0
    _vx = v_x if v_x is not None else target.v_x
    _vy = v_y if v_y is not None else target.v_y
    delta = target.delta
    cx = target.pixel_centres_x
    M = len(t_values)
    M_x = len(cx)
    F = np.zeros((M, M_x), dtype=complex)
    for m, t in enumerate(t_values):
        Rt = R_0 - _vy * t
        dx_k = _x0 + _vx * t - node.x_centre
        for q in range(M_x):
            F[m, q] = fresnel_coeff_x(cx[q], delta, dx_k, Rt)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


def build_G_vector(target: Target, t_values: np.ndarray, R_0: float,
                   z_T: float | None = None,
                   v_y: float | None = None) -> np.ndarray:
    r"""Build z-direction Fresnel vectors for every time step.

    Returns G_p(t_m) as shape ``(M, M_z)``.

    Under the v9 simplification R(t) varies with time so G_p is also
    time-varying (it depends on R(t)).  When v_y ≈ 0 the variation is
    negligible but we compute it correctly regardless.
    """
    _vy = v_y if v_y is not None else target.v_y
    _zT = z_T if z_T is not None else target.z_T
    delta = target.delta
    cz = target.pixel_centres_z
    M = len(t_values)
    M_z = len(cz)
    G = np.zeros((M, M_z), dtype=complex)
    for m, t in enumerate(t_values):
        Rt = R_0 - _vy * t
        for p in range(M_z):
            G[m, p] = fresnel_coeff_z(cz[p], delta, _zT, Rt)
    G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    return G


# =====================================================================
# FSSR computation
# =====================================================================

def compute_fssr(F: np.ndarray, G: np.ndarray,
                 P: np.ndarray, R_values: np.ndarray) -> np.ndarray:
    r"""Vectorised FSSR, eqs. (9)–(10).

    Parameters
    ----------
    F : (M, M_x)  — F_q^{(k)}(t_m)
    G : (M, M_z)  — G_p(t_m)
    P : (M_z, M_x)
    R_values : (M,)  — R(t_m)

    Returns
    -------
    epsilon : (M,)
    """
    M = F.shape[0]
    eps = np.empty(M)
    for m in range(M):
        GP = G[m] @ P          # (M_x,)
        inner = F[m] @ GP      # scalar
        S = 1j / R_values[m] * inner
        eps[m] = abs(1.0 + S) ** 2
    return eps


def compute_fssr_for_node(node: ArrayNode, target: Target,
                          t_values: np.ndarray, R_0: float,
                          x_0: float | None = None,
                          v_x: float | None = None,
                          v_y: float | None = None,
                          z_T: float | None = None) -> np.ndarray:
    """Convenience: compute FSSR time series for one node."""
    _vy = v_y if v_y is not None else target.v_y
    Rv = R_0 - _vy * t_values
    F = build_F_vectors(node, target, t_values, R_0,
                        x_0=x_0, v_x=v_x, v_y=v_y)
    G = build_G_vector(target, t_values, R_0, z_T=z_T, v_y=v_y)
    return compute_fssr(F, G, target.silhouette, Rv)


# =====================================================================
# Noise
# =====================================================================

def add_fssr_noise(fssr: np.ndarray, snr_db: float) -> np.ndarray:
    """Add AWGN to measured FSSR (after beamforming)."""
    sigma2 = 10.0 ** (-snr_db / 10.0)
    # Beamformed noise variance is σ²/N² but we measure |1+S+η|²;
    # for simplicity add Gaussian noise to the power measurement.
    noise = np.random.normal(0, np.sqrt(sigma2), size=fssr.shape)
    return fssr + noise
