"""
fresnel.py - Fresnel Diffraction Integrals and FSSR Computation (v5)
====================================================================

FSSR model for target moving in x-direction with constant z_T:

    ε_k(t) = |1 + (i/R) Σ_{p,q} F_q^(k)(t) · G_p · P_{p,q}|²

where:
    F_q^(k)(t): x-integral, TIME-VARYING (depends on Δx_k(t) = x_T(t) - X_k)
    G_p:        z-integral, CONSTANT (depends on z_T)

This is the reverse of v4 where F_q was constant and G_p was time-varying.
All coordinates normalised to wavelength λ.
"""

import warnings
import numpy as np
from scipy.special import erfi, erf
from typing import Tuple

from .geometry import ArrayNode, Target, SystemConfig


def _fresnel_diff(sqrt_factor: complex, prefactor: complex,
                  centre: float, delta: float, offset: float) -> complex:
    """
    Numerically stable computation of:
        prefactor * [erfi(sqrt_factor*(centre + delta + offset))
                   - erfi(sqrt_factor*(centre - delta + offset))]

    Uses erfi(z) = -i * erf(i*z) fallback for overflow.
    """
    a = sqrt_factor * (centre + delta + offset)
    b = sqrt_factor * (centre - delta + offset)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_a = erfi(a)
            val_b = erfi(b)
            result = prefactor * (val_a - val_b)
            if np.isfinite(result):
                return result
    except (OverflowError, FloatingPointError):
        pass

    erf_ia = erf(1j * a)
    erf_ib = erf(1j * b)
    result = prefactor * 1j * (erf_ib - erf_ia)
    if np.isfinite(result):
        return result

    return 0.0 + 0.0j


def fresnel_coeff_x(x_prime_centre: float, delta: float, x_offset: float,
                    R: float) -> complex:
    """
    Compute the x-direction Fresnel coefficient for a single pixel.

    F_q^(k)(t) at a given time when Δx_k(t) = x_offset.
    Phase argument: (x_T(t) - X_k + x') → offset = x_T(t) - X_k.
    """
    sqrt_factor = np.sqrt(1j * np.pi / R)
    prefactor = np.sqrt(R / (4j))
    return _fresnel_diff(sqrt_factor, prefactor,
                         x_prime_centre, delta, x_offset)


def fresnel_coeff_z(z_prime_centre: float, delta: float, z_T: float,
                    R: float) -> complex:
    """
    Compute the z-direction Fresnel coefficient for a single pixel.

    G_p depends on constant z_T.
    """
    sqrt_factor = np.sqrt(1j * np.pi / R)
    prefactor = np.sqrt(R / (4j))
    return _fresnel_diff(sqrt_factor, prefactor,
                         z_prime_centre, delta, z_T)


def compute_fresnel_coefficients(
    node: ArrayNode,
    target: Target,
    x_T_values: np.ndarray,
    R_override: float = None,
    v_x_override: float = None,
    x_0_override: float = None,
    z_T_override: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute all Fresnel coefficients F_q^(k)(t_m) and G_p for a given node.

    In v5: F_q is (M, M_x) time-varying, G_p is (M_z,) constant.

    Parameters
    ----------
    node : ArrayNode
    target : Target
        Provides pixel grid and z_T.
    x_T_values : np.ndarray, shape (M,)
        Target x-coordinates at each time step.
    R_override : float, optional
        Override node.R (for estimated R).
    v_x_override, x_0_override, z_T_override : float, optional
        Not used for coefficient computation directly (x_T_values
        already encodes the trajectory), but z_T_override overrides
        the z-offset.

    Returns
    -------
    F_q : np.ndarray, shape (M, M_x). Complex. Time-varying x-coefficients.
    G_p : np.ndarray, shape (M_z,). Complex. Constant z-coefficients.
    """
    delta = target.pixel_size / 2.0
    R = R_override if R_override is not None else node.R
    z_T = z_T_override if z_T_override is not None else target.z_T

    M_x = len(target.pixel_centres_x)
    M_z = len(target.pixel_centres_z)
    M = len(x_T_values)

    # G_p: constant z-coefficients (depends on z_T only)
    G_p = np.array([
        fresnel_coeff_z(zp, delta, z_T, R)
        for zp in target.pixel_centres_z
    ])

    # F_q^(k)(t_m): time-varying x-coefficients
    F_q = np.zeros((M, M_x), dtype=complex)
    for m, x_T in enumerate(x_T_values):
        x_offset = x_T - node.x_centre
        for q, xq in enumerate(target.pixel_centres_x):
            F_q[m, q] = fresnel_coeff_x(xq, delta, x_offset, R)

    # Replace overflow artefacts
    F_q = np.nan_to_num(F_q, nan=0.0, posinf=0.0, neginf=0.0)
    G_p = np.nan_to_num(G_p, nan=0.0, posinf=0.0, neginf=0.0)

    return F_q, G_p


def compute_fssr_model_vectorised(
    F_q: np.ndarray,
    G_p: np.ndarray,
    P: np.ndarray,
    R: float,
) -> np.ndarray:
    """
    Vectorised FSSR model computation for v5.

    Parameters
    ----------
    F_q : np.ndarray, shape (M, M_x) — time-varying x-coefficients
    G_p : np.ndarray, shape (M_z,) — constant z-coefficients
    P : np.ndarray, shape (M_z, M_x) — pixel values
    R : float

    Returns
    -------
    np.ndarray, shape (M,)
    """
    # GP = G_p^T @ P → shape (M_x,)
    # Projects out the z-dimension using constant G_p
    GP = G_p @ P  # (M_z,) @ (M_z, M_x) = (M_x,)

    # inner = F_q @ GP → shape (M,)
    # Each time step: sum over x-pixels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inner = F_q @ GP  # (M, M_x) @ (M_x,) = (M,)

    inner = np.nan_to_num(inner, nan=0.0, posinf=0.0, neginf=0.0)
    fssr = np.abs(1.0 + (1j / R) * inner) ** 2
    return fssr


def compute_fssr_direct(
    node: ArrayNode,
    target: Target,
    x_T_values: np.ndarray,
    R_override: float = None,
    z_T_override: float = None,
) -> np.ndarray:
    """
    Compute FSSR directly from the Fresnel integral.

    Parameters
    ----------
    node : ArrayNode
    target : Target
    x_T_values : np.ndarray, shape (M,)
        Target x-coordinates at each time step.

    Returns
    -------
    np.ndarray, shape (M,)
    """
    R = R_override if R_override is not None else node.R
    F_q, G_p = compute_fresnel_coefficients(
        node, target, x_T_values,
        R_override=R_override, z_T_override=z_T_override
    )
    return compute_fssr_model_vectorised(F_q, G_p, target.silhouette, R)


def add_noise_to_fssr(fssr: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add AWGN to FSSR observations.

    SNR is defined relative to the direct path power (= 1 in normalised units):
        SNR = 1 / sigma^2  →  sigma^2 = 10^(-SNR_dB/10)
    """
    snr_linear = 10 ** (snr_db / 10)
    sigma = np.sqrt(1.0 / snr_linear)
    noise = np.random.normal(0, sigma, size=fssr.shape)
    return fssr + noise
