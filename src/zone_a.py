"""
zone_a.py — Zone A: Gradient Descent with TV Denoising (v9)
=============================================================

Implements Algorithms 1–3 from the v9 document.

Algorithm 1  Shadow profile retrieval via gradient descent + TV prox.
Algorithm 2  TV denoising (proximal operator via dual iteration).
Algorithm 3  Joint parameter and profile estimation (5 parameters:
             v_x, v_y, z_T, R_0, x_0).

Key gradient formula (eq. 42):
    ∇_P F_{k,ℓ} = (4[ε_k − b_{k,ℓ}] / R(t_ℓ))
                   · Re[i · conj(1+S_{k,ℓ}) · G(t_ℓ) F^{(k)}(t_ℓ)^T]
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from .geometry import ArrayNode, Target, SystemConfig
from .fresnel import (
    build_F_vectors, build_G_vector, compute_fssr,
)


# =====================================================================
# TV Denoising — Algorithm 2
# =====================================================================

def _L(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Discrete divergence L(r, s) → (M_z, M_x).

    r : (M_z−1, M_x)
    s : (M_z, M_x−1)
    """
    Mz = r.shape[0] + 1
    Mx = r.shape[1]
    out = np.zeros((Mz, Mx))
    # r contribution: out[i,j] += r[i,j] − r[i−1,j]
    out[:Mz - 1, :] += r
    out[1:Mz, :] -= r
    # s contribution: out[i,j] += s[i,j] − s[i,j−1]
    out[:, :Mx - 1] += s
    out[:, 1:Mx] -= s
    # boundary fix
    out[Mz - 1, Mx - 1] = 0.0
    return out


def _LT(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete gradient L^T(m) → (p, q).

    p[i,j] = m[i,j] − m[i+1,j], shape (M_z−1, M_x)
    q[i,j] = m[i,j] − m[i,j+1], shape (M_z, M_x−1)
    """
    p = m[:-1, :] - m[1:, :]
    q = m[:, :-1] - m[:, 1:]
    return p, q


def _HC(m: np.ndarray) -> np.ndarray:
    """Clip to [0, 1]."""
    return np.clip(m, 0.0, 1.0)


def _HB(p: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project onto unit ball, element-wise over matching indices."""
    Mzm1, Mx = p.shape
    Mz = Mzm1 + 1
    Mxm1 = q.shape[1]

    r = np.zeros_like(p)
    s = np.zeros_like(q)

    # Interior: i < Mz−1, j < Mx−1
    for i in range(Mzm1):
        for j in range(Mxm1):
            norm = max(1.0, np.sqrt(p[i, j] ** 2 + q[i, j] ** 2))
            r[i, j] = p[i, j] / norm
            s[i, j] = q[i, j] / norm

    # Boundary column j = Mx−1 for p
    if Mx > Mxm1:
        for i in range(Mzm1):
            r[i, Mx - 1] = p[i, Mx - 1] / max(1.0, abs(p[i, Mx - 1]))

    # Boundary row i = Mz−1 for q
    for j in range(Mxm1):
        s[Mzm1, j] = q[Mzm1, j] / max(1.0, abs(q[Mzm1, j]))

    return r, s


def prox_tv(z: np.ndarray, lam: float, M_z: int, M_x: int,
            N_D: int = 20) -> np.ndarray:
    """TV proximal operator — Algorithm 2.

    Parameters
    ----------
    z : (M_z × M_x,) flattened (column-major)
    lam : λ_tv = γ_t · τ
    M_z, M_x : grid dimensions
    N_D : denoising iterations

    Returns
    -------
    (M_z × M_x,) flattened
    """
    Z = z.reshape(M_z, M_x, order='F')

    p_prev = np.zeros((M_z - 1, M_x))
    q_prev = np.zeros((M_z, M_x - 1))
    r = np.zeros_like(p_prev)
    s = np.zeros_like(q_prev)
    w = 1.0

    for t in range(1, N_D + 1):
        Lrs = _L(r, s)
        inner = _HC(Z - lam * Lrs)
        LTp, LTq = _LT(inner)

        p_new, q_new = _HB(
            r + LTp / (8.0 * lam + 1e-30),
            s + LTq / (8.0 * lam + 1e-30),
        )
        w_new = (1.0 + np.sqrt(1.0 + 4.0 * w ** 2)) / 2.0
        mom = (w - 1.0) / w_new
        r = p_new + mom * (p_new - p_prev)
        s = q_new + mom * (q_new - q_prev)
        p_prev = p_new
        q_prev = q_new
        w = w_new

    Lrs = _L(r, s)
    out = _HC(Z - lam * Lrs)
    return out.reshape(-1, order='F')


# =====================================================================
# Gradient and cost
# =====================================================================

def _compute_S_and_eps(F_k: np.ndarray, G: np.ndarray,
                       P: np.ndarray, R_values: np.ndarray):
    """Compute S_k(t_m) and ε_k(t_m) for one node.

    Parameters
    ----------
    F_k : (M_k, M_x)
    G   : (M_k, M_z)
    P   : (M_z, M_x)
    R_values : (M_k,)

    Returns
    -------
    S : (M_k,), complex
    eps : (M_k,), real
    """
    M_k = F_k.shape[0]
    S = np.empty(M_k, dtype=complex)
    for m in range(M_k):
        GP = G[m] @ P       # (M_x,)
        S[m] = 1j / R_values[m] * (F_k[m] @ GP)
    eps = np.abs(1.0 + S) ** 2
    return S, eps


def compute_fidelity_gradient(
    P_flat: np.ndarray,
    all_F: List[np.ndarray],
    all_G: List[np.ndarray],
    all_R: List[np.ndarray],
    fssr_obs: List[np.ndarray],
    M_z: int, M_x: int,
) -> np.ndarray:
    """Gradient ∇_P F, eq. (42), averaged over all measurements.

    Parameters
    ----------
    P_flat : (M_z·M_x,) column-major
    all_F  : list of (M_k, M_x) per node
    all_G  : list of (M_k, M_z) per node
    all_R  : list of (M_k,)  per node
    fssr_obs : list of (M_k,) per node
    """
    P = P_flat.reshape(M_z, M_x, order='F')
    grad = np.zeros((M_z, M_x))
    total = 0

    for k in range(len(all_F)):
        S, eps = _compute_S_and_eps(all_F[k], all_G[k], P, all_R[k])
        b = fssr_obs[k]
        residual = eps - b
        conj_1S = np.conj(1.0 + S)
        M_k = len(b)
        total += M_k

        for m in range(M_k):
            outer = np.outer(all_G[k][m], all_F[k][m])  # (M_z, M_x)
            grad += 4.0 * residual[m] / all_R[k][m] * np.real(
                1j * conj_1S[m] * outer
            )

    if total > 0:
        grad /= total
    return grad.reshape(-1, order='F')


def compute_cost(
    P_flat: np.ndarray,
    all_F: List[np.ndarray],
    all_G: List[np.ndarray],
    all_R: List[np.ndarray],
    fssr_obs: List[np.ndarray],
    M_z: int, M_x: int,
) -> float:
    """Cost C(P) = Σ (ε − b)²."""
    P = P_flat.reshape(M_z, M_x, order='F')
    cost = 0.0
    for k in range(len(all_F)):
        _, eps = _compute_S_and_eps(all_F[k], all_G[k], P, all_R[k])
        cost += float(np.sum((eps - fssr_obs[k]) ** 2))
    return cost


# =====================================================================
# Algorithm 1: Shadow Profile Retrieval
# =====================================================================

def retrieve_shadow_profile(
    fssr_obs: List[np.ndarray],
    nodes: List[ArrayNode],
    target: Target,
    t_per_node: List[np.ndarray],
    R_0: float, v_x: float, v_y: float, z_T: float, x_0: float,
    gamma: float = 5000.0, tau: float = 1e-6,
    N_S: int = 200, N_D: int = 20,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """Algorithm 1 — gradient descent with TV proximal.

    Returns
    -------
    P : (M_z, M_x), continuous [0, 1]
    final_cost : float
    """
    M_z = len(target.pixel_centres_z)
    M_x = len(target.pixel_centres_x)
    N_pix = M_z * M_x
    K = len(nodes)

    # Pre-compute Fresnel vectors
    all_F, all_G, all_R = [], [], []
    for k in range(K):
        F_k = build_F_vectors(nodes[k], target, t_per_node[k], R_0,
                              x_0=x_0, v_x=v_x, v_y=v_y)
        G_k = build_G_vector(target, t_per_node[k], R_0,
                             z_T=z_T, v_y=v_y)
        R_k = R_0 - v_y * t_per_node[k]
        all_F.append(F_k)
        all_G.append(G_k)
        all_R.append(R_k)

    # Initialise
    P_hat = np.zeros(N_pix)
    s = P_hat.copy()
    q_nes = 1.0

    # Adaptive step: compute initial gradient norm to scale gamma
    grad0 = compute_fidelity_gradient(
        s, all_F, all_G, all_R, fssr_obs, M_z, M_x)
    gnorm0 = np.linalg.norm(grad0)
    # Scale gamma so that first step moves pixels by ~0.1
    if gnorm0 > 1e-15:
        gamma_eff = min(gamma, 0.1 * np.sqrt(N_pix) / gnorm0)
    else:
        gamma_eff = gamma

    for t in range(1, N_S + 1):
        gamma_t = gamma_eff / np.sqrt(t)
        grad = compute_fidelity_gradient(
            s, all_F, all_G, all_R, fssr_obs, M_z, M_x)
        # Clip gradient to prevent divergence
        gnorm = np.linalg.norm(grad)
        if gnorm > 1.0:
            grad = grad / gnorm
        z_t = s - gamma_t * grad
        P_new = prox_tv(z_t, gamma_t * tau, M_z, M_x, N_D=N_D)
        q_new = (1.0 + np.sqrt(1.0 + 4.0 * q_nes ** 2)) / 2.0
        s = P_new + (q_nes - 1.0) / q_new * (P_new - P_hat)
        P_hat = P_new
        q_nes = q_new

        if verbose and (t % 50 == 0 or t == 1):
            c = compute_cost(P_hat, all_F, all_G, all_R, fssr_obs, M_z, M_x)
            print(f"      Alg1 iter {t:4d}: cost = {c:.6e}")

    final_cost = compute_cost(P_hat, all_F, all_G, all_R, fssr_obs, M_z, M_x)
    P_out = P_hat.reshape(M_z, M_x, order='F')
    return P_out, final_cost


# =====================================================================
# Algorithm 3: Joint Parameter + Profile Estimation
# =====================================================================

def joint_estimation(
    fssr_obs: List[np.ndarray],
    nodes: List[ArrayNode],
    target: Target,
    t_per_node: List[np.ndarray],
    v_x_init: float, v_y_init: float,
    z_T_init: float, R_0_init: float, x_0_init: float,
    gamma_profile: float = 5000.0, tau: float = 1e-6,
    N_S: int = 200, N_D: int = 20, N_V: int = 20,
    gamma_params: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Algorithm 3 — joint shadow profile and parameter estimation.

    5 parameters: Θ = (v_x, v_y, z_T, R_0, x_0).

    Parameters
    ----------
    gamma_params : (5,) or None
        Per-parameter step sizes [γ_vx, γ_vy, γ_zT, γ_R0, γ_x0].
        Defaults to reasonable values if None.

    Returns
    -------
    dict with P_continuous, P_binary, v_x, v_y, z_T, R_0, x_0, costs.
    """
    M_z = len(target.pixel_centres_z)
    M_x = len(target.pixel_centres_x)

    if gamma_params is None:
        # [v_x, v_y, z_T, R_0, x_0] — scale to produce meaningful updates
        # after gradient normalisation
        gamma_params = np.array([0.05, 0.01, 0.5, 50.0, 5.0])

    # Perturbation sizes for numerical gradients (relative to param scale)
    d_params = np.array([0.005, 0.001, 0.5, 10.0, 1.0])

    Theta = np.array([v_x_init, v_y_init, z_T_init, R_0_init, x_0_init])
    Theta_prev = Theta.copy()
    q_nes = 1.0
    costs = []
    best_cost = np.inf
    best_Theta = Theta.copy()
    best_P = None

    if verbose:
        print(f"  Algorithm 3 — joint estimation (N_V={N_V})")
        print(f"    Init: v_x={Theta[0]:.4f}, v_y={Theta[1]:.6f}, "
              f"z_T={Theta[2]:.2f}, R_0={Theta[3]:.1f}, x_0={Theta[4]:.1f}")

    def _cost_for(params, P_flat):
        vx, vy, zT, R0, x0 = params
        aF, aG, aR = [], [], []
        for k in range(len(nodes)):
            Fk = build_F_vectors(nodes[k], target, t_per_node[k], R0,
                                 x_0=x0, v_x=vx, v_y=vy)
            Gk = build_G_vector(target, t_per_node[k], R0, z_T=zT, v_y=vy)
            Rk = R0 - vy * t_per_node[k]
            aF.append(Fk); aG.append(Gk); aR.append(Rk)
        return compute_cost(P_flat, aF, aG, aR, fssr_obs, M_z, M_x)

    P_out = None

    for n in range(1, N_V + 1):
        vx, vy, zT, R0, x0 = Theta

        # Step 1: retrieve profile with current params
        P_cont, _ = retrieve_shadow_profile(
            fssr_obs, nodes, target, t_per_node,
            R_0=R0, v_x=vx, v_y=vy, z_T=zT, x_0=x0,
            gamma=gamma_profile, tau=tau,
            N_S=N_S, N_D=N_D, verbose=False,
        )
        P_flat = P_cont.reshape(-1, order='F')

        cost_curr = _cost_for(Theta, P_flat)
        costs.append(cost_curr)

        # Track best
        if cost_curr < best_cost:
            best_cost = cost_curr
            best_Theta = Theta.copy()
            best_P = P_cont.copy()

        # Step 2: numerical gradients for each parameter
        grad_Theta = np.zeros(5)
        for i in range(5):
            Theta_p = Theta.copy()
            Theta_p[i] += d_params[i]
            cost_p = _cost_for(Theta_p, P_flat)
            grad_Theta[i] = (cost_p - cost_curr) / d_params[i]

        # Normalise gradient to prevent divergence
        gnorm = np.linalg.norm(grad_Theta)
        if gnorm > 1e-10:
            grad_Theta = grad_Theta / gnorm

        # Step size decays as 1/n
        gamma_n = gamma_params / n
        Theta_tilde = Theta - gamma_n * grad_Theta

        # Enforce R_0 ≥ R_min
        Theta_tilde[3] = max(Theta_tilde[3], 100.0)

        # Nesterov momentum
        q_new = (1.0 + np.sqrt(1.0 + 4.0 * q_nes ** 2)) / 2.0
        mom = (q_nes - 1.0) / q_new
        Theta_new = Theta_tilde + mom * (Theta_tilde - Theta_prev)
        Theta_new[3] = max(Theta_new[3], 100.0)

        Theta_prev = Theta.copy()
        Theta = Theta_new
        q_nes = q_new
        P_out = P_cont

        if verbose:
            print(f"    iter {n:3d}: cost={cost_curr:.6e}, "
                  f"v_x={Theta[0]:.4f}, v_y={Theta[1]:.6f}, "
                  f"z_T={Theta[2]:.2f}, R_0={Theta[3]:.1f}, x_0={Theta[4]:.1f}")

    # Final retrieval with best parameters found
    Theta = best_Theta
    vx, vy, zT, R0, x0 = Theta
    P_final, final_cost = retrieve_shadow_profile(
        fssr_obs, nodes, target, t_per_node,
        R_0=R0, v_x=vx, v_y=vy, z_T=zT, x_0=x0,
        gamma=gamma_profile, tau=tau,
        N_S=N_S * 2, N_D=N_D, verbose=verbose,
    )

    P_binary = (P_final >= 0.5).astype(float)

    return {
        'P_continuous': P_final,
        'P_binary': P_binary,
        'v_x': float(vx),
        'v_y': float(vy),
        'z_T': float(zT),
        'R_0': float(R0),
        'x_0': float(x0),
        'costs': costs,
        'final_cost': float(final_cost),
    }


# =====================================================================
# Accuracy metrics
# =====================================================================

def compute_retrieval_accuracy(P_est: np.ndarray,
                               P_true: np.ndarray) -> dict:
    """Compute pixel accuracy, IoU, TPR, FPR."""
    tp = np.sum((P_est == 1) & (P_true == 1))
    fp = np.sum((P_est == 1) & (P_true == 0))
    fn = np.sum((P_est == 0) & (P_true == 1))
    tn = np.sum((P_est == 0) & (P_true == 0))
    total = P_true.size

    return {
        'pixel_accuracy': float((tp + tn) / total),
        'iou': float(tp / max(tp + fp + fn, 1)),
        'true_positive_rate': float(tp / max(tp + fn, 1)),
        'false_positive_rate': float(fp / max(fp + tn, 1)),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
    }
