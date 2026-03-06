"""
zone_a.py - Zone A: Gradient Descent with TV Denoising (v5)
============================================================

Implements the gradient-based method with denoising from [Draft5] and
simultaneous velocity estimation from [IETDraft4].

Algorithm 1 (Draft5): Gradient descent + TV proximal denoising
Algorithm 2 (Draft5): TV denoising via dual iterative method
Algorithm 3 (IETDraft4): Simultaneous v_x estimation wrapping Algorithm 1

The gradient of the fidelity function is computed analytically:
    ∂F_ℓ/∂P_{p,q} = 4[ε - b] · Re[(i/R) · F_q(t_ℓ) · G_p · conj(1+S)]
"""

import numpy as np
from typing import List, Tuple, Optional

from .geometry import ArrayNode, Target, SystemConfig
from .fresnel import (
    compute_fresnel_coefficients,
    compute_fssr_model_vectorised,
)


# =====================================================================
# TV Denoising (Algorithm 2 from Draft5)
# =====================================================================

def _operator_L(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Divergence operator L: R^{(M-1)xL} x R^{Mx(L-1)} -> R^{MxL}

    L(p, q)_{i,j} = p_{i,j} + q_{i,j} - p_{i-1,j} - q_{i,j-1}
    with boundary conditions p_{0,j} = q_{i,0} = 0, L(p,q)_{M,L} = 0.
    """
    M, L = p.shape[0] + 1, p.shape[1]
    result = np.zeros((M, L))

    # Interior
    for i in range(M):
        for j in range(L):
            val = 0.0
            if i < M - 1 and j < L - 1:
                val = p[i, j] + q[i, j]
                if i > 0:
                    val -= p[i - 1, j]
                if j > 0:
                    val -= q[i, j - 1]
            elif i == M - 1 and j < L - 1:
                val = q[i, j]
                if i > 0:
                    val -= p[i - 1, j]
                if j > 0:
                    val -= q[i, j - 1]
            elif i < M - 1 and j == L - 1:
                val = p[i, j]
                if i > 0:
                    val -= p[i - 1, j]
                if j > 0:
                    val -= q[i, j - 1]
            else:  # i == M-1, j == L-1
                val = 0.0
            result[i, j] = val

    return result


def _operator_L_vectorised(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Vectorised version of L operator."""
    M = p.shape[0] + 1
    L = p.shape[1]
    result = np.zeros((M, L))

    # p contribution
    result[:M-1, :] += p
    result[1:M, :] -= np.vstack([p, np.zeros((1, L))])[:M-1]
    # Fix: subtract p shifted down
    result[1:, :] -= np.vstack([p, np.zeros((1, L))])[:M-1]

    # Redo cleanly
    result = np.zeros((M, L))
    # p_{i,j} for i=0..M-2
    result[:M-1, :] += p
    # -p_{i-1,j} for i=1..M-1
    result[1:M, :] -= p

    # q_{i,j} for j=0..L-2
    result[:, :L-1] += q
    # -q_{i,j-1} for j=1..L-1
    result[:, 1:L] -= q

    # Boundary: L(p,q)_{M-1, L-1} = 0
    result[M-1, L-1] = 0.0

    return result


def _operator_LT(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gradient operator L^T: R^{MxL} -> R^{(M-1)xL} x R^{Mx(L-1)}

    p_{i,j} = m_{i,j} - m_{i+1,j},  i=0..M-2
    q_{i,j} = m_{i,j} - m_{i,j+1},  i=0..M-1, j=0..L-2
    """
    M, L = m.shape
    p = m[:M-1, :] - m[1:M, :]   # shape (M-1, L)
    q = m[:, :L-1] - m[:, 1:L]   # shape (M, L-1)
    return p, q


def _operator_HC(m: np.ndarray) -> np.ndarray:
    """Clip to [0, 1]."""
    return np.clip(m, 0.0, 1.0)


def _operator_HB(p: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project onto unit ball: (p,q) -> (r,s)

    For interior (i<M-1, j<L-1):
        norm = max(1, sqrt(p_{i,j}^2 + q_{i,j}^2))
        r = p/norm, s = q/norm
    For boundary:
        r_{i,L-1} = p_{i,L-1} / max(1, |p_{i,L-1}|)
        s_{M-1,j} = q_{M-1,j} / max(1, |q_{M-1,j}|)
    """
    M_minus_1, L = p.shape
    M = M_minus_1 + 1
    L_minus_1 = q.shape[1]

    r = np.zeros_like(p)
    s = np.zeros_like(q)

    # Interior: i=0..M-2, j=0..L-2
    for i in range(M_minus_1):
        for j in range(L_minus_1):
            norm = max(1.0, np.sqrt(p[i, j]**2 + q[i, j]**2))
            r[i, j] = p[i, j] / norm
            s[i, j] = q[i, j] / norm

    # Boundary column j=L-1 for p (i=0..M-2)
    if L > L_minus_1:
        for i in range(M_minus_1):
            r[i, L-1] = p[i, L-1] / max(1.0, abs(p[i, L-1]))

    # Boundary row i=M-1 for q (j=0..L-2)
    if M - 1 < q.shape[0]:
        pass  # q has M rows, last row is i=M-1
    for j in range(L_minus_1):
        s[M_minus_1, j] = q[M_minus_1, j] / max(1.0, abs(q[M_minus_1, j]))

    return r, s


def prox_tv(z_vec: np.ndarray, delta: float, M_z: int, M_x: int,
            N_D: int = 20) -> np.ndarray:
    """
    TV proximal operator (Algorithm 2 from Draft5).

    Performs TV-based denoising with [0,1] constraints.

    Parameters
    ----------
    z_vec : np.ndarray, shape (M_z * M_x,)
        Input pixel vector.
    delta : float
        Regularisation strength (gamma_t * tau).
    M_z, M_x : int
        Pixel grid dimensions.
    N_D : int
        Number of denoising iterations.

    Returns
    -------
    np.ndarray, shape (M_z * M_x,)
        Denoised pixel vector.
    """
    # Reshape to image matrix (column-first as in Draft5)
    m = z_vec.reshape(M_z, M_x, order='F')

    # Initialise dual variables
    p = np.zeros((M_z - 1, M_x))
    q_dual = np.zeros((M_z, M_x - 1))
    r = p.copy()
    s = q_dual.copy()
    w = 1.0

    for t in range(1, N_D + 1):
        # Compute L(r, s)
        Lrs = _operator_L_vectorised(r, s)
        # HC[m - delta * L(r, s)]
        inner = _operator_HC(m - delta * Lrs)
        # L^T(inner)
        LT_p, LT_q = _operator_LT(inner)

        # Update (p, q) = HB[(r,s) + (1/(8*delta)) * L^T(inner)]
        p_new, q_new = _operator_HB(
            r + LT_p / (8 * delta + 1e-30),
            s + LT_q / (8 * delta + 1e-30)
        )

        # Nesterov momentum
        w_new = (1 + np.sqrt(1 + 4 * w**2)) / 2
        r = p_new + (w - 1) / w_new * (p_new - p)
        s = q_new + (w - 1) / w_new * (q_new - q_dual)

        p = p_new
        q_dual = q_new
        w = w_new

    # Final output
    Lrs = _operator_L_vectorised(r, s)
    x_out = _operator_HC(m - delta * Lrs)

    return x_out.reshape(-1, order='F')


# =====================================================================
# Gradient of Fidelity Function
# =====================================================================

def compute_fidelity_gradient(
    P_flat: np.ndarray,
    all_F_q: List[np.ndarray],
    G_p: np.ndarray,
    fssr_observations: List[np.ndarray],
    R: float,
    M_z: int,
    M_x: int,
) -> np.ndarray:
    """
    Compute the gradient of the fidelity function ∇F(P).

    ∂F_{k,ℓ}/∂P_{p,q} = 4[ε - b] · Re[(i/R) · F_q^(k)(t_ℓ) · G_p · conj(1+S)]

    Parameters
    ----------
    P_flat : np.ndarray, shape (M_z * M_x,)
    all_F_q : list of np.ndarray, each shape (M_k, M_x)
    G_p : np.ndarray, shape (M_z,) — same for all nodes (constant z_T)
    fssr_observations : list of np.ndarray, each shape (M_k,)
    R : float
    M_z, M_x : int

    Returns
    -------
    np.ndarray, shape (M_z * M_x,)
    """
    P = P_flat.reshape(M_z, M_x, order='F')
    grad = np.zeros((M_z, M_x))

    K = len(all_F_q)
    total_samples = sum(len(obs) for obs in fssr_observations)

    for k in range(K):
        F_q_k = all_F_q[k]  # (M_k, M_x)
        b_k = fssr_observations[k]  # (M_k,)
        M_k = len(b_k)

        # Compute model FSSR
        GP = G_p @ P  # (M_x,)
        inner = F_q_k @ GP  # (M_k,)
        S = (1j / R) * inner  # (M_k,)
        epsilon = np.abs(1.0 + S) ** 2  # (M_k,)

        # Residual
        residual = epsilon - b_k  # (M_k,)

        # Gradient contribution: 4 * residual * Re[(i/R) * F_q * G_p * conj(1+S)]
        conj_1_plus_S = np.conj(1.0 + S)  # (M_k,)

        for m in range(M_k):
            # outer product: G_p (M_z,) x F_q_k[m] (M_x,)
            H_mat = np.outer(G_p, F_q_k[m])  # (M_z, M_x)
            grad_contrib = 4.0 * residual[m] * np.real(
                (1j / R) * H_mat * conj_1_plus_S[m]
            )
            grad += grad_contrib

    # Average over samples
    if total_samples > 0:
        grad /= total_samples

    return grad.reshape(-1, order='F')


def compute_cost(
    P_flat: np.ndarray,
    all_F_q: List[np.ndarray],
    G_p: np.ndarray,
    fssr_observations: List[np.ndarray],
    R: float,
    M_z: int,
    M_x: int,
) -> float:
    """Compute fidelity cost C(P) = Σ_{k,ℓ} (ε - b)²."""
    P = P_flat.reshape(M_z, M_x, order='F')
    cost = 0.0

    for k in range(len(all_F_q)):
        F_q_k = all_F_q[k]
        b_k = fssr_observations[k]

        GP = G_p @ P
        inner = F_q_k @ GP
        epsilon = np.abs(1.0 + (1j / R) * inner) ** 2
        cost += np.sum((epsilon - b_k) ** 2)

    return float(cost)


# =====================================================================
# Algorithm 1: Gradient Descent with TV Denoising (Draft5)
# =====================================================================

def retrieve_shadow_profile_gradient(
    fssr_observations: List[np.ndarray],
    nodes: List[ArrayNode],
    target: Target,
    x_T_values_per_node: List[np.ndarray],
    R_est: float,
    z_T_est: float,
    gamma: float = 5000.0,
    tau: float = 1e-6,
    N_S: int = 200,
    N_D: int = 20,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Retrieve shadow profile using gradient descent with TV denoising.

    Implements Algorithm 1 from [Draft5] extended to multi-node.

    Parameters
    ----------
    fssr_observations : list of np.ndarray
        FSSR observations per node, each shape (M_k,).
    nodes : list of ArrayNode
    target : Target
        Provides pixel grid.
    x_T_values_per_node : list of np.ndarray
        Target x-coordinates during Zone A for each node, shape (M_k,).
    R_est : float
        Estimated target distance.
    z_T_est : float
        Estimated z-offset.
    gamma : float
        Step size.
    tau : float
        TV regularisation tradeoff parameter.
    N_S : int
        Number of iterations.
    N_D : int
        Number of denoising iterations per step.
    verbose : bool

    Returns
    -------
    P_retrieved : np.ndarray, shape (M_z, M_x)
    final_cost : float
    """
    M_z = len(target.pixel_centres_z)
    M_x = len(target.pixel_centres_x)
    N_pixels = M_z * M_x
    K = len(nodes)

    if verbose:
        print(f"  Gradient descent retrieval:")
        print(f"    Pixels: {M_z}×{M_x}={N_pixels}, K={K}")
        print(f"    R={R_est:.1f}, z_T={z_T_est:.1f}")
        print(f"    gamma={gamma}, tau={tau}, N_S={N_S}")

    # Pre-compute Fresnel coefficients for all nodes
    all_F_q = []  # each (M_k, M_x)
    G_p = None    # (M_z,), same for all nodes

    for k, node in enumerate(nodes):
        F_q_k, G_p_k = compute_fresnel_coefficients(
            node, target, x_T_values_per_node[k],
            R_override=R_est, z_T_override=z_T_est,
        )
        all_F_q.append(F_q_k)
        if G_p is None:
            G_p = G_p_k
        # G_p should be the same for all nodes (same R, z_T)

    # Initialise
    P_hat = np.zeros(N_pixels)
    s = P_hat.copy()
    q_nesterov = 1.0

    costs = []

    for t in range(1, N_S + 1):
        gamma_t = gamma / np.sqrt(t)

        # Gradient step
        grad = compute_fidelity_gradient(
            s, all_F_q, G_p, fssr_observations, R_est, M_z, M_x
        )
        z_t = s - gamma_t * grad

        # TV denoising proximal step
        P_hat_new = prox_tv(z_t, gamma_t * tau, M_z, M_x, N_D=N_D)

        # Nesterov momentum
        q_new = (1 + np.sqrt(1 + 4 * q_nesterov**2)) / 2
        s = P_hat_new + (q_nesterov - 1) / q_new * (P_hat_new - P_hat)

        P_hat = P_hat_new
        q_nesterov = q_new

        # Track cost
        if t % 20 == 0 or t == 1:
            cost = compute_cost(
                P_hat, all_F_q, G_p, fssr_observations, R_est, M_z, M_x
            )
            costs.append(cost)
            if verbose:
                print(f"    iter {t:4d}: cost = {cost:.6e}")

    final_cost = compute_cost(
        P_hat, all_F_q, G_p, fssr_observations, R_est, M_z, M_x
    )

    P_retrieved = P_hat.reshape(M_z, M_x, order='F')

    if verbose:
        print(f"    Final cost: {final_cost:.6e}")

    return P_retrieved, final_cost


# =====================================================================
# Algorithm 3: Simultaneous v_x (and z_T) Estimation (IETDraft4)
# =====================================================================

def retrieve_with_velocity_estimation(
    fssr_observations: List[np.ndarray],
    nodes: List[ArrayNode],
    target: Target,
    t_values_per_node: List[np.ndarray],
    v_x_init: float,
    x_0_est: float,
    R_est: float,
    z_T_init: float = 0.0,
    gamma_profile: float = 5000.0,
    tau: float = 1e-6,
    N_S_inner: int = 100,
    N_V: int = 30,
    gamma_v: float = 1.0,
    dv: float = 0.01,
    estimate_z_T: bool = True,
    gamma_z: float = 0.1,
    dz: float = 0.5,
    estimate_R: bool = False,
    gamma_R: float = 10.0,
    dR: float = 10.0,
    N_D: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Simultaneously retrieve shadow profile and motion parameters.

    Wraps Algorithm 1 (profile retrieval) inside a velocity optimisation
    loop (Algorithm 1 from IETDraft4), extended with z_T and R estimation.

    Parameters
    ----------
    fssr_observations : list of np.ndarray
        FSSR per node, each shape (M_k,).
    nodes : list of ArrayNode
    target : Target
    t_values_per_node : list of np.ndarray
        Time values per node's Zone A, each shape (M_k,).
    v_x_init : float
        Initial velocity estimate (from Zone B).
    x_0_est : float
        Estimated starting x-coordinate (from Zone B, fixed).
    R_est : float
        Estimated distance (from Zone B; optionally refined).
    z_T_init : float
        Initial z_T estimate.
    gamma_profile : float
        Step size for profile retrieval.
    tau : float
        TV regularisation parameter.
    N_S_inner : int
        Inner iterations for profile retrieval.
    N_V : int
        Outer iterations for velocity estimation.
    gamma_v, dv : float
        Step size and perturbation for v_x gradient.
    estimate_z_T : bool
        Whether to also estimate z_T.
    gamma_z, dz : float
        Step size and perturbation for z_T gradient.
    estimate_R : bool
        Whether to also estimate R.
    gamma_R, dR : float
        Step size and perturbation for R gradient.
    N_D : int
        Denoising iterations.
    verbose : bool

    Returns
    -------
    dict with keys: P_retrieved, v_x, z_T, R, costs
    """
    v_x = v_x_init
    z_T = z_T_init
    R = R_est
    q_v = 1.0
    v_x_prev = v_x
    z_T_prev = z_T
    R_prev = R

    costs = []

    if verbose:
        print(f"\n  Joint parameter estimation:")
        print(f"    v_x_init={v_x_init:.4f}, z_T_init={z_T_init:.1f}, R_init={R_est:.1f}")
        print(f"    N_V={N_V}, N_S_inner={N_S_inner}")

    def _compute_x_T_values(v_x_val, x_0_val):
        """Compute x_T trajectories for current parameters."""
        return [x_0_val + v_x_val * t_k for t_k in t_values_per_node]

    def _cost_for_params(v_x_val, z_T_val, R_val, P_flat):
        """Cost function for given parameters and profile."""
        x_T_vals = _compute_x_T_values(v_x_val, x_0_est)
        all_F_q = []
        G_p = None
        for k, node in enumerate(nodes):
            F_q_k, G_p_k = compute_fresnel_coefficients(
                node, target, x_T_vals[k],
                R_override=R_val, z_T_override=z_T_val,
            )
            all_F_q.append(F_q_k)
            if G_p is None:
                G_p = G_p_k
        M_z = len(target.pixel_centres_z)
        M_x = len(target.pixel_centres_x)
        return compute_cost(P_flat, all_F_q, G_p, fssr_observations, R_val, M_z, M_x)

    P_retrieved = None

    for n in range(1, N_V + 1):
        # Step 1: Retrieve profile for current v_x, z_T, R
        x_T_values = _compute_x_T_values(v_x, x_0_est)

        P_retrieved, cost = retrieve_shadow_profile_gradient(
            fssr_observations=fssr_observations,
            nodes=nodes,
            target=target,
            x_T_values_per_node=x_T_values,
            R_est=R,
            z_T_est=z_T,
            gamma=gamma_profile,
            tau=tau,
            N_S=N_S_inner,
            N_D=N_D,
            verbose=False,
        )

        P_flat = P_retrieved.reshape(-1, order='F')
        cost_current = _cost_for_params(v_x, z_T, R, P_flat)
        costs.append(cost_current)

        # Step 2: Numerical gradients
        gamma_n = gamma_v / n

        # v_x gradient
        cost_dv = _cost_for_params(v_x + dv, z_T, R, P_flat)
        grad_v = (cost_dv - cost_current) / dv
        v_x_new = v_x - gamma_n * grad_v

        # z_T gradient (optional)
        if estimate_z_T:
            gamma_z_n = gamma_z / n
            cost_dz = _cost_for_params(v_x, z_T + dz, R, P_flat)
            grad_z = (cost_dz - cost_current) / dz
            z_T_new = z_T - gamma_z_n * grad_z
        else:
            z_T_new = z_T

        # R gradient (optional)
        if estimate_R:
            gamma_R_n = gamma_R / n
            cost_dR = _cost_for_params(v_x, z_T, R + dR, P_flat)
            grad_R = (cost_dR - cost_current) / dR
            R_new = R - gamma_R_n * grad_R
            R_new = max(R_new, 100.0)  # prevent negative R
        else:
            R_new = R

        # Step 3: Nesterov acceleration
        q_new = (1 + np.sqrt(1 + 4 * q_v**2)) / 2
        momentum = (q_v - 1) / q_new

        v_x_acc = v_x_new + momentum * (v_x_new - v_x_prev)
        z_T_acc = z_T_new + momentum * (z_T_new - z_T_prev)
        R_acc = R_new + momentum * (R_new - R_prev)
        R_acc = max(R_acc, 100.0)

        v_x_prev = v_x
        z_T_prev = z_T
        R_prev = R

        v_x = v_x_acc
        z_T = z_T_acc
        R = R_acc
        q_v = q_new

        if verbose:
            params = f"v_x={v_x:.4f}, z_T={z_T:.2f}"
            if estimate_R:
                params += f", R={R:.1f}"
            print(f"    outer iter {n:3d}: cost={cost_current:.6e}, {params}")

    # Final retrieval with converged parameters
    if verbose:
        print(f"\n  Final retrieval with converged parameters...")

    x_T_values = _compute_x_T_values(v_x, x_0_est)
    P_final, final_cost = retrieve_shadow_profile_gradient(
        fssr_observations=fssr_observations,
        nodes=nodes,
        target=target,
        x_T_values_per_node=x_T_values,
        R_est=R,
        z_T_est=z_T,
        gamma=gamma_profile,
        tau=tau,
        N_S=N_S_inner * 2,  # More iterations for final
        N_D=N_D,
        verbose=verbose,
    )

    # Hard threshold
    P_binary = (P_final >= 0.5).astype(float)

    if verbose:
        n_occupied = int(np.sum(P_binary))
        M_z = len(target.pixel_centres_z)
        M_x = len(target.pixel_centres_x)
        print(f"  Retrieved: {n_occupied} occupied pixels out of {M_z * M_x}")

    return {
        'P_continuous': P_final,
        'P_binary': P_binary,
        'v_x': float(v_x),
        'z_T': float(z_T),
        'R': float(R),
        'costs': costs,
        'final_cost': float(final_cost),
    }


def compute_retrieval_accuracy(
    P_retrieved: np.ndarray,
    P_true: np.ndarray,
) -> dict:
    """
    Compute retrieval accuracy metrics.

    Returns dict with: pixel_accuracy, true_positive_rate,
    false_positive_rate, iou, true_positives, false_positives, false_negatives.
    """
    correct = np.sum(P_retrieved == P_true)
    total = P_true.size

    tp = np.sum((P_retrieved == 1) & (P_true == 1))
    fp = np.sum((P_retrieved == 1) & (P_true == 0))
    fn = np.sum((P_retrieved == 0) & (P_true == 1))
    tn = np.sum((P_retrieved == 0) & (P_true == 0))

    pixel_accuracy = correct / total
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    iou = tp / max(tp + fp + fn, 1)

    return {
        'pixel_accuracy': pixel_accuracy,
        'true_positive_rate': tpr,
        'false_positive_rate': fpr,
        'iou': iou,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
    }
