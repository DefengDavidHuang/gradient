"""
geometry.py — System Geometry and Coordinate Definitions (v9)
=============================================================

Coordinate system (Section 1 of v9_model_and_algorithms.tex):
    - Plane wave illuminator in the +y direction.
    - K receive nodes along the x-axis at y = R_0.
    - Each node carries an N×N URA in the x–z plane, element spacing d = λ/2.
    - Element (n_x, n_z) of node k is at (X̄_k + n_x·d, R_0, n_z·d).
    - Vector index: n_x·N + n_z  (azimuth outer, elevation inner).

Target motion (Section 1.4):
    - Centre at (x_T(t), y_T(t), z_T) with constant altitude z_T.
    - x_T(t) = x_0 + v_x·t,   v_x = v·cos(θ)
    - y_T(t) = v_y·t,          v_y = v·sin(θ)
    - R(t) = R_0 − v_y·t       (propagation distance along y)

All lengths normalised to λ ⟹ λ = 1, d = 0.5.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class ArrayNode:
    """One N×N URA receive node.

    Attributes
    ----------
    x_centre : float
        Centre x-coordinate X̄_k (in λ).
    R_0 : float
        Nominal propagation distance (y-position of receiver plane).
    N : int
        Elements per dimension (N×N total).
    d : float
        Element spacing (default λ/2 = 0.5).
    """
    x_centre: float
    R_0: float
    N: int = 16
    d: float = 0.5

    @property
    def N2(self) -> int:
        """Total number of elements."""
        return self.N * self.N

    def element_x(self, n_x: int) -> float:
        """x-coordinate of element row *n_x*."""
        return self.x_centre + n_x * self.d

    def element_z(self, n_z: int) -> float:
        """z-coordinate of element column *n_z*."""
        return n_z * self.d

    @property
    def fresnel_zone_radius(self) -> float:
        return np.sqrt(self.R_0)


@dataclass
class Target:
    """Rigid-body target with oblique heading.

    Parameters match Table 1 of the v9 document.
    """
    x_0: float           # initial x-coordinate
    v: float             # speed (λ / time-unit)
    theta: float         # heading angle (rad), 0 = pure +x
    z_T: float           # constant altitude
    silhouette: Optional[np.ndarray] = None   # M_z × M_x binary
    pixel_size: float = 10.0                  # 2δ
    pixel_centres_x: Optional[np.ndarray] = None  # X'_q
    pixel_centres_z: Optional[np.ndarray] = None  # Z'_p

    # Derived -----------------------------------------------------------
    @property
    def v_x(self) -> float:
        return self.v * np.cos(self.theta)

    @property
    def v_y(self) -> float:
        return self.v * np.sin(self.theta)

    @property
    def delta(self) -> float:
        """Pixel half-width δ."""
        return self.pixel_size / 2.0

    # Kinematics --------------------------------------------------------
    def x_T(self, t: float | np.ndarray) -> float | np.ndarray:
        """x_T(t) = x_0 + v_x·t."""
        return self.x_0 + self.v_x * t

    def y_T(self, t: float | np.ndarray) -> float | np.ndarray:
        """y_T(t) = v_y·t."""
        return self.v_y * t

    def R(self, t: float | np.ndarray, R_0: float | None = None) -> float | np.ndarray:
        """R(t) = R_0 − v_y·t."""
        if R_0 is None:
            raise ValueError("R_0 must be provided")
        return R_0 - self.v_y * t

    def delta_x_k(self, t: float | np.ndarray, X_k: float,
                  R_0: float | None = None) -> float | np.ndarray:
        """Δx_k(t) = x_T(t) − X̄_k."""
        return self.x_T(t) - X_k

    def xi_k(self, t: float | np.ndarray, X_k: float,
             R_0: float) -> float | np.ndarray:
        """Fresnel parameter ξ_k(t) = |Δx_k(t)| / √R(t)."""
        dx = self.delta_x_k(t, X_k)
        Rt = self.R(t, R_0)
        return np.abs(dx) / np.sqrt(np.maximum(Rt, 1.0))


@dataclass
class SystemConfig:
    """Complete system configuration."""
    nodes: List[ArrayNode] = field(default_factory=list)
    target: Optional[Target] = None
    R_0: float = 1e4
    delta_node: float = 50.0   # inter-node spacing
    dt: float = 1.0            # snapshot interval
    xi_threshold: float = 2.0
    snr_db: float = 20.0
    M_z: int = 20
    M_x: int = 20
    # Algorithm hyper-parameters
    gamma_profile: float = 5000.0
    tau: float = 1e-6
    N_S: int = 200
    N_D: int = 20
    N_V: int = 20

    @property
    def K(self) -> int:
        return len(self.nodes)

    @property
    def N(self) -> int:
        return self.nodes[0].N if self.nodes else 16

    @property
    def sigma2(self) -> float:
        """Noise variance σ² = 10^(−SNR_dB/10)."""
        return 10.0 ** (-self.snr_db / 10.0)


# =====================================================================
# Factory
# =====================================================================

def create_system(
    K: int = 4,
    N: int = 16,
    R_0: float = 1e4,
    delta_node: float | None = None,
    x_0: float = 2000.0,
    v: float = 1.0,
    theta: float = 0.0,
    z_T: float = 10.0,
    dt: float = 1.0,
    target_shape: str = "rectangle",
    target_width: float = 100.0,
    target_height: float = 80.0,
    M_z: int = 20,
    M_x: int = 20,
    pixel_size: float = 10.0,
    snr_db: float = 20.0,
    xi_threshold: float = 2.0,
    gamma_profile: float = 5000.0,
    tau: float = 1e-6,
    N_S: int = 200,
    N_D: int = 20,
    N_V: int = 20,
) -> SystemConfig:
    """Create a v9 system configuration.

    Parameters
    ----------
    K : int
        Number of receiver nodes.
    N : int
        Elements per array dimension (N×N total per node).
    R_0 : float
        Nominal propagation distance (λ).
    delta_node : float or None
        Inter-node spacing.  Default √R_0 / 2.
    x_0 : float
        Initial target x-coordinate.
    v : float
        Target speed.
    theta : float
        Heading angle (rad), measured from +x toward +y.
    z_T : float
        Target altitude.
    dt : float
        Snapshot interval.
    target_shape : str
        ``"rectangle"``, ``"triangle"``, or ``"circle"``.
    target_width, target_height : float
        Shadow dimensions (λ).
    M_z, M_x : int
        Pixel grid dimensions.
    pixel_size : float
        Pixel side 2δ (λ).
    snr_db : float
        Per-element SNR (dB).
    xi_threshold : float
        Zone B / Zone A boundary.
    gamma_profile, tau, N_S, N_D, N_V : float / int
        Algorithm hyper-parameters (see Table 1).

    Returns
    -------
    SystemConfig
    """
    if delta_node is None:
        delta_node = np.sqrt(R_0) / 2.0

    # --- Nodes (eq. 2) ---
    nodes = []
    for k in range(K):
        X_k = (k - (K - 1) / 2.0) * delta_node
        nodes.append(ArrayNode(x_centre=X_k, R_0=R_0, N=N))

    # --- Pixel grid ---
    half_x = M_x / 2.0
    half_z = M_z / 2.0
    centres_x = (np.arange(M_x) - half_x + 0.5) * pixel_size  # X'_q
    centres_z = (np.arange(M_z) - half_z + 0.5) * pixel_size  # Z'_p

    silhouette = _create_silhouette(
        target_shape, target_width, target_height,
        centres_x, centres_z,
    )

    target = Target(
        x_0=x_0, v=v, theta=theta, z_T=z_T,
        silhouette=silhouette,
        pixel_size=pixel_size,
        pixel_centres_x=centres_x,
        pixel_centres_z=centres_z,
    )

    return SystemConfig(
        nodes=nodes, target=target, R_0=R_0,
        delta_node=delta_node, dt=dt,
        xi_threshold=xi_threshold, snr_db=snr_db,
        M_z=M_z, M_x=M_x,
        gamma_profile=gamma_profile, tau=tau,
        N_S=N_S, N_D=N_D, N_V=N_V,
    )


def _create_silhouette(shape: str, width: float, height: float,
                       cx: np.ndarray, cz: np.ndarray) -> np.ndarray:
    """Create a binary M_z × M_x silhouette."""
    M_z, M_x = len(cz), len(cx)
    P = np.zeros((M_z, M_x))
    for p, zp in enumerate(cz):
        for q, xq in enumerate(cx):
            if shape == "rectangle":
                if abs(xq) <= width / 2 and abs(zp) <= height / 2:
                    P[p, q] = 1.0
            elif shape == "triangle":
                if abs(zp) <= height / 2:
                    frac = 1.0 - (zp + height / 2) / height
                    if abs(xq) <= width / 2 * frac:
                        P[p, q] = 1.0
            elif shape == "circle":
                r = min(width, height) / 2.0
                if xq ** 2 + zp ** 2 <= r ** 2:
                    P[p, q] = 1.0
    return P


# =====================================================================
# Utility helpers used elsewhere
# =====================================================================

def node_positions(cfg: SystemConfig) -> np.ndarray:
    """Return array of X̄_k values, shape (K,)."""
    return np.array([n.x_centre for n in cfg.nodes])
