"""
geometry.py - System Geometry and Coordinate Definitions (v5)
=============================================================

Coordinate system:
    - Plane wave in +y direction
    - K phased array nodes at (X_k, R, 0), each with N elements along x
    - Target at (x_T(t), 0, z_T) moving in x-direction: x_T(t) = x_0 + v_x*t
    - z_T is a constant offset in z
    - All distances normalised to lambda (so lambda = 1)

Key change from v4: target moves in x (not z), with constant z_T.
Node spacing = sqrt(R) / 2 (half Fresnel zone radius).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class ArrayNode:
    """A single phased array node."""
    x_centre: float
    R: float
    N: int = 16
    d: float = 0.5  # lambda/2

    @property
    def element_positions(self) -> np.ndarray:
        return self.x_centre + np.arange(self.N) * self.d

    @property
    def aperture(self) -> float:
        return (self.N - 1) * self.d

    @property
    def fresnel_zone_radius(self) -> float:
        return np.sqrt(self.R)


@dataclass
class Target:
    """A single 2D target with shadow silhouette, moving in x-direction."""
    x_0: float           # initial x-coordinate
    v_x: float           # velocity in x-direction (wavelengths per time unit)
    z_T: float           # constant z-offset (wavelengths)
    silhouette: np.ndarray = None
    pixel_size: float = 10.0
    pixel_centres_x: np.ndarray = None
    pixel_centres_z: np.ndarray = None

    def x_T(self, t: float) -> float:
        """Target x-coordinate at time t."""
        return self.x_0 + self.v_x * t

    def xi_at_node(self, t: float, node_x: float, R: float) -> float:
        """Fresnel zone parameter ξ_k(t) = |x_T(t) - X_k| / sqrt(R)."""
        return abs(self.x_T(t) - node_x) / np.sqrt(R)


@dataclass
class SystemConfig:
    """Complete system configuration."""
    nodes: List[ArrayNode] = field(default_factory=list)
    target: Target = None
    fc_ghz: float = 30.0
    xi_threshold: float = 2.0
    snr_db: float = 20.0

    @property
    def wavelength_m(self) -> float:
        return 3e8 / (self.fc_ghz * 1e9)

    @property
    def K(self) -> int:
        return len(self.nodes)


def create_system(
    K: int = 4,
    N: int = 16,
    R: float = 1e4,
    x_0: float = 2000.0,
    v_x: float = -1.0,
    z_T: float = 10.0,
    target_shape: str = "rectangle",
    target_width: float = 100.0,
    target_height: float = 80.0,
    M_pixels: int = 20,
    pixel_size: float = 10.0,
    snr_db: float = 20.0,
    xi_threshold: float = 2.0,
) -> SystemConfig:
    """
    Create system with uniform node spacing = sqrt(R) / 2.

    Parameters
    ----------
    K : int
        Number of nodes.
    N : int
        Elements per node.
    R : float
        Target-to-receiver distance (wavelengths).
    x_0 : float
        Target initial x-coordinate (wavelengths).
    v_x : float
        Target velocity in x-direction.
    z_T : float
        Constant z-offset (wavelengths).
    target_shape : str
        "rectangle", "triangle", or "circle".
    target_width : float
        Target width in x' (wavelengths).
    target_height : float
        Target height in z' (wavelengths).
    M_pixels : int
        Pixels per side.
    pixel_size : float
        Pixel side length (wavelengths).
    snr_db : float
        SNR in dB.
    xi_threshold : float
        Zone transition threshold.

    Returns
    -------
    SystemConfig
    """
    r_F = np.sqrt(R)
    spacing = r_F / 2.0

    nodes = []
    for k in range(K):
        x_k = (k - (K - 1) / 2) * spacing
        nodes.append(ArrayNode(x_centre=x_k, R=R, N=N))

    half_grid = M_pixels / 2
    centres_x = (np.arange(M_pixels) - half_grid + 0.5) * pixel_size
    centres_z = (np.arange(M_pixels) - half_grid + 0.5) * pixel_size

    silhouette = _create_silhouette(
        target_shape, target_width, target_height,
        centres_x, centres_z, pixel_size
    )

    target = Target(
        x_0=x_0, v_x=v_x, z_T=z_T,
        silhouette=silhouette,
        pixel_size=pixel_size,
        pixel_centres_x=centres_x,
        pixel_centres_z=centres_z,
    )

    config = SystemConfig(
        nodes=nodes, target=target, fc_ghz=30.0,
        xi_threshold=xi_threshold, snr_db=snr_db,
    )

    # Compute M_total for simulation
    # Target starts at x_0, crosses all nodes, exits the other side
    # Total time: target traverses from x_0 to -x_0 (symmetric)
    T_total = abs(2 * x_0 / v_x)
    z_A_bound = xi_threshold * r_F
    t_zone_A_approx = 2 * z_A_bound / abs(v_x)
    n_pixels = M_pixels ** 2
    M_A_needed = max(int(3 * n_pixels / K), 400)
    dt_needed = t_zone_A_approx / M_A_needed
    config._M_total = max(int(T_total / dt_needed), 2000)

    return config


def _create_silhouette(shape, width, height, centres_x, centres_z, pixel_size):
    """Create binary silhouette on the pixel grid."""
    M_z = len(centres_z)
    M_x = len(centres_x)
    silhouette = np.zeros((M_z, M_x))
    for p, zp in enumerate(centres_z):
        for q, xq in enumerate(centres_x):
            if shape == "rectangle":
                if abs(xq) <= width / 2 and abs(zp) <= height / 2:
                    silhouette[p, q] = 1.0
            elif shape == "triangle":
                if abs(zp) <= height / 2:
                    frac = 1.0 - (zp + height / 2) / height
                    if abs(xq) <= width / 2 * frac:
                        silhouette[p, q] = 1.0
            elif shape == "circle":
                r = min(width, height) / 2
                if xq**2 + zp**2 <= r**2:
                    silhouette[p, q] = 1.0
    return silhouette
