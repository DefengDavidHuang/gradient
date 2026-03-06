"""
Distributed Phased Array Forward Scatter Radar (FSR) Simulation
===============================================================

This package simulates a distributed phased array FSR system with a single
target transitioning from Zone B (off-baseline, DOA estimation) to Zone A
(on-baseline, shadow profile retrieval via FSSR).

The coordinate system follows [Paper 1]:
    - Plane wave incident in +y direction (TX at y → -∞)
    - Receivers at (X_k, R, 0), arrays along x-axis, perpendicular to incident wave
    - Target at (x_T, 0, z_T(t)), moves in z-direction with constant x_T

Modules:
    geometry.py     - System geometry and coordinate definitions
    fresnel.py      - Fresnel diffraction integrals and FSSR computation
    zone_b.py       - Zone B processing: DOA estimation, Doppler analysis
    zone_a.py       - Zone A processing: pixel-based shadow profile retrieval
    simulation.py   - Main simulation runner (Zone B → Zone A transition)
    visualisation.py - Plotting and result visualisation

References:
    [Paper 1] X. Shen and D. Huang, "Pixel Based Shadow Profile Retrieval in
              Forward Scatter Radar Using Forward Scatter Shadow Ratio,"
              IEEE Access, vol. 12, pp. 60467-60474, 2024.
    [Paper 2] A. Ajorloo, Y. Qin, and F. Colone, "Multichannel Forward Scatter
              Radar Using Arbitrary Waveforms," IEEE Trans. Aerosp. Electron.
              Syst., vol. 61, no. 6, pp. 17858-17878, Dec. 2025.
"""

__version__ = "0.1.0"
