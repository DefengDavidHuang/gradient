"""
Distributed Phased Array Forward Scatter Radar (FSR) Simulation — v9
=====================================================================

A complete simulation of a distributed phased-array FSR system where a
three-dimensional rigid-body target moves obliquely across a baseline of
K planar (N×N) receive arrays.

Modules
-------
geometry        System geometry, coordinate definitions, target motion
fresnel         Fresnel diffraction integrals and FSSR computation
signal          Element-level signal generation and beamforming
zone_b          Zone B processing: 2-D DOA, Doppler, joint NLS
zone_a          Zone A processing: gradient descent + TV denoising
pipeline        End-to-end simulation pipeline
visualisation   Plotting and result visualisation

References
----------
[1] X. Shen and D. Huang, IEEE Access 12, 60467–60474, 2024.
[2] X. Shen and D. Huang, submitted, 2025.
[3] X. Shen and D. Huang, submitted to IET RSN, 2025.
"""

__version__ = "9.0.0"
