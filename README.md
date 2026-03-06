# Distributed Phased Array Forward Scatter Radar (FSR) — v5

A Python simulation of a **distributed phased array forward scatter radar** system with target moving along the **x-axis** (constant z_T offset), using **gradient descent with TV denoising** for shadow profile retrieval and **joint motion parameter estimation**.

## Overview

This project implements the mathematical model and signal processing algorithms for a novel FSR architecture:

- **K uniformly-spaced phased array nodes** (spacing = √R/2, half Fresnel zone radius)
- **Target moves in x-direction** with constant z_T offset
- **Zone B** (far from baseline): DOA + Doppler → estimate v_x, R, x_0
- **Zone A** (near baseline): Gradient descent + TV denoising → shadow profile + refined v_x, z_T, R
- **Joint parameter estimation**: simultaneous v_x and z_T estimation using gradient descent on the cost function

### Key Changes from v4

1. **Target motion**: x-axis (not z-axis), with constant z_T
2. **Node placement**: uniform at √R/2 spacing (no inner/outer tiers)
3. **Zone A algorithm**: gradient descent with TV denoising (replaces NLS)
4. **R estimation**: R is unknown, estimated from Zone B DOA + Doppler
5. **Joint estimation**: v_x, z_T refined in Zone A using Algorithm from [IETDraft4]

## Architecture

```
Zone B (target far from all nodes)           Zone A (target near nodes)
┌──────────────────────────────────┐        ┌────────────────────────────────┐
│ ALL nodes:                        │        │ ALL nodes:                      │
│   DOA → ℓ_x(t) time series      │        │   Beamform → FSSR samples      │
│   DOA rate β → v_x/R            │──────→  │   Gradient descent + TV denoise│
│   Doppler rate → v_x²/R         │(handoff)│   Joint v_x, z_T estimation    │
│   → v_x, R, x_0 estimates       │        │   → shadow profile P            │
│                                  │        │   → refined v_x, z_T, R         │
└──────────────────────────────────┘        └────────────────────────────────┘
     Target = point source                       Target = extended 2D
```

## Coordinate System

- **Plane wave** in +y direction (TX at y → −∞)
- **Receivers** at (X_k, R, 0), arrays along x-axis
- **Target** at (x_T(t), 0, z_T), moves in x-direction: x_T(t) = x_0 + v_x·t
- All distances **normalised to wavelength λ**

## FSSR Model

```
ε_k(t) = |1 + (i/R) Σ_{p,q} F_q^(k)(t) · G_p · P_{p,q}|²
```

where:
- F_q^(k)(t): x-Fresnel coefficient (time-varying, depends on Δx_k(t) = x_T(t) − X_k)
- G_p: z-Fresnel coefficient (constant, depends on z_T)
- P_{p,q}: pixel value at (Z'_p, X'_q)

## Installation

```bash
git clone https://github.com/DefengDavidHuang/gradient.git
cd gradient
pip install -r requirements.txt
```

## Usage

### Basic Run

```bash
python main.py                   # Default: K=4, rectangle, SNR=20dB
python main.py --shape triangle  # Triangular target
python main.py --K 6 --snr 30   # 6 nodes, 30dB SNR
python main.py --no-plot         # Skip figures
```

### All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--K` | 4 | Number of nodes |
| `--N` | 16 | Elements per node |
| `--R` | 1e4 | Target distance (wavelengths) |
| `--x0` | 2000 | Target initial x-coordinate |
| `--vx` | -1.0 | Velocity in x-direction |
| `--z_T` | 10 | Constant z-offset |
| `--shape` | rectangle | Target shape: rectangle, triangle, circle |
| `--width` | 100 | Target width (wavelengths) |
| `--height` | 80 | Target height (wavelengths) |
| `--snr` | 20 | SNR in dB |
| `--xi_th` | 2.0 | Zone transition threshold |
| `--doa` | fft | DOA method: fft, music, esprit |
| `--output` | results | Output directory |

### From Python

```python
from src.geometry import create_system
from src.simulation import run_simulation

config = create_system(K=4, R=1e4, x_0=2000, v_x=-1.0, z_T=10)
results = run_simulation(config, verbose=True)
```

## Project Structure

```
gradient/
├── main.py                 # CLI entry point
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── geometry.py         # System geometry, Target (x-motion), node placement
│   ├── fresnel.py          # Fresnel diffraction, FSSR (F_q time-varying, G_p constant)
│   ├── doa.py              # MUSIC/ESPRIT DOA estimation
│   ├── zone_b.py           # Zone B: DOA + Doppler → v_x, R, x_0
│   ├── zone_a.py           # Zone A: gradient descent + TV denoising + joint estimation
│   ├── simulation.py       # Full Zone B → A pipeline
│   └── visualisation.py    # Plotting
├── docs/
│   └── mathematical_model_v5.md   # Full mathematical derivation
└── results/                # Output figures (generated)
```

## Algorithms

### Zone B: Parameter Estimation
- DOA time series → linear fit → DOA rate β = v_x/R
- Doppler time series → linear fit → chirp rate f_dot = −v_x²/R
- **v_x = −f_dot/β**, **R = f_dot/β²**, **x_0 = α·R + X_k**

### Zone A: Gradient Descent with TV Denoising
1. **Inner loop** (Algorithm 1 from [Draft5]):
   - Gradient step on fidelity: ∇F(P) using closed-form gradient
   - TV proximal denoising with [0,1] clipping
   - Nesterov momentum acceleration
2. **Outer loop** (Algorithm 1 from [IETDraft4]):
   - Numerical gradient of cost w.r.t. v_x, z_T, R
   - Update with Nesterov acceleration
   - Zone B estimates as initial values

### Scale Ambiguity
For a single receiver, FSSR has scale ambiguity (μ scaling). With **multiple receivers at known positions**, the ambiguity is broken because node positions X_k are not scaled. This enables direct R estimation from multi-node data.

## References

1. X. Shen and D. Huang, "Pixel Based Shadow Profile Retrieval in Forward Scatter Radar Using Forward Scatter Shadow Ratio," *IEEE Access*, 2024.

2. X. Shen and D. Huang, "A Gradient-based Method with Denoising for Target Shadow Profile Retrieval in Forward Scatter Radar," [Draft5].

3. X. Shen and D. Huang, "Retrieving Target Motion Parameters Together with Target Shadow Profile in Forward Scatter Radar," [IETDraft4].

## License

MIT License
