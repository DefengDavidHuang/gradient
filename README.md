# Distributed Phased Array Forward Scatter Radar — Simulation (v9)

Python simulation of a distributed phased array forward scatter radar (FSR)
system, implementing the mathematical model and algorithms from
[v9_model_and_algorithms.tex](docs/v9_model_and_algorithms.tex).

## Overview

A 3D rigid-body target moves obliquely across a baseline of **K** planar
(N×N) receive arrays. The system estimates target motion parameters and
retrieves the 2D shadow profile through two processing zones:

- **Zone B** (far from baseline): 2D DOA + Doppler analysis → coarse
  parameter estimation (v_x, v_y, R_0, x_0, z_T)
- **Zone A** (near baseline): gradient-based shadow profile retrieval with
  TV denoising and joint parameter refinement

## Installation

```bash
pip install numpy scipy matplotlib
```

Or:

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run with all defaults (Table 1 of v9 document)
python main.py

# Skip plotting
python main.py --no-plot

# Custom parameters
python main.py --K 6 --N 16 --R_0 10000 --snr 30 --theta 0.1

# Faster test run (fewer iterations)
python main.py --N_S 100 --N_V 10 --no-plot
```

## CLI Parameters

All parameters from the v9 document (Table 1) are exposed as CLI options:

### System Geometry

| Option | Default | Description |
|--------|---------|-------------|
| `--K` | 4 | Number of receiver nodes |
| `--N` | 16 | Elements per array dimension (N×N URA) |
| `--R_0` | 10000 | Nominal propagation distance (λ) |
| `--delta_node` | √R_0/2 | Inter-node spacing (λ) |

### Target Motion

| Option | Default | Description |
|--------|---------|-------------|
| `--x_0` | 2000 | Initial target x-coordinate (λ) |
| `--v` | 1.0 | Target speed (λ/time) |
| `--theta` | 0.0 | Heading angle (rad), 0 = pure +x |
| `--z_T` | 10.0 | Target altitude (λ) |

### Sampling & Grid

| Option | Default | Description |
|--------|---------|-------------|
| `--dt` | 1.0 | Snapshot sampling interval |
| `--M_z` | 20 | Pixel rows |
| `--M_x` | 20 | Pixel columns |
| `--pixel_size` | 10.0 | Pixel side length 2δ (λ) |

### Target Shape

| Option | Default | Description |
|--------|---------|-------------|
| `--shape` | rectangle | Target shape (rectangle/triangle/circle) |
| `--target_width` | 100 | Shadow width (λ) |
| `--target_height` | 80 | Shadow height (λ) |

### Noise & Zones

| Option | Default | Description |
|--------|---------|-------------|
| `--snr` | 20 | Per-element SNR (dB) |
| `--xi_th` | 2.0 | Zone A/B threshold |

### Algorithm Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--gamma` | 5000 | Profile retrieval step size |
| `--tau` | 1e-6 | TV regularisation weight |
| `--N_S` | 200 | Profile retrieval iterations (Algorithm 1) |
| `--N_D` | 20 | TV denoising iterations (Algorithm 2) |
| `--N_V` | 20 | Outer parameter iterations (Algorithm 3) |

### Output Control

| Option | Default | Description |
|--------|---------|-------------|
| `--no-plot` | False | Skip figure generation |
| `--save_dir` | results | Output directory |
| `--quiet` | False | Minimal output |

## Mathematical Model

The simulation implements the complete model from the v9 document:

1. **FSSR Model** (Section 3): Fresnel diffraction integral discretised into
   separable form ε_k(t) = |1 + (i/R(t)) G^T P F^(k)|²

2. **Zone B** (Section 6):
   - 2D URA with N² elements per node
   - Phase-compensated covariance for DOA smearing
   - Separable azimuth/elevation DOA via MUSIC or ESPRIT
   - STFT-based Doppler extraction
   - Nonlinear LS for general heading angle θ

3. **Zone A** (Section 7):
   - Algorithm 1: Gradient descent + TV proximal denoising
   - Algorithm 2: Fast TV prox via dual iteration
   - Algorithm 3: Joint estimation of 5 parameters (v_x, v_y, z_T, R_0, x_0)
     with shadow profile

## Output

The simulation prints:
- Zone B parameter estimates vs ground truth
- Zone A convergence (cost per iteration)
- Final metrics: pixel accuracy, IoU, TPR, FPR

Figures saved to `results/`:
- `fssr_curves.png` — FSSR time series per node
- `zone_classification.png` — ξ_k(t) and zone boundaries
- `shadow_comparison.png` — true vs retrieved profiles
- `cost_convergence.png` — Algorithm 3 cost history
- `summary.png` — text summary

## Project Structure

```
main.py              CLI entry point
src/
  __init__.py        Package metadata
  geometry.py        System config, nodes, target, pixel grid
  fresnel.py         Fresnel coefficients and FSSR computation
  signals.py         Zone B signal generation (2D URA model)
  zone_b.py          DOA, Doppler, NLS parameter estimation
  zone_a.py          TV denoising, gradient descent, Algorithm 3
  pipeline.py        End-to-end simulation orchestration
  visualisation.py   Plotting functions
docs/
  v9_model_and_algorithms.tex   Mathematical model (LaTeX)
  API.md                        API reference
```

## References

1. X. Shen and D. Huang, "Pixel based shadow profile retrieval in forward
   scatter radar using forward scatter shadow ratio," *IEEE Access*, vol. 12,
   pp. 60467–60474, 2024.

2. X. Shen and D. Huang, "A gradient-based method with denoising for target
   shadow profile retrieval in forward scatter radar," submitted, 2025.

3. X. Shen and D. Huang, "Retrieving target motion parameters together with
   target shadow profile in forward scatter radar," submitted to *IET Radar,
   Sonar & Navigation*, 2025.
