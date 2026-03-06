# Mathematical Model v5: Target Moving Along x-axis
## Distributed Phased Array FSR with Gradient Descent Retrieval

---

## 1. GEOMETRY

### Coordinate System
- Plane wave illuminator: incident in +y direction (TX at y → -∞)
- K phased array nodes; node k has N elements (λ/2 spacing along x-axis)
- Element n of node k at position: p_{k,n} = (X_k + nd, 0, 0), n = 0,...,N-1, d = λ/2
- All distances normalised to λ unless otherwise stated

### Node Placement
- K nodes uniformly spaced at half the Fresnel zone radius:
  spacing = √R / 2
- Node k at: X_k = (k - (K-1)/2) · √R/2, k = 0,...,K-1

### Target
- Single 2D extended target, shadow silhouette Σ on its local x′-z′ plane
- Target centre at: (x_T(t), 0, z_T), where:
  - x_T(t) = x_0 + v_x · t (moving in x-direction)
  - z_T = constant offset in z
- Unknown parameters: x_0, v_x, z_T, R

---

## 2. FSSR MODEL

### 2.1 Continuous FSSR at Node k

Following [Paper 1], the FSSR at node k when the target centre is at
(x_T(t), 0, z_T) is:

  ε_k(t) = |1 + (i/R) ∬ F(x′,z′)
            · exp(iπ/R · [(x_T(t) - X_k + x′)² + (z_T + z′)²]) dx′dz′|²
                                                                    ... (1)

Key difference from v4: x_T(t) is time-varying; z_T is constant.

### 2.2 Pixel Discretisation

Divide the target shadow profile into M_z × M_x square pixels of side 2d.
Pixel (p, q) has centre (X′_q, Z′_p) and binary value P_{p,q} ∈ {0,1}.

The FSSR (1) is approximated as:

  ε_k(t) ≈ |1 + (i/R) Σ_{p,q} F_q^(k)(t) · G_p · P_{p,q}|²      ... (2)

where the Fresnel coefficients are:

  F_q^(k)(t) = √(R/4i) [erfi(√(iπ/R)(X′_q + d + Δx_k(t)))
             - erfi(√(iπ/R)(X′_q - d + Δx_k(t)))]                  ... (3)

  G_p = √(R/4i) [erfi(√(iπ/R)(Z′_p + d + z_T))
      - erfi(√(iπ/R)(Z′_p - d + z_T))]                             ... (4)

with Δx_k(t) = x_T(t) - X_k = x_0 + v_x·t - X_k.

**Note:** F_q^(k)(t) is time-varying (depends on target x-position relative
to node k), while G_p is constant (z_T fixed). This is the reverse of v4
where F_q was constant and G_p was time-varying.

### 2.3 Vectorised Computation

Define:
- G = [G_0, ..., G_{M_z-1}]^T, shape (M_z,)
- F^(k)(t_m) = [F_0^(k)(t_m), ..., F_{M_x-1}^(k)(t_m)], shape (M_x,)
- P, shape (M_z, M_x)

Then:
  GP = G^T · P → shape (M_x,)   [project out z-dimension]
  S_k(t_m) = (i/R) · F^(k)(t_m) · GP                               ... (5)
  ε_k(t_m) = |1 + S_k(t_m)|²                                       ... (6)

---

## 3. ZONE CLASSIFICATION

### 3.1 Per-Node Fresnel Parameter

  ξ_k(t) = |x_T(t) - X_k| / √R                                    ... (7)

Zone classification for node k:
- Zone B: ξ_k(t) > ξ_th (target far from node k)
- Zone A: ξ_k(t) ≤ ξ_th (target within Fresnel zone of node k)

### 3.2 Sequential Zone Entry

Since target moves in x, different nodes enter Zone A at different times.
Node k enters Zone A when x_T(t_k^A) = X_k ± ξ_th·√R.

The target crosses node k at time t_k* = (X_k - x_0) / v_x.

For v_x < 0 (target approaching from positive x):
- Rightmost nodes are crossed first
- Each node has its own Zone A time window

---

## 4. ZONE B: PARAMETER ESTIMATION

### 4.1 Point Source Model

When ξ_k >> 1, the target is a point source at (x_T(t), 0, z_T) as
seen from node k. The received signal at element n of node k:

  x_{k,n}(t) = A_D + G_T · exp(jψ_k(t)) · exp(jπ · n · ℓ_x^(k)(t)) + noise
                                                                    ... (8)

where:
- A_D = 1 (direct path, broadside)
- G_T = scattered amplitude ∝ target area / R
- ψ_k(t) = π(Δx_k(t)² + z_T²) / R (propagation phase)
- ℓ_x^(k)(t) = Δx_k(t) / D_k(t) (x-direction cosine)
- D_k(t) = √(Δx_k(t)² + R² + z_T²) (range)

### 4.2 DOA Estimation

The x-direction cosine is estimated using MUSIC/ESPRIT/FFT:

  ℓ̂_x^(k)(t) = Δx_k(t) / D_k(t)                                  ... (9)

For far-field (D_k ≈ R): ℓ_x^(k)(t) ≈ Δx_k(t) / R

The DOA is LARGE when target is far (Zone B) and DECREASES to zero as
target crosses the baseline at node k. This means all nodes can perform
DOA estimation in Zone B — no dedicated outer nodes needed.

### 4.3 DOA-Based Velocity and Range Estimation

In the far field, ℓ_x^(k)(t) ≈ (x_0 + v_x·t - X_k) / R is linear in t.

From linear fit: ℓ_x^(k)(t) = α_k + β_k · t

  β_k = v_x / R         → DOA rate (same for all nodes)             ... (10)
  α_k = (x_0 - X_k) / R → DOA intercept (node-dependent)           ... (11)

### 4.4 Doppler-Based Velocity Estimation

Instantaneous Doppler at node k (normalised units, λ=1):

  f_D^(k)(t) = -v_x · ℓ_x^(k)(t) ≈ -v_x · Δx_k(t) / R           ... (12)

This is also linear in t:
  f_D^(k)(t) = f_0^(k) + f_dot · t

  f_dot = -v_x² / R      (Doppler chirp rate, same for all nodes)  ... (13)
  f_0^(k) = -v_x · (x_0 - X_k) / R                                ... (14)

### 4.5 Joint v_x, R, x_0 Estimation

Combining DOA rate (10) and Doppler rate (13):

  v_x = -f_dot / β = -(-v_x²/R) / (v_x/R) = v_x ✓

Practically:
  |v_x| = |f_dot| / |β|                                             ... (15)
  R = |v_x| / |β| = f_dot / β²                                     ... (16)
  x_0 = α · R + X_k                                                 ... (17)

Sign of v_x: from DOA trend direction (β < 0 means v_x < 0 if R > 0).

### 4.6 z_T Estimation (Initial Guess)

z_T does not directly affect DOA or Doppler in the far field. Options:
1. Initial guess z_T = 0 (refined in Zone A)
2. Estimate from FSSR amplitude pattern: the depth of the FSSR dip
   depends on z_T through the Fresnel z-integral

---

## 5. ZONE A: GRADIENT DESCENT WITH DENOISING

### 5.1 Problem Formulation

Given noisy FSSR samples b_{k,ℓ} from K nodes and L time steps:

  P̂ = argmin_P { ℱ(P) + τ·ℛ(P) }                                  ... (18)

where:
  ℱ(P) = Σ_k Σ_ℓ [ε_k(t_ℓ; P) - b_{k,ℓ}]²                       ... (19)

  ℛ(P) = TV(P) = Σ_{p,q} √((P_{p,q}-P_{p+1,q})² + (P_{p,q}-P_{p,q+1})²)
                                                                    ... (20)

### 5.2 Gradient of the Fidelity Function

For sample (k, ℓ), let S = (i/R) Σ_j H_j P_j. Then:

  ∂ℱ_{k,ℓ}/∂P_{p,q} = 4[ε_k(t_ℓ) - b_{k,ℓ}] · Re[(i/R) · F_q^(k)(t_ℓ) · G_p · conj(1 + S)]
                                                                    ... (21)

In matrix form:

  ∇_P ℱ_{k,ℓ} = 4[ε - b] · Re[(i/R) · conj(1+S) · G ⊗ F^(k)(t_ℓ)]
                                                                    ... (22)

where ⊗ denotes outer product.

### 5.3 Shadow Profile Retrieval (Algorithm 1, from [Draft5])

Algorithm 1: Gradient descent with TV denoising
  input: FSSR samples {b_{k,ℓ}}, regularisation τ, step size γ
  set: t ← 1, P̂_0 ← 0, s_0 ← P̂_0, q_0 ← 1
  repeat:
    γ_t ← γ / √t
    z_t ← s_{t-1} - (γ_t / KL) Σ_{k,ℓ} ∇ℱ_{k,ℓ}(s_{t-1})
    P̂_t ← prox_ℛ(z_t, γ_t·τ)                    [TV denoising]
    q_t ← (1 + √(1 + 4q_{t-1}²)) / 2             [Nesterov momentum]
    s_t ← P̂_t + (q_{t-1}-1)/q_t · (P̂_t - P̂_{t-1})
    t ← t + 1
  until t = N_S
  return P̂_t

### 5.4 TV Denoising (Algorithm 2, from [Draft5])

The proximal operator prox_ℛ(z, δ) performs TV-based denoising while
enforcing 0 ≤ P_{p,q} ≤ 1. Uses an iterative dual method (see Draft5
Algorithm 2 for full details with operators ℒ, ℒ^T, ℋ_C, ℋ_B).

### 5.5 Simultaneous Velocity Estimation (from [IETDraft4])

Algorithm 2: Gradient descent on velocity
  input: FSSR samples, step size γ_v, perturbation dv, initial v̂_x
  set: n ← 1, q_0 ← 1
  repeat:
    P̂_n ← Algorithm1({b_{k,ℓ}}, v̂_{x,n-1})     [retrieve profile for current v_x]
    C(v) = Σ_{k,ℓ} [b_{k,ℓ} - ε_k(t_ℓ; P̂_n, v)]²
    C′ = [C(v̂ + dv) - C(v̂)] / dv                 [numerical gradient]
    γ_n ← γ_v / n
    v̂_{x,n} ← v̂_{x,n-1} - γ_n · C′
    q_n ← (1 + √(1 + 4q_{n-1}²)) / 2            [Nesterov acceleration]
    v̂_{x,n} ← v̂_{x,n} + (q_{n-1}-1)/q_n · (v̂_{x,n} - v̂_{x,n-1})
    n ← n + 1
  until n = N_V
  return v̂_x, P̂

### 5.6 Joint Parameter Estimation

Extend Algorithm 2 to jointly optimise v_x, z_T (and optionally R):
- Compute numerical gradients ∂C/∂v_x, ∂C/∂z_T, ∂C/∂R
- Update each parameter with Nesterov momentum
- Use Zone B estimates as initial values
- R estimation is partially degenerate (scale ambiguity) but multiple
  nodes at known positions break the ambiguity

---

## 6. SCALE AMBIGUITY (Multi-Node)

From [IETDraft4], for a SINGLE receiver at the origin, FSSR is unchanged if:
- Coordinates scale by μ: (x, z, v) → (μx, μz, μv)
- Distance scales by μ²: R → μ²R
- Shadow profile scales: F′(x′,z′) = F(x′/μ, z′/μ)

For MULTIPLE receivers at known positions X_k (not scaled), the FSSR at
receiver k depends on (x_T - X_k), which does NOT scale correctly:
  μ·x_T - X_k ≠ μ·(x_T - X_k)

Therefore, **the scale ambiguity is broken by multi-node observations**.
The degree of ambiguity breaking depends on |X_k| / |x_T|:
- When |X_k| << |x_T|: ambiguity approximately holds
- When |X_k| ~ |x_T| (Zone A): ambiguity is significantly broken

This enables direct estimation of R from multi-node FSSR data in Zone A.

---

## 7. PROCESSING PIPELINE

```
Zone B (target far from all nodes)         Zone A (target near nodes)
┌────────────────────────────────┐         ┌──────────────────────────────┐
│ ALL nodes:                      │         │ ALL nodes:                    │
│   DOA → ℓ_x(t) at each node   │         │   Beamform → FSSR samples    │
│   DOA rate β → v_x/R          │ ──────→  │   Grad descent + TV denoise  │
│   Doppler rate f_dot → v_x²/R │(handover)│   Simultaneous v_x, z_T est │
│   → v_x, R, x_0 estimates     │         │   → P (shadow profile)        │
│                                │         │   → refined v_x, z_T, R      │
└────────────────────────────────┘         └──────────────────────────────┘
     Target = point source                      Target = extended 2D
     Parameters: v_x, R, x_0                   Shadow profile + refinement
```

---

## 8. REFERENCES

1. X. Shen and D. Huang, "Pixel Based Shadow Profile Retrieval in Forward
   Scatter Radar Using Forward Scatter Shadow Ratio," IEEE Access, 2024.

2. X. Shen and D. Huang, "A Gradient-based Method with Denoising for Target
   Shadow Profile Retrieval in Forward Scatter Radar," [Draft5].

3. X. Shen and D. Huang, "Retrieving Target Motion Parameters Together with
   Target Shadow Profile in Forward Scatter Radar," [IETDraft4].
