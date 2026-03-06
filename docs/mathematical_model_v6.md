# Mathematical Model v6: 2D Receive Arrays for Altitude Estimation
## Distributed Phased Array FSR with Gradient Descent Retrieval

---

## Summary of Changes from v5.2

| Component | v5.2 | v6 |
|-----------|------|-----|
| Array geometry | N-element ULA along x | N×N URA in x–z plane |
| Elements per node | N | N² |
| Beamformed noise | σ²/N | σ²/N² |
| Measurable DOA | ℓ_x only | (ℓ_x, ℓ_z) both |
| Zone B z_T estimation | z_T = 0 default | ẑ_T = ℓ̂_z · R̂₀ from elevation DOA |
| Zone A z_T init | z_T⁽⁰⁾ = 0 | z_T⁽⁰⁾ = ẑ_T^(B) |
| Steering vector | a_x(ℓ_x) ∈ ℂ^N | a_x(ℓ_x) ⊗ a_z(ℓ_z) ∈ ℂ^(N²) |
| DOA algorithms | 1D MUSIC, 1D LS-ESPRIT | 2D MUSIC (joint/separable), 2D LS-ESPRIT |
| Steered beam (Doppler) | 1D steering | 2D steering (+10log₁₀N dB gain) |
| FSSR model | — | Unchanged |
| 3D target model | — | Unchanged |
| Zone A algorithms | — | Unchanged |

---

## 1. GEOMETRY

### Coordinate System
- Plane wave illuminator: incident in +y direction
- K phased array nodes at y = R₀; node k centred at (X̄_k, R₀, 0)
- All distances normalised to λ (λ = 1, d = 1/2)

### 2D Planar Array (NEW in v6)

Each node has **N × N elements** in a uniform rectangular array (URA) in the x–z plane.
Element (n_x, n_z) of node k is at:

  **p_{k,n_x,n_z} = (X̄_k + n_x·d, R₀, n_z·d)**

  n_x = 0,...,N-1,  n_z = 0,...,N-1

Each node has N² elements total. The z-dimension elements enable elevation DOA estimation.

### Node Placement
- K nodes uniformly spaced at √R₀/2:
  X̄_k = (k - (K-1)/2) · √R₀/2, k = 0,...,K-1

### Target (unchanged from v5.2)
- 3D rigid body, centre at (x_T(t), y_T(t), z_T)
- x_T(t) = x₀ + v_x·t, v_x = v·cos θ
- y_T(t) = v_y·t, v_y = v·sin θ
- z_T = constant altitude
- R(t) = R₀ - v_y·t

---

## 2. 3D TARGET AND SHADOW PROJECTION (unchanged from v5.2)

- Body-frame shape T(x_b, y_b, z_b) ∈ {0,1}
- Shadow profile F(x',z') in aperture plane
- Pixel-dependent effective propagation: R_eff(x',t) = R(t) - x'·tan θ

---

## 3. FSSR MODEL (unchanged from v5.2)

### Continuous FSSR
  ε_k(t) = |1 + ∬ F(x',z') · (i/R_eff) · exp[iπ/R_eff · ((Δx_k+x')² + (z_T+z')²)] dx'dz'/cos θ|²

### Pixel Discretisation
  ε_k(t) ≈ |1 + (1/cos θ) Σ_{p,q} (i/R_{eff,q}) · F_q^(k)(t) · G_{p,q}(t) · P_{p,q}|²

### Combined Coefficient Matrix
  H_{k,p,q}(t) = i / (cos θ · R_{eff,q}(t)) · F_q^(k)(t) · G_{p,q}(t)
  S_k(t) = Σ_{p,q} H_{k,p,q}(t) · P_{p,q}
  ε_k(t) = |1 + S_k(t)|²

---

## 4. UNIFIED NOISE MODEL

### Complex Signal Noise
At element (n_x, n_z) of node k:
  η_{k,n_x,n_z}(t) ~ CN(0, σ²), σ² = 10^(-SNR_dB/10)

i.i.d. across all N² elements.

### 2D Beamforming and FSSR Measurement

After 2D beamforming toward broadside:
  ỹ_k(t) = (1/N²) Σ_{n_x} Σ_{n_z} x_{k,n_x,n_z}(t) ≈ 1 + S_k(t) + η̃_k(t)

where **η̃_k(t) ~ CN(0, σ²/N²)**.

**SNR improvement**: noise power reduced by factor N compared to v5.2's σ²/N, giving +10·log₁₀(N) dB additional processing gain.

FSSR measured as: b_k(t) = |ỹ_k(t)|²

---

## 5. ZONE CLASSIFICATION (unchanged)

  ξ_k(t) = |Δx_k(t)| / √R(t)
  Zone B: ξ_k > ξ_th,  Zone A: ξ_k ≤ ξ_th

---

## 6. ZONE B: PARAMETER ESTIMATION WITH 2D ARRAYS

### 6.1 Received Signal Model (NEW: 2D phase progression)

At element (n_x, n_z) of node k:

  **x_{k,n_x,n_z}(t) = A_D + G_T(t)·exp(jψ_k(t))·exp(jπ·n_x·ℓ_x^(k)(t))·exp(jπ·n_z·ℓ_z^(k)(t)) + η**

where:
- ℓ_x^(k)(t) = Δx_k(t)/D_k(t) : x-direction cosine (same as v5.2)
- **ℓ_z^(k)(t) = z_T/D_k(t) : z-direction cosine (NEW)**
- D_k(t) = √(Δx_k² + R(t)² + z_T²) : range

The phase term exp(jπ·n_z·ℓ_z) across the z-dimension enables elevation DOA.

### 6.2 2D Steering Vector

  a(ℓ_x, ℓ_z) = a_x(ℓ_x) ⊗ a_z(ℓ_z) ∈ ℂ^(N²)

where a_x(u) = [1, e^(jπu), ..., e^(jπ(N-1)u)]^T and similarly for a_z.

### 6.3 Time-Varying DOA

**Azimuth (ℓ_x)**: varies significantly with time (same issue as v5.2).
**Elevation (ℓ_z)**: varies slowly since z_T is constant and D_k changes slowly in Zone B.
  |ℓ̇_z| << |ℓ̇_x|

→ Phase compensation needed only in x-dimension, not z.

### 6.4 Phase-Compensated 2D Covariance

  x̃_{k,n_x,n_z}(t_m) = x_{k,n_x,n_z}(t_m) · exp(-jπ·n_x·β̂_x·(t_m - t_ref))

No z-compensation needed. The N²×N² compensated covariance:
  R̃_k = (1/M) Σ_m x̃_k(t_m) · x̃_k^H(t_m)

### 6.5 2D DOA Estimation

**Method 1: Joint 2D MUSIC**
  P_MUSIC^(2D)(ℓ_x, ℓ_z) = 1 / [a^H(ℓ_x,ℓ_z) · E_n · E_n^H · a(ℓ_x,ℓ_z)]

Cost: O(N⁶) eigendecomposition.

**Method 2: Separable Estimation (efficient)**
- Row-averaged cov R_x ∈ ℂ^(N×N) → 1D MUSIC/ESPRIT → ℓ̂_x
- Column-averaged cov R_z ∈ ℂ^(N×N) → 1D MUSIC/ESPRIT → ℓ̂_z
- Cost: O(N³) per dimension

**Method 3: 2D LS-ESPRIT**
- Shift-invariance in both x and z dimensions
- Selection matrices via Kronecker structure

### 6.6 Altitude Estimation from Elevation DOA (NEW)

**Far-field**: ℓ_z^(k)(t) ≈ z_T / R(t)

Direct estimate:
  **ẑ_T = ℓ̂_z · R̂(t)**

Time-averaged over K nodes and L time steps:
  ẑ_T^(B) = (1/KL) Σ_{k,ℓ} ℓ̂_z^(k)(t_ℓ) · R̂₀

**Exact inversion**: z_T = ℓ̂_z · √(Δx_k² + R²) / √(1 - ℓ̂_z²)

### 6.7 Achievable z_T Precision

CRB for elevation direction cosine:
  var(ℓ̂_z) ≥ 6 / (π²·ρ·M·N·(N²-1))

Altitude precision (far-field):
  std(ẑ_T) ≈ R · √(6 / (π²·ρ·M·N·(N²-1)))

For N=16, ρ=100 (20 dB), M=100, R=10⁴: **std(ẑ_T) ≈ 1.7λ**

### 6.8 Elevation DOA Time Series

  ℓ_z^(k)(t) ≈ z_T/R₀ + (z_T·v_y/R₀²)·t ≡ α_z + β_z·t

- α_z = z_T/R₀ → gives z_T directly when combined with R̂₀
- β_z = z_T·v_y/R₀² → small, confirms slow variation
- For θ = 0: β_z = 0, elevation DOA is constant

### 6.9 Azimuth DOA and Doppler (unchanged from v5.2)

  ℓ_x^(k)(t) ≈ α_k + β_k·t + γ_k·t²
  β_k = v_x/R₀ + v_y(x₀-X̄_k)/R₀²  (node-dependent)

Doppler via STFT on **2D-steered beam** (NEW: extra N gain):
  y_k(t) = (1/N²) Σ_{n_x,n_z} x_{k,n_x,n_z}(t) · exp(-jπ(n_x·ℓ̂_x + n_z·ℓ̂_z))

### 6.10 Joint Parameter Estimation

Steps 1-2 unchanged (v_x, v_y, R₀, x₀ from azimuth DOA rates + Doppler).

**Step 3 (NEW)**: z_T from elevation DOA:
  ẑ_T^(B) = α̂_z · R̂₀

Cross-check if v_y ≠ 0: ẑ_T = β̂_z · R̂₀² / v̂_y

---

## 7. ZONE A: GRADIENT DESCENT WITH TV DENOISING (unchanged)

### Algorithm 1: Shadow Profile Retrieval
(Identical to v5.2 — gradient descent + Nesterov + TV proximal)

### Algorithm 2: TV Denoising
(Identical to v5.2)

### Algorithm 3: Joint Parameter and Profile Estimation

**Only change**: initialised with ẑ_T^(B) from Zone B elevation DOA instead of z_T = 0.

  Input: ..., ẑ_T⁽⁰⁾ = ẑ_T^(B), ...

---

## 8. SCALE AMBIGUITY (unchanged)

Multi-node observations at known X̄_k break the single-receiver scale ambiguity.

---

## 9. PROCESSING PIPELINE

```
Zone B (2D arrays, all nodes)              Zone A (FSSR retrieval)
┌──────────────────────────────────┐       ┌─────────────────────────────────┐
│ 2D DOA → (ℓ_x, ℓ_z) per node   │       │ 2D beamform → FSSR samples     │
│ Azimuth: phase-compensated cov   │       │ (σ²/N² noise — extra N gain)   │
│ Elevation: direct from z-dim     │       │                                 │
│ ℓ_z → z_T estimate (NEW)        │──────→│ Grad descent + TV denoise       │
│ β_k fit → v_x, v_y              │(init) │ Init z_T = ẑ_T^(B) (NEW)      │
│ Doppler (2D-steered) → R₀       │       │ Joint v_x, v_y, z_T, R₀ est   │
│ → v_x, v_y, R₀, x₀, z_T       │       │ → P̂ (shadow profile)           │
└──────────────────────────────────┘       └─────────────────────────────────┘
```

---

## 10. REFERENCES

1. X. Shen and D. Huang, "Pixel Based Shadow Profile Retrieval in FSR Using FSSR," IEEE Access, 2024.
2. X. Shen and D. Huang, "A Gradient-based Method with Denoising for Target Shadow Profile Retrieval in FSR," [Draft5].
3. X. Shen and D. Huang, "Retrieving Target Motion Parameters Together with Target Shadow Profile in FSR," [IETDraft4].
