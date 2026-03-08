# API Reference

## `src.geometry`

### `ArrayNode`
Dataclass representing one N×N URA receive node.

| Attribute | Type | Description |
|-----------|------|-------------|
| `x_centre` | float | Centre x-coordinate X̄_k (λ) |
| `R_0` | float | Nominal propagation distance |
| `N` | int | Elements per dimension (default 16) |
| `d` | float | Element spacing (default 0.5 = λ/2) |
| `N2` | property | Total elements N² |

### `Target`
Dataclass for the rigid-body target with oblique heading.

| Attribute | Type | Description |
|-----------|------|-------------|
| `x_0` | float | Initial x-coordinate |
| `v` | float | Speed (λ/time) |
| `theta` | float | Heading angle (rad) |
| `z_T` | float | Constant altitude |
| `silhouette` | ndarray | M_z × M_x binary shadow |
| `pixel_size` | float | 2δ (default 10.0) |
| `v_x`, `v_y` | property | Velocity components |
| `delta` | property | Pixel half-width δ |

Methods: `x_T(t)`, `y_T(t)`, `R(t, R_0)`, `delta_x_k(t, X_k)`, `xi_k(t, X_k, R_0)`

### `SystemConfig`
Complete system configuration dataclass.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `nodes` | list | [] | ArrayNode list |
| `target` | Target | None | Target instance |
| `R_0` | float | 1e4 | Propagation distance |
| `delta_node` | float | 50 | Inter-node spacing |
| `dt` | float | 1.0 | Snapshot interval |
| `xi_threshold` | float | 2.0 | Zone boundary |
| `snr_db` | float | 20.0 | Per-element SNR |
| `gamma_profile` | float | 5000 | Algorithm 1 step size |
| `tau` | float | 1e-6 | TV weight |
| `N_S, N_D, N_V` | int | 200, 20, 20 | Iteration counts |

### `create_system(**kwargs) → SystemConfig`
Factory function; all v9 Table 1 parameters as keyword arguments.

---

## `src.fresnel`

### `fresnel_coeff_x(xp, delta, dx_k, R) → complex`
x-direction Fresnel coefficient F_q^{(k)}(t), eq. (13).

### `fresnel_coeff_z(zp, delta, z_T, R) → complex`
z-direction Fresnel coefficient G_p(t), eq. (14).

### `build_F_vectors(node, target, t_values, R_0, ...) → ndarray (M, M_x)`
Pre-compute all F_q^{(k)}(t_m) for one node.

### `build_G_vector(target, t_values, R_0, ...) → ndarray (M, M_z)`
Pre-compute all G_p(t_m). Time-varying due to R(t).

### `compute_fssr(F, G, P, R_values) → ndarray (M,)`
Vectorised FSSR: ε_k(t) = |1 + (i/R(t)) G^T P F|².

### `compute_fssr_for_node(node, target, t_values, R_0, ...) → ndarray (M,)`
Convenience: build coefficients and compute FSSR for one node.

### `add_fssr_noise(fssr, snr_db) → ndarray`
Add AWGN to FSSR measurements.

---

## `src.signals`

### `steering_2d(ell_x, ell_z, N) → ndarray (N²,)`
2D steering vector a(ℓ_x, ℓ_z) = a_x ⊗ a_z.

### `direction_cosines(dx_k, R, z_T) → (ell_x, ell_z)`
Azimuth and elevation direction cosines (eqs. 22–23).

### `generate_zone_b_signal(node, target, t_values, R_0, snr_db) → ndarray (N², M)`
Zone B received signal r^(B)_{k,n_x,n_z}(t), eq. (21).

### `beamform_2d(X, ell_x, ell_z, N) → ndarray (M,)`
Steer N²-element array toward (ℓ_x, ℓ_z).

### `measure_fssr_from_signal(X, N) → ndarray (M,)`
Broadside beamform → |ỹ_k(t)|².

---

## `src.zone_b`

### `estimate_parameters_zone_b(nodes, signals, dt, R_0_nominal, ...) → dict`
Full Zone B pipeline: DOA → Doppler → closed-form init → NLS refinement.

Returns dict with keys: `v_x, v_y, R_0, x_0, z_T, doa_rates, doppler_rates`.

Internal functions:
- `extract_azimuth_covariance(R_full, N)` — eq. (31)
- `extract_elevation_covariance(R_full, N)` — eq. (32)
- `estimate_doa_music_1d(R_sub, N)` — 1D MUSIC on marginal covariance
- `estimate_doa_esprit_1d(R_sub)` — 1D LS-ESPRIT
- `stft_doppler(y, dt, W, H)` — STFT with argmax peak detection, eq. (45)
- `nls_refinement(phi_init, ...)` — Gauss-Newton/LM for general θ, eq. (54)

---

## `src.zone_a`

### `prox_tv(z, lam, M_z, M_x, N_D) → ndarray`
TV proximal operator (Algorithm 2): dual iteration with [0,1] box constraints.

### `retrieve_shadow_profile(fssr_obs, nodes, target, t_per_node, R_0, v_x, v_y, z_T, x_0, ...) → (P, cost)`
Algorithm 1: gradient descent with TV denoising. Returns continuous P ∈ [0,1]^{M_z×M_x}.

### `joint_estimation(fssr_obs, nodes, target, t_per_node, v_x_init, v_y_init, z_T_init, R_0_init, x_0_init, ...) → dict`
Algorithm 3: joint parameter + profile estimation. Optimises 5 parameters
(v_x, v_y, z_T, R_0, x_0) around profile retrieval. Returns dict with
`P_continuous, P_binary, v_x, v_y, z_T, R_0, x_0, costs`.

### `compute_retrieval_accuracy(P_est, P_true) → dict`
Metrics: `pixel_accuracy, iou, true_positive_rate, false_positive_rate`.

---

## `src.pipeline`

### `run_simulation(config, doa_method, verbose) → dict`
End-to-end simulation. Returns dict with:
- `config` — SystemConfig
- `fssr` — clean/noisy FSSR, zone masks
- `zone_b` — parameter estimates
- `zone_a_estimated` — Algorithm 3 results
- `zone_a_oracle` — oracle (true params) results
- `timing` — per-stage wall clock times

---

## `src.visualisation`

### `plot_all(results, save_dir)`
Generate all figures: FSSR curves, zone classification, shadow comparison,
cost convergence, summary.
