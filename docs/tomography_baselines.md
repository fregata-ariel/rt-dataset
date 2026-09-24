# RF tomography baselines: intensity, delay, phase, all hybrids, simultaneous and non-simultaneous capture

Target file: `docs/tomography_baselines.md`. Status: design approved 2026-09-25 (all §9.2 defaults adopted); Phase 0 (T01–T10) is implemented on `explore/tomography-p0` and pending integration; its pinned conventions and deviations are in the Phase 0 implementation notes at the end of §8 Phase 0. This document merges three independent proposals: a physics and signal-processing design, a hybrid-fusion design, and an evaluation-and-data design. §1.4 records each conflict between them and why it was resolved the way it was. This revision also resolves the critique of the first draft. The main changes are:

- a LoS atom with its phase fixed by the model;
- a phase-only global gauge;
- τ-aware N-mode back-projection;
- two nested sublattices;
- a G_b-weighted virtual-source operator;
- held-out prediction rules;
- one absolute noise variance per dataset;
- exact Rician/projected-normal CRBs;
- a computed `ill_posed` flag;
- plane-wave L0 data;
- grid-consistent CI thresholds;
- a measured compute budget;
- smaller tasks.

The design is based on master (`src/plateau_rt/domain/rf_camera/{imaging,calibration,camera,delay}.py`) and on the exploratory branches being merged into `dev/m1-rf-optical-dataset`: rich-mock-scene, multi-bs, path-gt-rich, image-sources, observed-impairments, gauge-alignment, partial-observations, rf-gs-toy, solver-profiles and feature/11-optical-reference. At the time of writing, `dev/m1-rf-optical-dataset` equals master, so every branch below is still a pending dependency.

---

## 1. Purpose and scope

### 1.1 Purpose

This suite builds **classical (non-learned) tomography baselines** for the RF Gaussian-Splatting model (#8, #9). Each baseline recovers scene structure from RF-camera captures using a specific kind of information:

- intensity (I), delay (D) and phase (P), alone and in every combination;
- captured **simultaneously** (S: one phase and time reference across views) or **non-simultaneously** (N: coherent only within one capture).

Every configuration runs on the same capture data, the same geometry, the same forward-operator library and the same metrics. That makes three things measurable:

1. how much each information type is worth, and how much *combinations* add (synergy);
2. how much synchronisation is worth, for each information type;
3. how far RF-GS gets beyond a classical solver given the same information (the gain from representation and learning).

Every baseline is an RF-GS loss evaluated on a fixed voxel or point representation and solved classically (§4.4). Among its outputs is a Gaussian list that can initialise RF-GS directly.

### 1.2 The core matrix: 7 × 2 = 14 configurations, plus 3 lattice-completion nodes

- **Information subsets:** I, D, P, ID, IP, DP, IDP. **Sync modes:** S, N. Configurations are named `<subset>-<sync>`, e.g. `ID-N`.
- Canonical P and IP are narrowband, at the DC bin.
- Three extra nodes make both analysis lattices nested (§2.3):
  - **I@n0**, narrowband intensity;
  - **P_W** and **IP_W**, wideband phase with an independent phase unknown per (capture, bin).
- These three nodes are sync-invariant by construction, so each runs once and fills both the S and the N column.
- §2.3 gives the exact definitions.

### 1.3 Extra variants and why they are included

| Variant family | What it adds | Why |
|---|---|---|
| **Omni (aperture-incoherent) column** `X-o` | The same subsets with the aperture's inter-element coherence removed (an independent unknown phase per capture and element): RSS, PDP, per-element CSI | Separates "angle from the aperture" from "information type". It is also the summary-UE end of the RF-GS roadmap. A single-element, S-coherent variant (`IDP-1el`, `DP-1el`) keeps the "single-antenna CSI" reading outside the lattice. |
| **Lattice-completion nodes** I@n0, P_W, IP_W | Nested narrowband and wideband sublattices | Without them, "value of phase" and Shapley values are confounded by data volume (§5.3). |
| **Partial-D ×K variants** P×K-S, IP×K-S (per-bin processing with phase shared across captures) | More looks at several frequencies | Reported, but **excluded from the D-free lattice**: in S mode, cross-view coherence at several f_n resolves differential delay (TDOA). |
| **Sync extras**: S_τ (time-synced, phase unknown), N-sep (φ_v+φ_b, τ_v+τ_b), coherence-group continuum (F5) | Separate the value of *timing* sync from *phase* sync; realistic multi-BS clocks; an S↔N curve instead of two points | Hypothesis H3 (§5.4), tested only under VS models that do not make it true by construction. |
| **N-blind vs N-LoS** for every N cell | Whether the direct path restores sync | A LoS anchor with its phase fixed by the model gives τ_c exactly, and φ_c up to the LoS phase-model error (§3.3). |
| **Stacked-marginals** ablation for every hybrid (ID-stack = {I, D}, …) | Joint observable minus the stacked parts | Answers directly whether combinations matter. |
| **Hybrid fusion strategies F0–F8** (§5) | *How* information is combined, not only *what* is available | The rich hybrid models the user asked for. |
| **Oracle rows**: oracle LoS removal, oracle support, `los=False` trace, optical-geometry prior (F8) | Upper bounds that separate the failure sources | Always reported separately and never ranked with the non-oracle rows. |
| **Model-mismatch sweep** (spherical wavefront, beam squint, element jitter) | NMSE floor of the plane-wave operator | L0 core data matches Sionna's plane-wave model, so the mismatch is a separate axis (§6.2). |
| **I-T transmission tomography (RTI)**, optional | Building occupancy from link attenuation | The only baseline sensitive to building interiors. It needs hundreds of links, so it is deferred and conditional (Q2). |

Out of scope: learned models, and scene changes between captures (the static-scene assumption, Q4).

### 1.4 Conflicts between the proposals and how they were resolved

| # | Conflict | Decision | Reason |
|---|---|---|---|
| C1 | Letter semantics. Two proposals used I = angle-resolved power, D = angle-integrated PDP, P = phase, and one used DP = single-element CSI. The third used I = amplitude, D = inter-frequency coherence, P = carrier phase, with angle always available from the camera. | **Invariance semantics.** I = amplitude, D = inter-frequency coherence (time of flight), P = carrier phase. Every core configuration has the aperture's angle resolution. The other readings survive as the omni column and `X-1el`. | The user asked for intensity, delay and phase, not angle. The camera natively resolves angle. Shapley analysis is run on two nested sublattices (§2.3, §5.3). |
| C2 | P definition: unit-modulus DC bin, per-frequency with compounding, or complex narrowband at the centre bin | **P = unit-modulus aperture field at the DC bin n0 = N/2. IP = complex field at n0.** Wideband phase without delay is P_W/IP_W, with a phase unknown per (capture, bin). Compounding with the phase shared across captures is the partial-D ×K variant. | The DC bin is immune to timing offsets. A single bin cannot leak group delay. Per-(capture, bin) phase unknowns are the only leak-free wideband form. |
| C3 | D-only: per-pixel normalised delay distribution, or angle-integrated PDP | **D = per-pixel return-delay list** from a fixed CFAR, with no amplitude. T(u,t) is kept as a variant; the PDP is ID-o. | Follows from C1. A fixed CFAR makes the one-bit detection leak identical across configurations. |
| C4 | Fusion: joint multi-likelihood MAP, or finest likelihood only | **The finest-likelihood rule (R1) is the default.** Joint MAP is hybrid F4, with coarser terms weighted by model-discrepancy variances. | Statistics derived from the same Y double count noise when multiplied. |
| C5 | Windowing: always, or for detection only | A **fixed Taylor window (−35 dB, n̄ = 4)** for back-projection maps, detection and noise estimation. Model-based solvers use unwindowed data. | Unwindowed peak lists are dominated by −13 dB sidelobes (image-sources: 15 of 18 peaks spurious). |
| C6 | Grid spacing: 0.5 m or 1 m | E1 maps on the physical grid at **0.5 m**; VS grid at **1 m**; E2 on **pruned supports**; coherent cross-view processing only in **λ/4 ROI windows** or gridless; CI at 2 m with sub-voxel refinement. | Compute budget (§4.1). Coherent full-scene grids are infeasible (§3.5). |
| C7 | View layout: 12-view ring, 8-view ring, or 64-pose bank | **64-pose bank per scene with nested subsets** (`nested_view_order`). Core V = 16, B = 1. | Nested subsets give monotone, low-variance curves. |
| C8 | SNR reference: full CFR or scattered power | **One absolute σ² per dataset**, σ² = P_ref/10^{SNR/10}. P_ref is the full-CFR power (both hemispheres summed) of a fixed reference capture: the median-power LoS-visible training capture. The same σ² is applied to every (v, b, h). The achieved per-capture SNR and the scatter-referenced SNR are recorded. | Thermal noise is fixed. Setting noise per capture would make weak NLoS captures artificially clean. |
| C9 | Tx pattern: NumPy port or isotropic | **NumPy tr38901 (V-pol) plus an `iso` option.** G_b enters both the BV and the VS operators (first-order VS: along the mirrored departure direction, §3.2). | The per-view direct-path amplitude ratio (0.21–2.38) comes from the BS pattern. |
| C10 | Headline N row: blind or LoS-anchored | **Both**. Stratified by per-(v, b) LoS visibility where LoS-free captures exist; otherwise N-blind is a *strategy* that ignores the LoS. | Anchoring uses only known BS and UE geometry. |
| C11 | Package name | `src/plateau_rt/domain/rf_tomography/` (NumPy/SciPy only) | A separate subsystem that depends on `rf_camera`. |
| C12 | Number of delay bins | Tomography profile uses **N = 128** (383.7 m unambiguous). 64-bin data is supported with a periodic operator and a wrap flag. | The longest rich-mock path (179 m) nearly reaches the 192 m wrap at 64 bins. |
| C13 | LoS handling in physical space | Known-geometry nuisance atoms. **LoS atom: phase fixed by the model**, a_LoS = g_c·G_b(d)·P_rx(u)·λ/(4πr)·e^{−jkr} with g_c real and > 0. **Ground-bounce atom: free complex amplitude** (its reflection coefficient is unknown). For power data, the amplitudes are free and nonnegative. In N mode they are fitted jointly with (φ_c, τ_c). The LoS phase-model error (model vs path-GT `a_baseband`) is recorded and is the floor on φ_c recovery. | With a free complex LoS amplitude, only the product with e^{jφ_c} is identifiable, so the LoS would give τ_c but never φ_c. |

---

## 2. Observation models

### 2.1 Notation

- **Capture** c = (v, b): view v = 1..V with UE pose (p_v, R_v). R_v is world-from-local, with columns local x (forward), y (left), z (up). BS b = 1..B at position t_b, with complex pattern G_b(ω).
- **Hemisphere** h ∈ {F, K}: front u_x ≥ 0 (index 0), back u_x < 0 (index 1), as in `rf_camera_split`.
- **Element** m = (r, col): 8×8 at spacing d = λ_c/2, with local offset q_m = (0, y_col, z_r). The index-to-offset map must match `imaging.reshape_planar_column_first` and the row flip in `calibration.calibrate_angular_cfr` (pinned by T05).
- **Frequency** n = 0..N−1: δf_n = (n − N/2)·B/N, f_n = f_c + δf_n. **DC bin n0 = N/2.** k_c = 2π/λ_c.
- **Raw data** Y[v, b, h, r, col, n], complex128. It is `aperture_cfr[b, h, r, col, n]` of view v: manifest v3 from multi-bs, or v2 read as B = 1.
- **Angle-delay volume** c_{c,h}(u, t) = Σ_{m,n} w_m w_n Y e^{−j k_c u·q_m} e^{+j2π δf_n t}.
  - The unitary, unpadded version (w = 1) is the domain of every likelihood, and its noise is white CN(0, σ²).
  - A Taylor-windowed version, oversampled 8× in angle (64 per axis) and 8× in delay, is used for E1 and for detection.

| Quantity (3.5 GHz, 8×8 @ 0.5λ, 100 MHz) | Value |
|---|---|
| λ_c | 85.65 mm; λ/4 = 2.1 cm |
| Aperture span | 3.5λ = 0.30 m (diagonal 0.42 m). The plane-wave phase error is k(\|q\|² − (u·q)²)/(2r). For this aperture it is 0.17 rad max (0.08 rms) at 10 m, 0.04 rad max at 40 m, and falls below 1e-2 only beyond about 170 m. Sionna (`synthetic_array=True`) uses the plane-wave model, so the operator matches the data exactly. Spherical wavefronts are a mismatch sweep. |
| Angular cell (direction cosine) | 0.25 (≈ 14° at boresight). Cross-range 7.5 m at 30 m, 10 m at 40 m, 28 m at 112 m |
| Delay cell | 1/B = 10 ns = 3.0 m of path length |
| Unambiguous delay | N/B: 640 ns = 191.9 m (N = 64); 1280 ns = 383.7 m (N = 128) |
| Fractional bandwidth | 2.9 %, about 35 carrier cycles per delay main lobe (the cause of cycle skipping) |
| Samples per capture-hemisphere | 64 × N complex |
| Delay CRB, isolated path, 20 dB per sample | ≈ 6 ps (2 mm); for a path 30 dB below LoS, ≈ 0.2 ns (6 cm) |

For isolated paths, the noise-limited precision is far finer than the resolution cell. The real error sources are:

- paths that share an angle-delay cell: multi-bounce clusters and diffuse returns, and LoS plus ground bounce in the omni column only;
- model mismatch;
- gauge and pose errors.

### 2.2 What the data contains (facts that constrain the design)

1. **No beam squint.** The per-element phase ramp is applied at the carrier only, and path-gt-rich resynthesises `H = Σ_p a_baseband · e^{−j2π δf τ_p}` to about 1e-5. The PSFs are exactly separable: c(u,t) = α·AF(u − u_p)·D_N(t − τ_p). There is no single-view range from curvature.
2. **The BS pattern is not isotropic** (tr38901, V-pol, aimed at the target).
3. **The regime is sparse and specular.** The rich mock city (BS at (−70, 5, 25), 12 ring views at r = 40 m, UE height 1.5 m) has 57 valid paths in total.
   - **LoS and ground bounce.** Each ring view has both. The ground bounce arrives at about the negative of the LoS elevation, with 0.7–1.9 m of excess path. The two share a delay cell but are resolved in angle: Δu_z ≈ 0.44 at 110 m and ≈ 1.27 at 30 m, which is 1.8–5 angular cells (only marginal after Taylor broadening at the farthest views). They are unresolved only in the omni column.
   - **Building reflections.** 0–10 per view, at 7.8–134 m excess.
   - **Through-building paths.** Four views also have a path through a building, about 40 dB down (transmission contrast).
   - **Visibility source.** LoS visibility is read per (v, b) from the path GT (`los_visible`, T17) and is never assumed. Coverage-sampled bank poses may lack a LoS.
   - **Diffuse energy.** It appears only with `scattering_coefficient > 0` (S = 0.3 gives about 285k paths in the street canyon, 30–35 % change in the CFR).
4. **A specular virtual source is view-invariant in position** (`virtual_source_positions` agrees to about 1e-6 m). Its amplitude is not: it carries G_b along the mirrored departure direction and the Fresnel coefficient at the view's incidence angle.
5. **rf-gs-toy commits the inverse crime.** Its scatterers sit on voxel centres, and it uses 400 MHz with 16 bins, which aliases at 12 m. Its physics is kept; its numerics and protocol are not.
6. **Sionna GPU tracing is not bit-reproducible.** All experiments read frozen, hashed traces.

### 2.3 Information subsets and the two nested sublattices

"Removing" an information type means becoming invariant to it:

- **Remove P (carrier phase):** keep only power. No phase is exported, so cross-view coherence is gone.
- **Remove D (delay):** no inter-frequency coherence. Power is integrated over the band. Complex data is restricted to bin n0 (narrowband) or given an independent phase unknown per (capture, bin) (wideband, `_W`).
- **Remove I (amplitude):** normalise magnitudes: Y/|Y| per sample (PHAT), with samples below k·noise floor masked. Relative amplitudes remain visible through interference (the phase of a multipath sum depends on the ratios |α_i|).

| Node | Observable per capture c and hemisphere h | Shape per (c, h) | Classical analogue | Angle | Range | Amplitude | Carrier phase | Invariant to |
|---|---|---|---|---|---|---|---|---|
| **I** | I(u) = Σ_t \|c(u,t)\|² = Σ_n \|beam_n(u)\|² (Parseval) | [8,8] | intensity camera | ✓ | – | ✓ | – | φ, τ |
| **I@n0** | \|beam(Y[:,:,n0])\|² | [8,8] | narrowband intensity | ✓ | – | ✓ | – | φ, τ |
| **D** | up to M = 3 return delays per pixel from a fixed CFAR on windowed \|c\|², sub-bin interpolated. Variant T(u,t); variant D_PHAT (CFAR on the PHAT delay spectrum) | list | depth / ToF camera | ✓ | ✓ | (1-bit) | – | φ |
| **P** | Y[m,n0]/\|Y[m,n0]\| | [8,8] | phase-only interferometer | ✓ | – | relative (via interference) | ✓ | τ |
| **P_W** | Y/\|Y\| for all n, with an independent phase unknown per (c, n) | [8,8,N] | multi-frequency phase-only, incoherent across bins | ✓ | – | relative | within-bin only | φ, τ, inter-bin phase |
| **ID** | \|c(u,t)\|² | [8,8,N] | transient imaging | ✓ | ✓ | ✓ | – | φ |
| **IP** | Y[m,n0] | [8,8] | narrowband holography | ✓ | – | ✓ | ✓ | τ |
| **IP_W** | Y for all n, with an independent phase unknown per (c, n) | [8,8,N] | per-bin holography, incoherent across bins | ✓ | – | ✓ | within-bin only | φ, τ, inter-bin phase |
| **DP** | Y/\|Y\| for all n | [8,8,N] | SRP-PHAT | ✓ | ✓ | relative | ✓ | – |
| **IDP** | Y | [8,8,N] | coherent SAR-like imaging | ✓ | ✓ | ✓ | ✓ | – |

**Two nested sublattices** (Shapley values and synergies are computed on each separately, §5.3):

- **Narrowband NB** = {∅, I@n0, P, IP}. Every node is a function of IP.
- **Wideband WB** = {∅, I, D, P_W, ID, IP_W, DP, IDP}.
  - Data-nested edges: I ⊂ ID, I ⊂ IP_W (Parseval), P_W ⊂ IP_W, P_W ⊂ DP, D ⊂ ID, ID ⊂ IDP, IP_W ⊂ IDP, DP ⊂ IDP.
  - The one exception is **D → DP**, which is nested only by invariance. For that edge, D_PHAT (a function of DP) is reported as a sensitivity check.

I, I@n0, P_W and IP_W are sync-invariant, so their S and N values are identical by construction.

The partial-D variants **P×K-S** and **IP×K-S** use K = 4 spread bins with the phase shared across captures per bin. They leak differential delay in S mode, so they are reported only as variants, outside both lattices. Their N forms coincide with P_W/IP_W restricted to K bins.

**Omni column (`X-o`).** Aperture coherence is removed by an independent random phase per (capture, element), which makes every omni node a function of IDP-o:

- **IDP-o** = 64 per-element CFRs, with per-(c, m) phase unknown.
- **DP-o** = their unit-modulus version.
- **ID-o** = Σ_m |IFFT_n Y_m|², the element-incoherent PDP.
- **D-o** = CFAR returns from ID-o, built from both hemispheres. With M = 1 this is the "D0" floor row. It is *not* `view_dominant_delay`, which is front-only.
- **I-o** = RSS, the "I0" floor row.
- **IP-o and P-o** get a per-(c, m, n) phase, as in the `_W` construction, which makes them degenerate. IP-o reduces to the per-element magnitudes |Y_{m,n}|, and P-o carries nothing (it equals ∅). They are computed only to complete the lattice.

**`X-1el` variants** (outside the lattice): element (3,3) only, S-coherent across views. This is the "distributed single-antenna CSI" reading. Omni and 1el cells run the E1/E2 tiers only.

### 2.4 Synchronisation modes and the gauge model

`Y_obs[c] = e^{jφ_c} · diag_n(e^{−j2π δf_n τ_c}) · Y[c] + W`. This is the `gauge.align_common_phase_and_delay` convention: the carrier part of a timing error is absorbed into φ.

| Mode | Unknowns | Physical meaning |
|---|---|---|
| **S** (core) | none | all UEs and BSs share one LO and one timebase, including the absolute BS emission time; an upper bound |
| **N** (core) | (φ_c, τ_c) per capture, independent; φ ~ U[0, 2π), τ ~ N(0, σ_t) with σ_t ∈ {1, 10, 100} ns, or uniform over the period | coherence only inside one capture |
| S_τ (extra) | φ_c only | time-synced (GNSS-grade), phase-free |
| N-sep (extra) | φ_v + φ_b, τ_v + τ_b | one UE receiver hears all BSs in one clock epoch |
| Coherence groups (F5) | one gauge per group | a continuous path from S to N |

**Gauge fixing.**

- **The only true null space is the global phase**, because it is absorbed by the scene's complex amplitudes. A common timing shift is *not* a null space: t_b and p_v are absolute, which is the GNSS clock-bias argument.
- **Estimators** fix φ_{c0} = 0 and estimate every τ_c, including τ_{c0}.
- **The reference capture c0** is chosen by a fixed rule: the first training view (in `nested_view_order`) with b = 0. It never depends on the data.
- **Data generation:** `draw_gauges` sets the gauge of c0 to exactly (0, 0), so the reference convention holds in the data too, and the estimated τ_{c0} is a check.
- **M6** scores phase after removing the best global phase, and scores delay in absolute terms.

**Realism.** 3GPP TDD cell sync is ±1.5 µs, and GNSS time is 10–20 ns, so real multi-BS systems sit between N and N-sep. S also needs pose error below λ/10 ≈ 9 mm (pose-jitter sweep).

**Expected effect of N (also CI invariance checks):**

| Node | φ_c | τ_c | Expected S vs N |
|---|---|---|---|
| I, I@n0, P_W, IP_W | invariant | invariant | **S ≡ N to numerical precision** (a control, not a finding) |
| D, ID | invariant | circular shift along t | absolute bistatic range lost, relative delays kept. Restored by the LoS anchor or by pseudoranges when enough VS are shared (§4.2 rows 4, 8) |
| P, IP | rotates | no effect (DC bin) | cross-view coherence lost, so geometry ≈ I@n0. The LoS anchor restores φ_c up to the LoS phase-model error |
| DP, IDP | enters | enters | the coherent synthetic aperture is lost unless self-calibration, the LoS anchor or pseudoranges succeed |

### 2.5 Paired data generation (common random numbers)

1. **Trace once** per (scene, mechanism profile, pose bank, BS set). Freeze Y_clean [V, B, 2, 8, 8, N] and the path GT, and store their sha256 hashes.
2. **Apply the hardware effects shared by S and N**, with seeds from `SeedSequence([dataset_seed, v, b, realization])`:
   - **Element gains:** default 0.5 dB / 5°, and 0 on the ideal track. They are corrected by **hardware pre-calibration**: a separate calibration capture of a known far-field source with its own noise draw, applied equally to all configurations and recorded as calibration GT (§4.3).
   - **Front/back collapse** with a finite F/B ratio, on the observed track only.
   - **AWGN with one absolute σ² per dataset** (C8). σ² = P_ref/10^{SNR/10}, where P_ref is the full-CFR power of the reference capture, summed over both hemispheres. The same σ² is used for every (v, b, h), and for both hemispheres on the ideal track. The achieved per-capture SNR and the scatter-referenced SNR go into the impairments GT.

   The result is Y_S.
3. **Compute Y_N** = e^{jφ_c} e^{−j2π δf τ_c} Y_S, with gauges from their own stream and c0 fixed at (0, 0). The gauge is a unit-modulus diagonal, so S and N share **the same noise realisation**. N-sep and S_τ are built the same way.
4. **Extract all observables** from the same Y with one pure function. Noise is never added per observable.

**Tracks:** `ideal-S`, `ideal-N`, `observed-S`, `observed-N`, `N-sep`, `S_τ`. The core tables use the **ideal** track (hemispheres separate, element errors zero, reference SNR 30 dB).

---

## 3. Common reconstruction space

### 3.1 Path-level model (exactly what Sionna produces)

`Y[c,h,m,n] = Σ_l α_{c,l} · 1_h(u_{c,l}) · e^{+j k_c u_{c,l}·q_m} · e^{−j2π δf_n τ_{c,l}}`

- α_{c,l} is the aperture-centre baseband coefficient. It contains e^{−j2π f_c τ}, the BS and Rx patterns, polarisation and the material coefficients.
- This is the equation path-gt-rich `synthesize_cfr` verifies, which gives an exact convention test on real traces (T05, T21).

### 3.2 Two scene spaces sharing one operator library

**BV: bistatic Born voxel space (physical scene).** A point x has complex reflectivity ρ (coherent) or power σ = E|ρ|² (incoherent):

- r1 = |x − t_b|, r2 = |x − p_v|, u = R_vᵀ(x − p_v)/r2, τ(x) = (r1 + r2)/c
- α(x; c) = ρ · G_b((x − t_b)/r1) · P_rx(u) · λ_c/((4π)^1.5 r1 r2) · e^{−j k_c (r1 + r2)}

**VS: virtual-source space (one map per BS).** A point s has amplitude β:

- r = |s − p_v|, d = (p_v − s)/r, u = R_vᵀ d, τ = r/c
- Departure direction for a first-order image: n = (s − t_b)/|s − t_b| and d_dep = d − 2(d·n)n, the Householder reflection of d. For s = t_b (the LoS), d_dep = d.
- α(s; c) = β_{(·)} · G_b(d_dep) · P_rx(u) · λ_c/(4π r) · e^{−j k_c r}

Properties:

- **Sources.** The LoS is s = t_b. A single bounce off plane Π is mirror_Π(t_b), and higher orders are nested mirrors.
- **The amplitude model** β_{(·)} is one of three options:
  - **(a) view-independent β** with the G_b factor above. This is exact for a perfect first-order mirror and admits cross-view coherence.
  - **(a″) constrained per-view β_{k,v} = b_k(θ_inc,v) · e^{jψ_k}**: one shared phase ψ_k, and b_k real and smooth (quadratic in cos θ_inc). b_k may change sign once, because the V-pol ground reflection is TM and flips sign at Brewster. This absorbs Fresnel magnitude changes without absorbing the carrier phase.
  - **(b) free per-view β_{k,v}.** This absorbs anything, including the per-view carrier phase. **Under (b), S and S_τ are identical by construction.**
- **Higher-order VS.** The composed mirror is not determined by s, so the first-order G_b factor is wrong for them and their amplitudes are handled by (b).
- **Extent.** The VS grid extends below ground and behind walls, bounded by c·τ_max.

**Primary space.**

- VS is primary on specular Sionna scenes (L2–L4).
- BV is primary on L0 point phantoms, on diffuse profiles and for surface maps.
- Every configuration runs in both spaces where applicable.

### 3.3 Operators

- **Exact element-level operator** (reference, dense for small M). Implements §3.1–3.2 directly with `wavefront="plane"`, the carrier-only per-element phase (as Sionna does). It is used to generate L0 core data and as the reference.
  - `wavefront="spherical"` and `squint=True` exist only for the model-mismatch sweep.
  - The inverse crime is avoided by off-grid positions and a different discretisation, never by different physics.
- **Separable exact operator on point sets (E2/E3).** For points {x_p}, with at most 5e4 of them on CPU:
  - `Y_c = A_ang,c · diag(γ_c ⊙ x) · D_cᵀ`, where A_ang,c [64 × P] = e^{j k_c u_p·q_m}, D_c [N × P] = e^{−j2πδf_n τ_p}, and γ_c holds the pattern, spreading and carrier factors;
  - it is exact, bound by BLAS time, and has an exact adjoint;
  - the gauge enters as a diagonal on n.
- **Fast back-projection (E1, full grids).**
  - Per (c, h), build the Taylor-windowed complex angle-delay volume once by FFT, oversampled 8× in angle and delay.
  - Per voxel: `(A^H y)(x) ≈ conj(γ(x)) · c(u_c(x), τ(x) + τ_c mod T)`. γ already holds the carrier factor e^{−j2π f_c τ(x)}, so conj(γ) is the carrier compensation. The +τ_c sign follows from the §2.4 gauge (corrected in Phase 0; see the Phase 0 implementation notes).
  - Interpolation is trilinear (8 taps) by default and tricubic optionally, periodic in u (period 2 at d = λ/2) and in t (period N/B).
  - Per-capture sparse gather indices and weights are precomputed on the pruned grid.
  - Accuracy against exact: about −30 dB (trilinear) and −40 dB (tricubic). The trilinear error is below the AWGN of any atom weaker than the reference.
- **Incoherent kernels.** `E|c(u,t)|² = Σ_i σ_i |γ_i|² 1_h(u_i) |AF(u − u_i)|² |D_N(t − τ_i)|² + σ²`.
  - The kernel is separable and periodic in t.
  - On point sets, the forward is a separable real GEMM. On full grids, Kᵀy is computed as an FFT correlation of y with |AF|² ⊗ |D_N|², followed by the trilinear gather.
  - K_I = Σ_t K_ID, and K_D is the angle-integrated kernel (omni).
  - The model fails when paths share an angle-delay cell: overlapping multi-bounce clusters, diffuse patches, and LoS plus ground bounce in the omni column. This error is reported, and coherent configurations are the ones that can remove it.
- **Hemispheres.** Ideal data gives two independent images per capture. Collapsed data uses front + g·back, with each pixel back-projected onto both mirror rays using weights 1 and g (g² for power).
- **Nuisance atoms (C13).**
  - The LoS atom has its phase fixed by the model, with only a real g_c > 0 free.
  - The ground-bounce atom has a free complex amplitude.
  - Power configurations use nonnegative powers for both atoms.
  - In N mode, (φ_c, τ_c) are fitted jointly with the atoms. The LoS anchor determines τ_c, and determines φ_c up to the recorded LoS phase-model error ε_LoS,c = arg(a_LoS,GT / a_LoS,model) (T17).
- **Required tests:**
  - adjoint tests;
  - fast BP vs exact at −30 dB (trilinear) and −40 dB (tricubic) for off-grid atoms beyond 5 m;
  - the convention chain (T05);
  - VS atoms from path GT resynthesise `aperture_cfr` with gauge-aligned NMSE < 1e-3 on the rich mock (heavy CI).

### 3.4 Delay periodicity

The operator is periodic in t with period N/B. The tomography profile uses N = 128. GT paths longer than the period are flagged `beyond_period` and excluded from recall denominators.

### 3.5 Grids

| Grid | Extent (L2) | Spacing | Size | Used by |
|---|---|---|---|---|
| G_phys | [−50, 50]² × [−2, 40] m | 0.5 m | 3.4e6 | E1 BV maps |
| G_vs (per BS) | view bbox dilated by c·τ_max, capped at [−180, 180]² × [−80, 80] m, then pruned to the E1 support | 1 m | ≤ 2e6 after bbox pruning | E1 VS maps |
| Pruned support | each configuration's own E1 map: NMS, 2 m dilation, cap | as parent | ≤ 5e4 points (CPU), ≤ 1e6 (T33) | E2, MMV |
| ROI windows | 0.5 m cubes (up to 1 m) around the configuration's own detections | λ/4 = 2.1 cm | 1.3e4 (1.0e5 for 1 m) | coherent cross-view refinement |
| Gridless | atoms | – | K ≤ 100 | E3 |
| CI micro / smoke | 5×5×3 / L2 bbox | 2 m, with sub-voxel refinement | 75 / ~5e4 | CI |

A full-scene coherent grid at λ/4 would have about 1.5e10 voxels for about 1e5 data values, so it is not attempted. On any grid coarser than λ/4, coherent S maps are used only through their **per-capture envelope**, Σ_c |A_c^H y_c|² with the S-mode τ = 0. Cross-view carrier coherence is evaluated only in ROI windows or gridlessly.

**Common output of every configuration:**

- a density on G_phys and on G_vs per BS (coherent results as |ρ|², mean_v |ρ_v|² or mean_n |ρ_n|²);
- a detection list refined to sub-voxel precision (a quadratic fit on the 3×3×3 envelope neighbourhood of each peak);
- a Gaussian list;
- the estimated gauges (N mode).

### 3.6 Noise models and likelihoods

**Noise.** Raw AWGN is W ~ CN(0, σ²) with one σ² per dataset. When σ² is not taken from the impairments GT, it is estimated from the Taylor-windowed volume using only bins that are **both** evanescent in angle (u_y² + u_z² > 1) **and** beyond the longest geometric path in delay. N = 128 leaves more than 200 m of empty delay. The estimate is the median/ln 2, with the window noise-gain correction applied. T04 validates it against the GT σ².

**Likelihoods.** The data is a deterministic scene plus AWGN. **Exact** models (used for M8 CRBs, and optionally as solver losses) are therefore separated from **robust** solver losses (used by E2 and E3, and in the model-discrepancy branch):

| Observable | Exact model (M8; optional loss) | Robust solver loss |
|---|---|---|
| Y (IDP, IP; IP_W per bin with the per-(c,n) phase profiled) | complex Gaussian ‖Y − Ŷ‖²/σ²; N: profiled over the gauge (`gauge_aligned_nmse` form) | same |
| \|c\|² (ID); I@n0 per beam pixel | noncentral χ² (Rician power), 2 dof per unitary bin, noncentrality 2\|s\|²/σ² | exponential / Itakura–Saito, μ = Kσ + σ² |
| I | noncentral χ², 2N dof per pixel (N independent unitary bins) | Gamma with L_eff looks |
| P, P_W, DP | projected normal of s + w. The phase information is F_ψ(γ), which tends to 2γ; the SNR information is F_γ(γ); γ = \|s\|²/σ²; there is no cross term, by symmetry | von Mises, κ = 2\|s\|²/σ², with \|s\| as a nuisance magnitude. PHAT BP is its first-order linearisation |
| D lists | conditional on detection: Gaussian on (u_y, u_z, t) per return, with variance equal to the complex-Gaussian single-path CRB at the return's SNR | the same, plus a Bernoulli miss/false-alarm term at the fixed CFAR rate |
| T(u,t) variant | none (no likelihood) | cross-entropy (heuristic) |

**Model-discrepancy variance.** Each modality m gets σ²_eff,m = σ²_m + s²_m, where s²_m is estimated by type-II ML (s²_m ← L_m/n_m, alternating with the scene update). These weights are used by F4 and F7.

---

## 4. Algorithms per configuration

### 4.1 Solver tiers and compute budget

**E1 back-projection.**

- S coherent: the per-capture envelope on coarse grids, and |A^H y| in ROI windows.
- N coherent: Σ_c |A_c^H(τ̂_c) y_c|². **This is invariant to φ_c only.** Back-projection samples c(u, τ(x) + τ_c), so each capture needs τ̂_c from one of:
  - (a) the LoS anchor (N-LoS);
  - (b) the blind per-capture τ search that maximises cross-view consistency;
  - (c) delay-marginalised angle-only per-capture images (the fallback).
- Power: Kᵀy, with τ̂_c for the ID-N and D-N rows.
- Phase-only: PHAT back-projection.
- D: return-to-point splatting into an occupancy log-odds grid.
- I: log-domain (geometric-mean) fusion across views, which acts as a soft visual hull.

**E2 regularised inversion** on the configuration's own pruned support.

- Power: KL-EM, IS-MLEM, NN-FISTA ℓ1, optionally with TV.
- Coherent: Tikhonov LSQR/CGLS, complex-ℓ1 FISTA, and group lasso ℓ2,1 (MMV per view in VS space).

**E3 sparse / parametric.**

- CLEAN or OMP, then NOMP / LM on continuous parameters under the configuration's own likelihood, using the exact element-level operator on ≤ 100 atoms.
- Per-view smoothed ESPRIT, then VS triangulation, then the pseudorange solver.
- Output is exported as `gaussian_init`.

**Equal budgets:** 200 forward+adjoint pairs for E2, and 50 LM iterations for E3.

**Compute budget.** Targets are for an 8-thread CPU and are measured by the T07b/T07c benchmarks and recorded in M9. They cover L2, V = 16, B = 1, with 32 capture-hemispheres:

| Operation | Implementation | Size | CPU target |
|---|---|---|---|
| E1 BP on full G_phys | trilinear gather (T07a/b) | 3.4e6 × 32 × 8 ≈ 9e8 gathers | ≤ 60 s |
| E1 BP on G_vs (bbox-pruned) | same | ≤ 2e6 × 32 × 8 | ≤ 60 s |
| E2 coherent A or A^H | separable GEMM (T07c), exact | 32 × (64 × P)(P × N), P = 5e4 | ≤ 3 s |
| E2 power A or A^H | separable real GEMM (T08) | same | ≤ 0.3 s |
| E2 run (200 pairs) | – | – | ≤ 20 min coherent, ≤ 2 min power |
| MMV iterate | P × V complex128 | 5e4 × 16 = 13 MB | – |
| E3 | exact operator, ≤ 100 atoms | – | ≤ 5 min |
| Heavy smoke (all nodes) | 2 m grid, E2 capped at 10 iterations | – | ≤ 10 min |

Full-grid E2 and MMV on unpruned G_vs run **only** on the optional accelerated backend (T33). Tuning (§6.5) runs on L0 and on subsampled L2 (V = 8, P ≤ 2e4).

### 4.2 Configuration table

Budget classes: **Pw** = E1 ≤ 1 min plus E2 ≤ 2 min. **Co** = E1 ≤ 1 min plus E2 ≤ 20 min, plus ROI windows. **×A** = multiplied by the alternation iterations or restarts.

| # | Config | Methods (E1 / E2 / E3) | Gauge handling | Budget | Expected identifiability |
|---|---|---|---|---|---|
| 1 | I-S | cone BP + log-mean fusion / KL-EM or Gamma-MLEM + ℓ1 / CLEAN | none | Pw | Bearings per view; 3D position by ray intersection (V ≥ 2), with vertical precision ≈ r·Δu_z on a single-height ring. σ only up to the G_b and path-loss scale |
| 2 | I-N | identical code path | invariant | Pw | **must equal I-S** (control) |
| – | I@n0 | as I, on bin n0 | invariant | Pw | narrowband NB-lattice node |
| 3 | D-S | return→point splat (BV: single-bounce range on ray; VS: p + cτ·R_v u) / occupancy fit / clustering + Gauss–Newton on ranges | none | Pw | Single-view 3D points for resolved returns; no reflectivity; association ghosts |
| 4 | D-N | as D-S, with τ_c from (a) the LoS anchor, (b) min-entropy cross-view search, (c) joint alternation, or (d) the **T24c pseudorange solver** (per-capture bias plus VS triangulation) | τ_c | Pw ×A | LoS-anchored ≈ D-S. Blind: identifiable only when enough VS are shared across views (3 measurements per return against 3K + V unknowns). `ill_posed` is computed (§6.5) |
| 5 | P-S | per-capture PHAT envelope (coarse) → PHAT coherent BP in ROI / projected-normal IRLS / CLEAN | none | Co | Support, plus λ-scale fringes inside the envelope (integer ambiguity) for view-consistent point scatterers |
| 6 | P-N | per-capture phase-only beam images summed incoherently / bearing triangulation; N-LoS: φ_c from the LoS anchor, then as P-S | φ_c (τ none) | Pw | Blind ≈ I@n0 geometry; LoS-anchored approaches P-S up to ε_LoS |
| – | P_W | per-bin PHAT beams, each with its own phase, compounded | invariant | Pw | WB-lattice node; more looks than P, no delay |
| 7 | ID-S | angle-delay BP (BV, VS) / IS-MLEM + ℓ1/TV / CFAR → VS and single-bounce points → power-weighted clustering | none | Pw | Single-view 3D per resolved return, plus σ. **Strongest phase-free baseline** |
| 8 | ID-N | as ID-S, with τ_c from LoS+ground anchoring (resolved in angle), power cross-correlation alternation, or the **T24c pseudoranges** | τ_c | Pw ×A | LoS-anchored ≈ ID-S; blind: as D-N, but better conditioned |
| 9 | IP-S | envelope → coherent BP in ROI / Tikhonov LSQR and density-compensated diffraction tomography on ROI / OMP → NOMP | none | Co | Complex ρ at λ scale for view-consistent scatterers. Localisation along the bisector comes from view/BS diversity |
| 10 | IP-N | smoothed 2D ESPRIT per capture, triangulated / self-cal (ρ, φ_c = arg⟨A_c ρ, y_c⟩) / VarPro / LoS anchor | φ_c | Co ×A | Bearings and \|ρ\|; cross-view coherence if self-cal or the anchor succeeds |
| – | IP_W | per-bin coherent processing, each bin with its own phase, compounded | invariant | Co | WB-lattice node |
| 11 | DP-S | wideband SRP-PHAT envelope → ROI / projected-normal IRLS / CLEAN, NOMP | none | Co | 3D plus λ-scale detail; geometry and relative amplitude only |
| 12 | DP-N | per-capture SRP-PHAT with τ̂_c, incoherent fusion / self-cal with `gauge.py` / LoS anchor / pseudoranges | φ_c, τ_c | Co ×A | Degrades to whitened ID-N if self-cal fails |
| 13 | IDP-S | envelope → ROI coherent / LSQR, complex-ℓ1 on the pruned support, VS-MMV with β option (a), (a″) or (b) / NOMP, 3D ESPRIT + VS triangulation | none | Co | Everything identifiable; **upper bound** |
| 14 | IDP-N | (i) VarPro with `align_common_phase_and_delay` as the inner solver; (ii) LoS+ground anchoring, then the S pipeline; (iii) self-cal cascade from ID-N; (iv) pseudoranges (T24c); (v) per-view MMV | φ_c, τ_c | Co ×A (5–10×) | LoS-anchored ≈ IDP-S up to ε_LoS. Blind VS positions identifiable when #obs ≥ 3K + V − 1 (the global phase) |
| – | `X-o`, `X-1el` | E1/E2 of the matching node with angle removed (ellipsoid BP, bistatic multilateration, RSS field fit) | as camera | Pw | Needs many captures; ID-o needs ≥ 3–4 non-degenerate (v, b) per point; z-mirror ambiguity for coplanar UEs |
| – | P×K-S, IP×K-S | per-bin, phase shared across captures | none | Co | partial-D variant, outside the lattices |
| – | S_τ, N-sep | the N machinery with fewer unknowns | φ only / separable | as N | timing-only sync; multi-BS clock structure |
| – | I-T (optional) | Wilson–Patwari ellipse weights, Tikhonov | per-view gain | s | building occupancy; needs hundreds of links |

### 4.3 Notes that apply across configurations

- **N-mode MMV.**
  - Per-view complex maps absorb φ_c exactly, and ℓ2,1 couples the views through shared support.
  - The per-view maps do not absorb τ_c, so τ_c still comes from anchoring, alternation or pseudoranges.
  - With β option (b), S ≡ S_τ holds by construction (§3.2).
- **Self-calibration identifiability.** It needs a shared ρ and more than one well-separated atom per capture. Each run records its restart count and the spread across restarts.
- **Coherent pose fragility.** Pose error must be below about 9 mm (1 cm ≈ 42° of phase). The pose-jitter sweep applies to every coherent S configuration.
- **Element-gain calibration** is a **hardware pre-calibration** from a separate calibration capture (a known far-field source), applied identically to every configuration and recorded as calibration GT. No configuration calibrates from its own scene data. This respects fairness rule 5 and works in NLoS captures.
- **Single-bounce inversion:** `r2 = (L² − |t_b − p_v|²) / (2(L − (R_v u)·(t_b − p_v)))`, with L = c·t.
- **Support pruning** uses only the configuration's own E1 map (fairness rule 5).

### 4.4 Correspondence with RF-GS (#8, #9)

| Config | RF-GS renderer mode | RF-GS loss |
|---|---|---|
| I, I@n0 | incoherent splat into angular pixels with \|AF\|² | Gamma / IS on I (Rician for exact) |
| D | first-return / delay rendering | range residual |
| ID | transient angle × delay | IS on \|c\|² |
| P, IP, P_W, IP_W | coherent narrowband render per bin | phase-only / per-bin profiled complex |
| DP | coherent wideband render | projected normal / cosine |
| IDP-S | coherent wideband Σ_g ρ_g G(x_g) | complex L2 |
| IDP-N | same, plus per-view gauge | `gauge_aligned_nmse` |

A **frozen-Gaussian consistency test** (in #8) checks that RF-GS, with Gaussians frozen at voxel centres and the same loss, reproduces the baseline's σ̂ or ρ̂. RF-GS is compared against the baseline with the same node and sync mode, under the prediction rules of §6.4 M4.

---

## 5. Hybrid fusion strategies and the ablation design

The multi-type nodes (ID, IP, IP_W, DP, IDP) define *what* is available. F0–F8 define *how* it is combined. Every hybrid applies to any node that contains the modalities it uses.

### 5.1 Two fusion regimes

- **R1: every modality derived from the same Y.**
  - FIM(statistic) ≤ FIM(Y), and multiplying likelihoods double counts noise.
  - **Rule:** the final estimate uses only the likelihood of the finest statistic.
  - Coarser statistics enter only as initialisation, as annealed continuation terms, or as robustness terms weighted by 1/σ²_eff.
- **R2: heterogeneous observations.** Different captures deliver different modalities, and the noise is independent across captures, so `L = Σ_m Σ_{c∈C_m} L_m(c)/σ²_eff,m` is correct.

### 5.2 Fusion strategies

| Id | Strategy | Definition | Role |
|---|---|---|---|
| F0 | Late pooling | Per-modality E1 maps normalised to pseudo-posteriors. Log-linear (AND) and linear (OR) pooling, with weights on the simplex chosen by validation | zero-optimisation reference |
| F1 | Stacked marginals | joint objective over the marginal observables (for ID: I and D), weights 1/ν_m | "joint − stack" |
| F2 | Joint observable | the lattice node itself | core row |
| F3 | Cascade with self-calibration (**main hybrid**) | A: ID IS-MLEM on a 1 m grid → support. B: gauges (LoS+ground anchor, blind envelope alignment or pseudoranges). C: coherent sparse LS on the support (shared ρ in S; MMV ρ_v in N, or β option (a″)). D: gridless NOMP with continuation (envelope → within-view phase differences → full carrier), with a λ/4 local search in a ±λ ball. E: finest-likelihood polish, then s_m and `gaussian_init` | expected overall winner. Subset versions: ID = A only; DP = SRP-PHAT coarse, then C/D; IP = per-bin C/D |
| F4 | Robust multi-likelihood MAP | θ* = argmin Σ_m L_m/σ²_eff,m + R(θ), with ρ = √σ e^{jψ} and σ²_eff by type-II ML | a power term gives robustness, the coherent term sharpens |
| F5 | Partially coherent (coherence groups) | I(x) = Σ_G \|Σ_{(c,h,f)∈G} …\|². Groups share phase; **each group's timing is τ̂_G** (anchor or blind search) unless the group shares a timebase. Groups: per capture (the N limit), per BS, per view, sub-band, all (the S limit) | "value of coherence" curve |
| F6 | Dual-space VS → plane → surface, plus Born residual; cross-BS facet fusion | VS s_k → the bisector plane of t_b and s_k → specular points → facet extent; residual energy goes to BV; planes from different BSs vote | classical route to physical geometry; supplies BV surface and held-out-BS predictions |
| F7 | Heterogeneous R2 fusion | per-capture modality masks and summaries; the joint likelihood of §5.1 | mixed-modality observation mixes |
| F8 | Optical-geometry prior (**oracle**) | #11 depth/mesh as a support or visibility prior | "geometry known" upper bound |

### 5.3 Ablation design

- **Common random numbers.** One Y per (scene, seed, impairment setting), so all comparisons are paired.
- **Lattice analysis**, per metric and sync mode, **separately on NB and WB** (§2.3):
  - NB: marginal contributions along 4 edges, Shapley values of I and P over 2 orderings, and the I×P interaction.
  - WB: 12 edges, Shapley values of I, D and P over 6 orderings, and the pairwise interactions. The D → DP edge also uses D_PHAT.
  - Synergy Δ_syn = M(node) − max over proper sub-nodes.
  - Joint − stack (F2 − F1), and hybrid − best member.
  - The omni lattice analysed the same way.
  - Predictions: I×D strongly synergistic in omni; D×P redundant at coarse scale.
  - The same analysis on M8 CRBs, as the algorithm-free prediction.
- **Sync value.** Δ_sync = M(S) − M(N) per node and N strategy, against S_τ and N-sep, and along the F5 curve. Sync-invariant nodes are reported as controls.
- **Factor sweeps.** One factor at a time around the default (V = 16, B = 1, 8×8, N = 128, 100 MHz, reference SNR 30 dB, ideal track), plus targeted two-factor grids.

| Factor | Levels | Source |
|---|---|---|
| Views V | 1, 2, 4, 8, 16, 32 (nested prefixes); multi-height on/off | pose bank + `nested_view_order` |
| BS B | 1, 2, 4 (nested); N vs N-sep | multi-bs |
| Aperture | 8×8, 4×4, checkerboard, random 50 % | `element_mask` |
| Bandwidth | 20, 100, 400 MHz (N raised so N/B ≥ 1.5 × max path); sub-bands of 16 and 4 bins | `select_subband`, profile |
| Reference SNR (absolute σ²) | ∞, 30, 20, 10, 0 dB | §2.5 |
| Timing std σ_t | 1, 10, 100 ns, uniform over period | §2.4 |
| Impairments | F/B 10/20 dB; element-gain error after pre-calibration; per-view AGC | observed-impairments |
| Pose jitter | 0, 1 mm, 1 cm, 5 cm | §2.5 |
| Wavefront mismatch (L0) | plane (default), spherical, spherical + squint | T03 |
| VS amplitude model | (a), (a″), (b) | §3.2 |
| Mechanism | specular, +refraction, +diffraction, +diffuse S = 0.3 | solver-profiles |
| Mixes (R2) | {2 full + 6 power-only}, {1 full + 7 D0}, {4 ID + 12 I-o} | partial-observations |
| Geometry prior | off / on (F8) | #11 |

### 5.4 Hypotheses the suite must confirm or refute

1. I-S ≡ I-N, and likewise for I@n0, P_W and IP_W. This is a sanity check.
2. Geometry ranking on specular Sionna data: IDP-S ≥ IDP-N(LoS) ≳ ID-S ≈ DP-S > ID-N(LoS) > D-S > IP-S > I ≈ P-N ≈ IP-N > P-S (extended scenes) > D-N (blind, few shared VS).
3. Under VS options (a) or (a″), **phase sync adds little and only timing sync matters** (S_τ ≈ S), except for isotropic, view-consistent scatterers (edges, poles, diffuse patches, L0). Under option (b), S ≡ S_τ by construction; this is reported as a check, not as evidence.
4. P-N and IP-N collapse to about I@n0 geometry.
5. With a visible LoS, N-LoS ≈ S, up to the LoS phase-model error for phase-bearing nodes. Blind D-N and ID-N are identifiable only when enough VS are shared across views.
6. F3 beats every single-likelihood solver on IDP, and joint observables beat their stacked marginals (ID > ID-stack).

---

## 6. Evaluation protocol

### 6.1 Ground truth: `tomography_gt.npz`

One derived file per dataset. It is CPU-only, deterministic given the frozen path GT and mesh, and recorded in the manifest as `tomography_gt: {artifact, source_path_gt_sha256, source_mesh_sha256}`.

| Key | Shape | Derivation |
|---|---|---|
| `vs_pos`, `vs_bs`, `vs_order`, `vs_plane_ids` | [M,3], [M]… | specular-only valid paths, `virtual_source_positions`, clustered by (b, primitive sequence) |
| `vs_spread` | [M] | must be below 1e-3 m |
| `vs_visibility`, `vs_power` | [M,V] | path exists for (v, b); Σ\|a\|² |
| `vs_rho_eff` | [M,V] complex | a / (λ/(4π cτ) · G_b(d_dep) · P_rx(u)), with d_dep from the path-GT departure angles. Matches the operator of §3.2 |
| `vs_theta_inc` | [M,V] | incidence angle (first order), for option (a″) and interpolation |
| `los_phase_model_error`, `los_amp_model_error_db` | [V,B] | ε_LoS: path-GT LoS a_baseband against the C13 LoS model |
| `interaction_points` (+ v, b, path, depth, type, object) | [Q,3] | path-gt-rich `vertices` |
| `surface_samples`, `surface_normals`, `surface_object` | [S,3]… | Poisson-disk samples at 0.25 m on the mesh, ground included |
| `surface_observable`, `surface_specular_support` | [S] bool | raycast-visible from ≥ 1 BS and ≥ 1 UE hemisphere; within 0.5 m of an interaction point |
| `los_visible`, `ground_bounce_visible` | [V,B] bool | stratification |
| `path_type`, `beyond_period` | per path | mechanism labels from `interactions`; delay-wrap flag |
| `points_pos`, `points_rho` | [K,3], [K] | analytic scenes only |

On Sionna scenes, reflectivity maps are scored as **consistency** against PSF-blurred splats of `vs_power`, not as physical error.

### 6.2 Scene ladder

| Level | Scene | Purpose |
|---|---|---|
| L0a | 1 isotropic point, free space, off-grid; plane-wave data | PSF, CRB check |
| L0b | 2 points, separation 0.1–10 m along range, cross-range and vertical | two-point resolution |
| L0c | K ∈ {4, 16, 64} random points in a 10 m box, \|ρ\| log-uniform over 30 dB | detection, NMSE |
| L0d | 4 m × 4 m plate at λ/4 sampling | Born-on-specular failure, VS vs BV |
| L0e | analytic image-method phantom with ITU Fresnel coefficients (TM ground with Brewster, TE walls) | exact VS GT with realistic anisotropy |
| L0f | oracle resynthesis of the rich mock from path GT | convention check |
| L0-mm | L0a/L0c regenerated with spherical wavefront, squint and element jitter | model-mismatch sweep; NMSE floor of the plane-wave operator reported (expected about −25 to −35 dB at 20–40 m) |
| L1 | master mock box | pipeline smoke only, **not scored** |
| L2 | rich mock city + ground (tomography profile) | **core table** |
| L3 | Sionna `simple_street_canyon` | dense specular canyon |
| L4 | one PLATEAU LoD2 tile, ~200 m ROI | realism; VS, surface and held-out metrics only |

The mechanism axis (L2, L3) runs specular → +refraction → +diffraction → +diffuse. Diffuse runs are scored in the CFR domain only.

### 6.3 Views, BSs and splits

- **Pose bank:** 64 poses per scene.
  - Rings from `generate_ring_views` at radii 20/30/40 m, plus coverage sampling.
  - Heights of 1.5 m, plus 10 and 25 m if Q2 approves.
  - Look-at is the ROI centre with ±15° jitter.
  - Placement is recorded as `views[].placement`.
- **Nested subsets:** `nested_view_order(bank, seed)` gives one fixed permutation per seed, and V ∈ {1, …, 32} takes its prefixes. There are 5 seeds for coverage placement and 1 for rings.
- **BSs:** B ∈ {1, 2, 4}, nested.
- **Held out:** 25 % of the bank (at least 4 views), plus one BS when B ≥ 2. The split is stored in the manifest and shared with RF-GS.

### 6.4 Metrics

| Id | Metric | Details |
|---|---|---|
| M1 | VS detection and localisation | Sub-voxel-refined detections. Hungarian matching at gates {0.5, 1, 2, 4} m against detectable GT (within 30 dB of the capture's strongest path in ≥ 1 training capture). Precision/recall (plain and power-weighted), FROC, AP@1 m, recall at 1 FA/1000 m³. Error median/P90 split into range, horizontal and vertical |
| M2 | Physical geometry | precision/recall/F@d (d ∈ {0.5, 1, 2} m) over `surface_observable`, weighted Chamfer, energy-within-r |
| M3 | Planes (F6) | normal angle error and offset per matched wall |
| M4 | **Held-out cross-modal prediction matrix** (shared with RF-GS) | Rows = configuration; columns = predicted observable (I, D, ID, IDP). Metrics: power NMSE and dB log-spectral distance on ID; circular EMD for D; `gauge_aligned_nmse` for IDP. **Prediction rules:** (1) Held-out BS: only BV and F6 physical-space outputs are scored; VS outputs are N/A. (2) Per-view amplitudes (MMV, option (b), F3 stage C in N mode) predict an unseen view with a declared interpolator: the default is nearest training view in bistatic angle; the alternative is fitted b_k(θ_inc) with circular-mean phase. The interpolator is recorded per row. (3) For views not captured simultaneously, the gauge alignment is kept (profiled phase and delay for complex; delay-shift alignment for power). (4) RF-GS uses its own renderer under the same rules (1) and (3). Estimates that are only power are N/A in the IDP column |
| M5 | Reconstruction NMSE (L0) | complex NMSE with best global phase; power NMSE with best global scale; against sharp and PSF-blurred GT |
| M6 | Gauge recovery (N) | wrapped phase error after removing the best global phase; `circular_delay_error_s` absolute; ε_LoS reported next to them |
| M7 | Resolution | L0a probes on a 5×5×3 lattice: −3 dB widths, PSLR, ISLR; L0b resolvable separation; resolution maps |
| M8 | Information bound | CRB from the exact models of §3.6 (T30), with gauges as nuisance (Schur complement) and the global-phase null removed. Operator singular values on the ROI. RMSE/CRB. Nodes without a likelihood (T(u,t), pooling) are N/A |
| M9 | Cost | wall time, peak RSS, A/A^H counts, iterations, restarts, hardware. Reported, not ranked |

All results are stratified by `los_visible` (where LoS-free captures exist) and by path type.

### 6.5 Fairness protocol

1. **Same capture:** identical Y per realisation for all configurations and fusions.
2. **Same geometry:** poses, BS subsets, ROI, grid specifications, windows, nuisance-atom treatment and NMS procedure.
3. **Tuning:** hyperparameters are tuned per configuration on a validation set (L0c seeds 100–119 and L2 validation poses, subsampled to V = 8), 20 log-spaced trials each, scored by held-out loss in the configuration's own observable. The values are then frozen for S and N. Oracle-tuned numbers go in a separate column only.
4. **Detection thresholds** are swept (FROC), never tuned.
5. **No statistic outside the subset**, including for ROI selection, pruning and thresholds. Documented exceptions: D's one-bit CFAR, and the shared hardware pre-calibration.
6. **Statistics:** R = 20 realisations on L0 and R = 5 on L2–L4, with 95 % bootstrap CIs and Wilcoxon signed-rank tests.
7. **`ill_posed` is computed, not declared.** It is true when the gauge-reduced FIM at the estimate (T15b; exact form after T30) has condition number > 1e8, or any position CRB std > 10 m. Every configuration is run and reported either way.

### 6.6 Reporting

- **Main table per scene level:**
  - rows: the 14 core configurations, then the lattice-completion nodes, stack/fusion rows, omni and 1el rows, and oracle rows in a separate block;
  - columns: localisation median/P90, AP@1 m, recall at 1 FA/1000 m³, F@1 m, NMSE (L0), held-out ID NMSE, held-out IDP gauge-aligned NMSE (with the interpolator named), PSF widths, PSLR, CRB ratio, `ill_posed`, runtime;
  - N rows appear as blind and LoS-anchored.
- **Figures:**
  - NB and WB Hasse plots, S and N side by side, with Shapley and interaction tables;
  - S-vs-N slope charts, and the F5 curve;
  - factor-sweep curves;
  - FROC, PSF slices, CRB vs achieved;
  - the cross-modal heatmap;
  - optical and mesh overlays.
- **Artifacts:** `report.md`, `results.jsonl`, `recon/<scene>/<config>/<solver>/<realisation>.npz` (map, grid spec, detections, gauges [V, B, 2], Gaussians, `space`, `bs_index`) and `run_manifest.json`.

### 6.7 CI

- **Unit tests** (CPU, `unit-tests.yml`, under 60 s):
  - adjoint tests: interpolation primitives, the separable operator, incoherent kernels;
  - fast BP vs exact (−30 dB trilinear);
  - the convention chain;
  - `extract` invariances:
    - the sync-invariant nodes (I, I@n0, P_W, IP_W) are equal in S and N;
    - D_N is the circular shift of D_S;
    - P and IP are τ-invariant at the DC bin;
    - power observables are φ-invariant;
  - lattice derivability: I from IP_W equals I from ID (Parseval); ID-o and I-o from IDP-o;
  - **N-mode E1 test:** with random τ_c ≠ 0, the N map must differ from the S map unless τ̂_c = τ_c is applied, and then it must match S to 1e-10;
  - gauge reference: c0 has (0, 0) exactly, and σ² is identical for every (v, b, h);
  - `gt.py` on a synthetic mirror plane;
  - metrics on handmade cases;
  - the `ill_posed` test: a one-view D-N case is flagged; eight views with three shared VS are not;
  - **L0 micro-scene:** 5×5×3 grid at 2 m, 4 views, 4×4 aperture, 16 bins, 2 points at controlled offsets ≤ 0.2 m from voxel centres.
    - Every node runs E1 and E2 and returns finite outputs.
    - After sub-voxel refinement, ID-S and IDP-S localise both points within 0.6·spacing (1.2 m).
    - IDP-S with one 0.5 m ROI window at λ/4 localises within 0.25 m.
    - D-N blind: its `ill_posed` equals the computed test.
- **Heavy CI smoke** (a step in `scripts/ci/run-heavy.sh` on milestone dev-branch pushes). It runs L1 until rich-mock-scene is merged, then L2 with 8 ring views × 2 BS, a 2 m grid with sub-voxel refinement, a 32-bin sub-band, all nodes, E1 + E2 capped at 10 iterations, E3 off, R = 1, in ≤ 10 min. `scripts/ci/check_tomography_smoke.py` asserts:
  - the output schemas, and that all values are finite;
  - **on the raw E1 map before any nuisance fitting**, the LoS (t_b) is found within 0.6·√3·spacing (2.1 m) by I, ID and IDP in S;
  - the fitted LoS magnitude matches the path-GT LoS within 1 dB;
  - the ground-mirror VS of bs_000, and at least one building VS, are found within 2.1 m by ID-S and IDP-S (VS space);
  - I_S == I_N;
  - N-LoS gauge recovery at 30 dB: delay error < 1 ns, and phase error < 10° + |ε_LoS,c| per capture;
  - the report is uploaded as an artifact.

  Tolerances allow for GPU non-determinism.
- **Full suite:** manual or nightly, never gating (Q5).

---

## 7. Data and format requirements; dependencies on the M1 branches

### 7.1 Branch dependencies

| Branch | What the suite uses | Required change or addition |
|---|---|---|
| multi-bs | manifest v3, `aperture_cfr[bs, h, r, c, f]`, `views[].bs[]` | none; v2 read as B = 1 |
| path-gt-rich | `a`, `a_baseband`, `tau`, angles, `interactions`, `vertices`, `object_index`, `synthesize_cfr`; stored as [rx, tx, pattern, row, col, path] | none (the tx axis exists). Mechanism labels are derived from `interactions` in T17 |
| image-sources | `virtual_source_positions`, `mirror_point` | none |
| gauge-alignment | `align_common_phase_and_delay`, `gauge_aligned_nmse`, `nmse` | none |
| observed-impairments | `ImpairmentConfig`, `apply_impairments`, `timing_phase_ramp` | a `bs` axis; **an absolute-σ² option** (the current code sets σ² from each call's own signal power); a hardware-only entry point; `apply_gauge`; per-(v, b) gauge GT with a fixed reference capture; N-sep and S_τ; pose perturbation with GT; calibration-capture generation |
| partial-observations | `element_mask`, `apply_element_mask`, `select_subband`, `element_power` | none. `select_views` is not used for nesting (its draws are not nested and it takes a fraction); `nested_view_order` lives in `rf_tomography`. D-o builds omni delay from both hemispheres instead of calling the front-only `view_dominant_delay` |
| rf-gs-toy | bistatic scatterer physics, `compare_direct_path` | physics re-implemented; the dense solver stays in tests |
| rich-mock-scene | L2 scene, mesh | none |
| solver-profiles | mechanism profiles, diffuse S = 0.3 | a named `tomography` profile set |
| feature/11-optical-reference | optical renders, depth, `project_local_points` | depth/mesh access for F8 and overlays |

### 7.2 New dataset requirements

1. **Tomography profile:**
   - the 64-pose bank (elevated heights if Q2 approves);
   - 3–4 BSs;
   - N = 128;
   - max_depth 3–5;
   - specular and diffuse variants;
   - an oracle `los=False` trace;
   - an optional dense UE grid for I-T.
2. **Frozen data:**
   - Y_clean with sha256 hashes;
   - per-(v, b) impairments GT (φ, τ, gains, pose perturbation);
   - **the dataset σ² and P_ref**, the identity of the reference capture, and the achieved full-CFR and scatter-referenced SNR per capture;
   - calibration-capture GT.
3. **BS pattern:**
   - `pattern`, orientation and polarisation in the manifest;
   - a NumPy tr38901 implementation validated against Sionna;
   - the Rx polarisation factor validated against direct-path amplitudes.
4. **`tomography_gt.npz`** (§6.1) and its manifest entry.
5. **Fixed splits and seeds** (train / validation / held-out views and BSs, and the `nested_view_order` seeds), shared with RF-GS.
6. **`views[].placement`** as an optional field, with no version bump.
7. **Both hemispheres** are developed from raw `aperture_cfr`, so no back-hemisphere image artifact is needed.

### 7.3 Code boundaries

- `src/plateau_rt/domain/rf_tomography/` is NumPy/SciPy-only and is added to `SIONNA_FREE_MODULES`.
- Mesh-dependent code lives in the application layer.
- The optional accelerated backend (torch or CuPy) sits behind the same API in an adapter module. It is never imported by the domain layer, and it is the only route for full-grid E2 and unpruned VS MMV.

---

## 8. Implementation plan

Each task touches one or two source files, has explicit I/O and numeric acceptance criteria, and comes with its own tests. `rt` = `src/plateau_rt/domain/rf_tomography`.

### 8.0 Conventions binding every task

- **Arrays:** float64 and complex128, SI units.
- **Data:** `Y[V, B, 2, 8, 8, N]`.
  - Axis 2: 0 = front, 1 = back.
  - Axes 3 and 4: (row r, column col), per `imaging.reshape_planar_column_first`.
  - Axis 5: bin n with δf_n = (n − N/2)·B/N.
- **Gauges:** `phi[V, B]` in rad and `tau[V, B]` in s. The reference c0 = (first entry of `nested_view_order`, b = 0).
- **Grid:** `VoxelGrid(origin[3], spacing, shape=(nx, ny, nz))`. The flat index is `np.ravel_multi_index((ix, iy, iz), shape)` (C order, z fastest). Maps are `[nx, ny, nz]`. Point sets are `[P, 3]`.
- **Angle-delay volume:** `[V, B, 2, Qy, Qz, Nt]`, fftshifted, with u ∈ [−1, 1) and period 2 at d = λ/2, and t ∈ [0, N/B).
- **Interpolation:** periodic trilinear (8 taps) by default; periodic tricubic (64 taps) optional.
- **Random numbers:** `np.random.Generator` only, seeded from `SeedSequence`.

### Phase 0: foundations (master only)

**T01 Geometry, grids and view order.**
- Files: `rt/__init__.py`, `rt/geometry.py`, `rt/views.py`, `tests/test_rf_tomography_geometry.py`.
- Implement:
  - `VoxelGrid` with `centers() -> [P,3]` and `index(points) -> int[P]`;
  - `CaptureGeometry(ue_pos[V,3], ue_rot[V,3,3], bs_pos[B,3], elem_offsets[64,3], freq_offsets[N], f_c)`;
  - `local_direction(x[P,3], v) -> [P,3]`, `bistatic_delay(x, v, b) -> [P]`, `vs_delay(s, v)`;
  - `vs_departure_dir(s[P,3], v, b) -> [P,3]` (Householder);
  - `single_bounce_point`, `mirror_point`;
  - `nested_view_order(n_bank, seed) -> int[n_bank]`.
- Tests:
  - delay symmetry;
  - single-bounce round trip within 1e-9 m;
  - mirror involution;
  - the Householder departure direction reproduces an analytic mirror ray within 1e-12;
  - prefixes of `nested_view_order` are nested and deterministic.
- Deps: none.

**T02 NumPy BS pattern.**
- Files: `rt/antenna.py`, tests.
- Implement `tr38901_gain(theta, phi)` and `bs_pattern(dir_world[P,3], bs_orientation[3,3], kind={"tr38901","iso"}) -> complex[P]`.
- Tests: 8 dBi at boresight; −3 dB at ±32.5°; −30 dB floor; iso = 1.
- Deps: none.

**T03 Exact reference operator.**
- Files: `rt/forward_exact.py`, tests.
- Implement:
  - `atom_cfr(points[P,3], amps[P] or [P,V,B], geom, space={"bv","vs"}, wavefront={"plane","spherical"}, squint=False, pattern="tr38901") -> Y[V,B,2,8,8,N]`;
  - `dense_matrix(points, geom, space) -> [V·B·2·64·N, P]`, guarded by P·rows ≤ 2e8.
- Tests:
  - a VS at t_b equals the analytic LoS;
  - hemisphere mask;
  - **the plane vs spherical phase difference is ≤ k·max|q|²/(2r) + 1e-12 at r ∈ {10, 40, 200} m, and < 1e-2 relative at r = 200 m**;
  - an atom at τ + T gives the same Y.
- Deps: T01, T02.

**T04 Observables.**
- Files: `rt/observables.py`, tests.
- Implement:
  - `angle_delay_volume(Y, window=None|"taylor", oversample=(1,1)|(8,8)) -> [V,B,2,Qy,Qz,Nt]`;
  - `extract(Y, name, params) -> Observable(data, mask, noise_var, meta)` for every node in §2.3, the omni and 1el nodes, the partial-D variants and T(u,t);
  - `cfar_returns(power_volume, pfa, max_returns=3)`;
  - `noise_var_estimate(Y, max_path_m) -> float`, from bins that are evanescent and beyond the maximum path, windowed, noise-gain corrected, median/ln 2.
- Tests:
  - Parseval (I from ID equals I from IP_W to 1e-12);
  - the invariances of §6.7;
  - the noise estimate is within 10 % of the GT σ² at 30 dB with a 69 dB LoS peak present.
- Deps: T01.

**T05 Convention tests.**
- Files: `tests/test_rf_tomography_conventions.py`.
- Check:
  - a T03 atom pushed through `imaging.aperture_to_angular_fft → calibration.calibrate_angular_cfr → delay.angular_cfr_to_delay` peaks within half a bin of the analytic (u_y, u_z, t);
  - T04 agrees with the master pipeline up to the documented flip and phase.
- Deps: T03, T04.

**T06 Sync and paired tracks.**
- Files: `rt/sync.py`, tests.
- Implement:
  - `draw_gauges(V, B, mode, sigma_t, rng, ref=(v0, 0)) -> (phi[V,B], tau[V,B])`, with the reference exactly (0, 0);
  - `apply_gauge(Y, phi, tau, freq_offsets)`;
  - `reference_power(Y_clean, los_visible) -> (p_ref, c_ref)`;
  - `make_tracks(Y_clean, snr_db, p_ref, seeds) -> ({name: Y}, gt)`, with one σ² for every (v, b, h), and the achieved and scatter-referenced SNR in `gt`.
- Tests:
  - the gauge is unit-modulus;
  - Y_N·conj(gauge) == Y_S;
  - N_sep structure;
  - reference gauge exactly 0;
  - identical σ² across captures and hemispheres.
- Deps: T01, T04.

**T07a Interpolation primitives.**
- Files: `rt/interp.py`, tests.
- Implement:
  - `periodic_weights(coords[P,3], shape[3], periods[3], kind={"trilinear","tricubic"}) -> (idx int64[P,T], w float64[P,T])`;
  - `gather(vol, idx, w) -> [P]`;
  - `scatter_add(vals[P], idx, w, shape) -> vol`.
- Tests:
  - ⟨gather(v), a⟩ = ⟨v, scatter_add(a)⟩ to 1e-12;
  - wrap at the period boundary;
  - on an 8×-oversampled Dirichlet kernel, relative error < 3e-2 (trilinear) and < 1e-3 (tricubic).
- Deps: none.

**T07b E1 fast back-projection.**
- Files: `rt/backproject.py`, tests.
- Implement:
  - `capture_lookup(geom, points, space, v, b, h) -> Lookup(idx, w, carrier[P], valid[P])`;
  - `backproject(Y, geom, points, space, tau=None, per_capture=True) -> complex[V,B,2,P]`, with carrier compensation and optional τ̂_c;
  - `envelope_sum(bp) -> [P]`.
- Tests:
  - agreement with exact `dense_matrix`ᴴ·Y below −30 dB (trilinear) on 10 off-grid atoms beyond 5 m;
  - the N-mode random-τ test of §6.7;
  - a benchmark recorded (not gating).
- Deps: T03, T04, T07a.

**T07c Separable exact operator.**
- Files: `rt/forward_sep.py`, tests.
- Implement `SeparableOperator(points[P,3], geom, space, beta_model={"shared","per_view","constrained"}, gauges=None)` with `matvec`, `rmatvec` and `as_linear_operator()`. Its layout: x is [P] (shared), [P,V,B] (per view) or constrained parameters; y is flattened `Y`.
- Tests:
  - adjoint to 1e-10;
  - equals T03 `atom_cfr` to 1e-12;
  - timing at P = 5e4 recorded against the §4.1 target.
- Deps: T03.

**T08 Incoherent kernels.**
- Files: `rt/kernels.py`, tests.
- Implement:
  - `power_operator(points, geom, space, product={"ID","I","I_n0","ID_omni","I_omni"}) -> LinearOperator` (separable real GEMM);
  - `power_backproject_grid(power_volume, geom, grid, tau=None)` (FFT correlation, then T07a gather).
- Tests:
  - adjoint;
  - Monte-Carlo E|c|² for separated atoms within 5 %;
  - nonnegativity.
- Deps: T07a, T07c.

**T09 Synthetic phantoms L0a–L0e, L0-mm.**
- Files: `rt/synthetic.py`, tests.
- Implement generators returning (Y_clean, geom, gt). Plane-wave by default; `mismatch={"spherical","squint","element_jitter"}` for L0-mm; `offset_max` for CI placement.
- Tests:
  - reproducible for a fixed seed;
  - points off-grid by at least 0.05·spacing and at most `offset_max`;
  - L0d sampled at λ/4;
  - L0-mm NMSE floor matches the analytic prediction within 3 dB.
- Deps: T03.

**T10 Core metrics.**
- Files: `rt/metrics.py`, tests.
- Implement:
  - `nms_peaks(map, grid, radius, refine=True)` (quadratic sub-voxel fit);
  - `match` (Hungarian), `froc`, `ap_at`, `loc_error_decomposed`;
  - `nmse_global_phase`, `nmse_power_scale`;
  - `gauge_errors(est, gt)` (global phase removed; delay absolute).
- Tests: handmade cases; refinement recovers a 0.3·spacing offset within 0.05·spacing on a Gaussian blob.
- Deps: T01.

#### Phase 0 implementation notes

Phase 0 is implemented on `explore/tomography-p0`. Each task is one module under `rt` (NumPy/SciPy only; `rt/__init__.py` stays docstring-only with no re-exports) plus `tests/test_rf_tomography_<module>.py`. T05 is tests only. Every number below was measured in the ci container on CPU unless stated otherwise. Phase 1 must treat the conventions in this subsection as binding.

| Task | Module | Main entry points |
|---|---|---|
| T01 | `geometry.py`, `views.py` | `VoxelGrid`, `CaptureGeometry` (+ `from_orientations`, `select`), `planar_element_offsets`, `mirror_point`, `single_bounce_point`, `hemisphere_index`, `nested_view_order` |
| T02 | `antenna.py` | `tr38901_gain` (linear power), `bs_pattern` (complex field), `bs_orientation` |
| T03 | `forward_exact.py` | `capture_factors`, `atom_cfr`, `dense_matrix` |
| T04 | `observables.py` | `angle_delay_volume`, `aperture_centre_phase`, `volume_axes`, `extract` (23 nodes), `cfar_returns`, `noise_var_estimate` |
| T06 | `sync.py` | `gauge_factor`, `apply_gauge`, `draw_gauges`, `capture_power`, `reference_power`, `make_tracks` |
| T07a | `interp.py` | `periodic_weights`, `gather`, `scatter_add` |
| T07b | `backproject.py` | `capture_volume`, `capture_lookup`, `apply_lookup`, `backproject`, `envelope_sum` |
| T07c | `forward_sep.py` | `SeparableOperator`, `incidence_cosine`, `project_shared_phase` |
| T08 | `kernels.py` | `PowerOperator`, `power_operator`, `power_backproject_grid`, `dirichlet_power`, `noise_floor` |
| T09 | `synthetic.py` | `l0a_point` … `l0e_image_method`, `l0_mm`, `generate`, `ring_geometry`, `plane_wave_floor`, `mismatch_floor_prediction` |
| T10 | `metrics.py` | `nms_peaks`, `match`, `froc`, `ap_at`, `recall_at_fa`, `loc_error_decomposed`, `nmse_global_phase`, `nmse_power_scale`, `gauge_errors` |

**Pinned conventions (checked against real Sionna 2.0.1).**

- **Element offsets.** q_m for m = r·C + col is (0, d(col − (C−1)/2), d((R−1)/2 − r)): row 0 is the top (+z), and m is the C-order flattening of `Y[..., r, col, :]`. Sionna's `PlanarArray` passed through `reshape_planar_column_first` matches this to 5.3e-9 m. The element phase is e^{+j k_c u·q_m} at the carrier only, with u the UE-local unit vector toward the source.
  - The fixture `tests/fixtures/rf_tomography/sionna_los_aperture.npz` is a CPU-traced LoS with rolled UEs, one front and one back. The model matches it with relative spread 3.6e-6 / 5.4e-6, and the other hemisphere is exactly 0.
  - The wrong variants give spreads of 0.66 to 18.
- **Frequency grid.** `CaptureGeometry.from_orientations` uses `imaging.frequency_offsets`, the float32 grid Sionna traces on, promoted to float64. For N = 128, `delay_period` is 1.279999974 µs rather than exactly 1280 ns. The DC bin N/2 is exactly 0 Hz. Operators use the exact per-bin `freq_offsets`; the analytic kernels use the uniform δf, which differs from them by about 2e-6 relative.
- **BS pattern and orientation (T02).**
  - G_b is the Sionna V-pol TR 38.901 element as a field amplitude √gain with phase 0. Boresight is local +x.
  - The BS orientation is `rotation_matrix(look_at_orientation(t_b, target))`, i.e. roll 0, exactly as `Transmitter(look_at=…)` sets it.
  - Measured against Sionna: element 1.4e-6, rotation 5.1e-8, and an end-to-end PathSolver LoS ratio of 1.0e-6.
- **Polarisation (T03).** `atom_cfr(..., polarization="vv")` adds the co-polar factor pol = (H W_b θ̂(W_bᵀ d_dep)) · (R_v θ̂(u)), where H is the Householder mirror for image sources and the identity for the LoS and BV.
  - It is required for float32 agreement with Sionna. On the split-pattern GPU mock, the max relative error drops from 8.3e-3 … 2.7e-2 to at most 4.8e-4. The residual is a pure common phase at or below k r ε_f32; after one complex scalar is fitted it is 1e-5.
  - The design's scalar model stays the default (`"none"`). T09 phantoms use `"none"`.
- **Hemisphere.** The front hemisphere is u_x ≥ 0 (u_x = 0 counts as front). An atom enters only its own hemisphere, and the other is exactly 0 in every operator.
- **Angle-delay volume (T04).** The axis order is `[V, B, H, Qy, Qz, Nt]` = (u_y, u_z, t). u = fftshift(fftfreq(Q))/s and t = it/(Nt δf). The scale is always 1/√(RCN), so the unwindowed (1,1) volume is unitary.
  - **Phase reference.** The volume uses element (0,0), the index origin, not the aperture centre. With centred offsets an even aperture is antiperiodic in u (AF(u+2) = −AF(u)). The index-origin volume is exactly periodic, with period 2 in u and T = 1/δf in t. The centred field of §2.1 is `vol · aperture_centre_phase(u_y, u_z)`.
  - **Relation to the master pipeline (T05).** The chain `aperture_to_angular_fft → calibrate_angular_cfr → angular_cfr_to_delay` gives the centred c(u,t) of §2.1 divided by N. The two agree as `centred[:, iz] = √(N/(RC)) · cir[iz−1].T` for iz ≥ 1, with `ky == u_y` and `kz[i] == u_z[i+1]`. The u_z = −1 edge row is (−1)^{R−1} times the master kz = +1 row. The worst deviation is 2.4e-15.
  - **Peaks.** Atoms peak within half a cell of the analytic (u_y, u_z, τ mod T) in both chains; the worst case is 0.4991 cells over 156 cases. Back-hemisphere images are not mirrored.
- **E1 integer-element shift (T07b).** Interpolating the index-origin volume and applying the centre phase after the gather reaches only −30.7 dB for trilinear with Taylor. The cause is the 3.5-element phase ramp, which rotates 0.34 rad per oversampled sample.
  - `capture_volume` therefore multiplies the volume by the exactly periodic integer shift e^{−j2πs(−K_c u_y + K_r u_z)}, with K = (L−1)//2. Only the residual half-element phase is applied per point, inside `Lookup.carrier` = √(RCN) · conj(γ) · residual.
  - Measured against dense^H(W·Y): trilinear −42.4 dB (vs) and −42.2 dB (bv); tricubic −76 dB.
- **Gauge sign (T06, T07b, T07c, T08).** Every module uses Y_obs = e^{jφ_c} e^{−j2π δf_n τ_c} Y (`sync.gauge_factor`). This is the same as `gauge.align_common_phase_and_delay` on explore/gauge-alignment and as Sionna `Paths.cfr`.
  - E1 therefore samples c(u, τ(x) **+** τ_c). §3.3 and §4.1 had "−τ_c" and were corrected. T07b compensates τ_c on Y before the FFT, so its lookups do not depend on τ. With the true τ_c the N map equals the S map to 6e-16.
  - `draw_gauges` returns φ in [0, 2π) and an unwrapped τ. gauge.py wraps φ to (−π, π] and τ to [−T/2, T/2); `metrics.gauge_errors` compares modulo 2π, and modulo T when it is given `period`.
- **Interpolation (T07a).** All three axes are periodic, and sample k of axis a sits at `origins[a] + k·periods[a]/shape[a]`.
  - The angle-delay volume uses periods (1/s, 1/s, 1/δf), i.e. (2, 2, N/B), and origins (u_y[0], u_z[0], 0) = (−1, −1, 0).
  - Taps are in C order over the three axis offsets, with a C-order flat index. Tricubic is Keys with a = −0.5.
  - `scatter_add` is the exact adjoint of `gather`.
- **Observables (T04).**
  - `noise_var` is always the raw per-sample σ². Power nodes carry `meta["noise_dof"]` (I 2N; I_n0 and ID 2; ID-o 2M; I-o 2MN), and the noise-only mean is dof/2 · σ² (`kernels.noise_floor`).
  - IP_W and P_W remove one phase per (v, b, n), shared by both hemispheres. The omni column removes one phase per (v, b, h, r, col). The per-(c, m) form of §2.3 would keep the inter-hemisphere phase, so P-o would not be empty.
  - D, D_PHAT and D-o use an order-statistic CFAR (per-profile median / ln 2, threshold −ln(pfa)·noise, pfa 1e-4, at most 3 returns). They need no σ² and are exactly covariant with integer circular shifts.
  - Parameter defaults: PHAT mask |Y| ≥ 1·σ; partial-D bins (2k+1)N/8; 1el element (3,3). Node names are ASCII, with aliases for `I@n0`, `P×K` and `IP×K`.
- **Incoherent kernels (T08).** K = 1[h_p = h] · const · |γ|² · K_y K_z K_t, where K_axis = `dirichlet_power` on the fftshifted native grid and K_t is evaluated at it/N − δf(τ_p + τ_c). const is 1 (ID), N (I), 1 (I_n0), M (ID-o) and MN (I-o). The layout equals `extract(Y, node).data`, and the kernel matches `extract(atom_cfr)` to 3e-15.
- **Reference captures and RNG streams (T06, T09).**
  - Two different references are in use. The gauge reference c0 = (order[0], 0) is passed as `ref`. The power reference `c_ref` is the lower-median-power LoS-visible capture (C8); callers pass `los_visible & training_mask`.
  - P_ref sums the hemisphere powers (mean over r, col, n of Σ_h |Y|²), not |front + back|².
  - Capture streams are `SeedSequence([ds, v, b, realization])`. The noise is drawn first; for b = 0 only, the view's element gains and calibration noise follow. Gauge streams are `SeedSequence([ds, realization, 2**32−1, k])`. A tag word is needed because SeedSequence zero-pads short entropy.
  - T09 uses `SeedSequence([seed, 0|1|2])` for placement, amplitudes and jitter, so an L0-mm phantom has exactly the points of the plane-wave phantom with the same seed.
- **Singular points.** A point within `LOS_VS_TOLERANCE_M` = 1e-9 m of a UE, or of a BS in BV space, makes `capture_factors`, `SeparableOperator` and `PowerOperator` raise. `backproject` and `power_backproject_grid` return 0 for such points instead (`forward_exact._singular_mask`). In VS space a point at a BS is the LoS source and is valid.

**Deviations from the §8 entries.** No acceptance threshold was relaxed.

- **Extended signatures.**
  - `CaptureGeometry` has `aperture_shape` (for the 4×4 micro scene) and an optional `bs_rot` (`from_orientations(..., bs_look_at=…)`).
  - `periodic_weights` takes `origins`, and `noise_var_estimate` takes `delta_f`.
  - `power_backproject_grid` takes a required `space` and the keywords `product`, `kind`, `oversample`, `pattern` and `polarization`.
  - `make_tracks` takes `freq_offsets`, `sigma_t`, `ref`, `hardware`, `Y_los` and `c_ref`. The scatter-referenced SNR needs `Y_los` and is NaN without it.
  - `draw_gauges` also accepts `sigma_t="uniform"`, which needs `period`.
  - `atom_cfr` takes `polarization`, and `nms_peaks` names its first argument `density`.
- **T01 test tolerance.** The N = 128 `delay_period` test uses 1e-13 s because of the float32 grid.
- **T03 plane-vs-spherical bound.** k·max|q|²/(2r) is not strict for every direction: it can be exceeded by up to k|q|⁴/(8r³), which is 1.8e-8 rad at 10 m. The test uses directions where the bound holds; at boresight it is tight (0.16480 vs 0.16482 rad).
- **T07b: what BP approximates.** BP approximates A^H(W·Y), with W the peak-normalised separable Taylor window and no division by the window gain. The −30 dB gate is therefore measured against dense^H(W·Y). A non-planar aperture (element jitter) raises; L0-mm jitter must use `forward_exact` or `forward_sep`.
- **T07c.**
  - Only `wavefront="plane"` without squint is supported.
  - "constrained" (a″) is linear in complex quadratic coefficients x[P, 3] of cos θ_inc = |d·n|, where cos θ_inc = 1 for the LoS. It is defined in VS space only. The shared-phase constraint is left to the solver via the exact projection `project_shared_phase`.
  - A_c and D_c are cached up to 4 GiB. They cannot come from a recurrence, because the float32 grid is non-uniform by about 2 Hz, which would give about 2.5e-5 rad of error.
- **T08.** The MC test draws random per-trial atom phases, the incoherent model's own assumption. Fixed-phase cross terms (2–15 % on a 4×4 aperture) are the §3.3 limitation and are not gated. The BP gates were set from measurements, because the doc gives none: trilinear 6e-2 max / 3e-2 l2, tricubic 3e-3 / 2e-3.
- **T09 choices.**
  - Default capture: an 8-view ring at r = 30 m around (0, 0, 5), UE height 1.5 m, BS (−70, 5, 25) with tr38901, N = 128.
  - Default grid: a 20 m cube at 0.5 m. L0c points stay in the 10 m box.
  - Element jitter: 3-D isotropic, default std λ/200, shared by all captures.
  - L0d: a Born plate with ρ_s = j√(4π)Γ/λ, which is 1.005 of the image source at normal incidence.
  - L0e: LoS plus first-order images on infinite planes, with scalar ITU-R P.2040 TM (ground) and TE (walls) coefficients.
  - L0b on the 2 m CI grid with offset_max 0.2 m is feasible only for separations near a multiple of the spacing (2 m works; 1 m raises).
- **T10.** `loc_error_decomposed` needs a caller-supplied reference point for the range direction. `min_value` is effectively required in `nms_peaks`, because zero plateaus are local maxima.

**Acceptance evidence (task tests).**

| Criterion | Measured | Gate |
|---|---|---|
| T03 `atom_cfr` (tr38901, vv) vs Sionna GPU mock, 7 LoS views | 6.2e-5 … 4.8e-4 max rel. The through-building view ue_000001 fits one complex β = 0.082 with residual 2.9e-5 | 1e-3 |
| T04 noise estimate, 69 dB LoS peak, 30 dB | ≤ 5.3 % over 25 seeds; diffuse clutter ≤ 7.0 % | 10 % |
| T07a Dirichlet, 8× oversampled | trilinear 1.8e-2, tricubic 4.7e-4 | 3e-2 / 1e-3 |
| T07b fast BP vs dense^H(W·Y), 10 off-grid atoms beyond 5 m | trilinear −42.4 dB, tricubic −76 dB; real Sionna data −42.4 dB | −30 / −40 dB (pins −36 / −60) |
| T07c adjoint; equality with `atom_cfr` | 3e-19 … 1e-17; 5e-16 | 1e-10; 1e-12 |
| T08 adjoint; MC E\|c\|² (T = 20 000) | exact to round-off; max 2.6 % (SE ≤ 0.73 %) | 1e-12; 5 % |
| T09 L0-mm floor vs prediction | ≤ 0.14 dB; combined floor −27.4 … −30.0 dB at 20–40 m | 3 dB |
| T10 sub-voxel refinement, 0.3·spacing offset | 0.021 voxel (σ = 1.5 voxels); tilted blob 0.031 | 0.05 |

**Independent review spot checks.** These scripts were written by the reviewer and do not reuse the task tests.

- **Forward model vs Sionna.** A from-scratch model of `sionna_mock_los.npz` shares no code with `forward_exact`: it re-derives the element positions, the TR 38.901 gain, the look-at rotation and the V-pol vectors. It gives max relative errors of 4.52e-4 (front) and 4.75e-4 (back), |β| = 1.000000, and a residual of 2.9e-6 / 3.6e-6 after the fit. `atom_cfr` gives the same numbers, and the grid equals the fixture grid exactly.
- **End-to-end peaks.** 60 random VS/BV atoms were placed with random UE yaw/pitch/roll, N = 32. The worst peak offset is 0.497 cells through the master chain (64×64 FFT) and 0.498 oversampled cells in the T04 (8,8) volume. No atom entered the wrong hemisphere.
- **Adjoints.**
  - T07c: 8e-19 … 1e-17 for all β models, cached and uncached, with tr38901 + vv and random gauges. The gauged forward equals `apply_gauge(atom_cfr)` to 8e-16.
  - T08: ≤ 6e-17 for all five products with random τ_c. One column equals `extract(apply_gauge(atom_cfr))` to 4e-15.
- **E1 vs exact.** 12 atoms and 312 evaluation points beyond 5 m, V = 3, B = 2. With Taylor, trilinear is −42.6 / −43.7 dB (vs/bv), and on pure-noise Y it is −42.9 dB. Tricubic ranges from −61 to −82 dB over all cases. Without a window on noise, the trilinear max/peak is −34.7 dB: it still meets the −30 dB gate but not the −36 dB atom pin.
- **Noise estimate.** L0e scene, 6 views, 30 dB SNR, 20 seeds: +2.8 % mean bias and ≤ 5.4 % error. The bias comes from the −32 dB angular sidelobes of the 62 dB LoS peak. On pure noise the estimate has +0.1 % mean and ≤ 3.5 % error.

**Known limitations to carry into Phase 1.**

- The noise estimate degrades with dense strong specular multipath: 10 paths per view at 63–69 dB give about +8 %, and 20 paths give about 16 %.
- The E1 per-point cost is about 0.75 µs per (point, capture), mostly `capture_factors` and `periodic_weights`. The §4.1 extrapolation is 42.9 s for G_phys with 16 captures against a 60 s target. Uncached T07c and T08 take about 4–12 s per operator application at P = 5e4, against 0.3–0.7 s cached, which is the default below 4 GiB.
- The T09 phantoms use the scalar polarisation model. Every operator defaults to `polarization="none"` and accepts `"vv"` (`forward_exact`, `forward_sep`, `backproject`, `kernels`).

### Phase 1: the baselines (E1/E2) and the runner

**T11 E1 solvers.**
- Files: `rt/solvers/bp.py`, tests.
- Implement:
  - cone BP and log-mean fusion (I, I@n0);
  - return-to-point splatting (D);
  - PHAT BP (P, P_W, DP);
  - coherent envelope and ROI BP (IP, IP_W, IDP);
  - N-mode sums with a `tau_hat` argument;
  - the blind τ search: `blind_tau_search(bp_fn, Y, geom, grid, tau_range) -> tau[V,B]`, which maximises cross-view envelope consistency.
- Tests (L0 micro, §6.7 thresholds):
  - IDP-S and ID-S within 1.2 m after refinement;
  - IDP-S ROI within 0.25 m;
  - the blind τ search recovers τ within 1 ns at 30 dB.
- Deps: T07b, T08, T09, T10.

**T12 E2 power inversion.**
- Files: `rt/solvers/power.py`, tests.
- Implement `kl_em`, `is_mlem` and `nn_fista_l1(tv=False)` over T08 operators on a pruned support, plus `prune_support(map, grid, radius, cap)`.
- Tests: monotone objective; two points 3 m apart recovered within 1 m on L0.
- Deps: T08.

**T13 E2 coherent inversion.**
- Files: `rt/solvers/coherent.py`, tests.
- Implement `tikhonov_lsqr`, `complex_l1_fista`, `mmv_group_lasso(beta_model)` and `roi_grids_from_detections(det, half_width=0.25, spacing=λ/4)`.
- Tests:
  - matches a dense solve within 1e-6;
  - MMV recovers the support (recall 1.0 on L0c K = 4) with random per-view phases.
- Deps: T07c.

**T14 Nuisance atoms and gauge solvers.**
- Files: `rt/gauges.py`, tests.
- Implement:
  - `fit_los_ground(Y_c[2,8,8,N], geom, v, b, mode={"complex","power"}, with_gauge: bool) -> dict(g_los: float>0, a_ground: complex, phi, tau, resid)`, with the LoS phase fixed by the model;
  - `power_xcorr_delay`;
  - `self_calibrate(solve, Y, op_factory, n_iter=10, init=None, ref=c0) -> (x, phi[V,B], tau[V,B], history)`. `solve(Y_aligned) -> x`. Gauges per capture come from `align_common_phase_and_delay(Y_c, forward_c(x))`. It stops when the relative change in the profiled loss is below 1e-4;
  - `varpro_cost_and_grad`.
- Tests:
  - with LoS present at 30 dB, injected (φ, τ) recovered within 1° + |ε_LoS| and 0.05 ns;
  - with a free complex LoS amplitude (negative control), φ is unrecoverable;
  - self-cal on L0c K = 4, V = 8, 30 dB: gauges within 2° / 0.05 ns after removing the global phase, and M5 complex NMSE within 1 dB of the S-mode NMSE of the same solver.
- Deps: T06, T07c, T13; **gauge-alignment merged**.

**T15 Configuration registry.**
- Files: `rt/configs.py`, tests.
- A declarative `CONFIGS` covering the 14 core configurations, I@n0, P_W, IP_W, and the omni, 1el, partial-D and sync variants. Each entry gives: observable, lattice membership (NB/WB/omni/none), spaces, solver chain, N strategies and hyperparameter ranges. There is **no static ill-posed flag**.
- Tests: all names present; chains reference existing callables; lattice edges match §2.3.
- Deps: T11–T14.

**T15b Identifiability test.**
- Files: `rt/identifiability.py`, tests.
- Implement:
  - `gauge_reduced_fim(jac[n_obs, n_par], weights[n_obs], nuisance_idx) -> J`;
  - `ill_posed(J, pos_idx, cond_max=1e8, std_max_m=10) -> (bool, cond, crb_std)`;
  - `numeric_jacobian(model_fn, theta, eps)`.
- Weights are Gaussian until T30 provides exact ones.
- Tests: one-view D-N flagged; eight views with three shared VS not flagged; the global-phase null is removed.
- Deps: T07c.

**T16 Benchmark runner and CLI.**
- Files: `src/plateau_rt/application/rf_tomography_io.py`, `.../rf_tomography_benchmark.py`, a `rf-tomo-bench` command in `cli/main.py`, tests.
- Input: `--dataset MANIFEST --suite {unit,smoke,full} [--tracks …] --out DIR`.
- Output: `results.jsonl`, `recon/…/*.npz`, `run_manifest.json`.
- The runner computes `ill_posed` via T15b.
- Tests: the §6.7 micro-scene criteria and the output schemas.
- Deps: T15, T15b; multi-bs for v3.

### Phase 2: ground truth, dataset profile, heavy CI

**T17 tomography_gt.**
- Files: `rt/gt.py`, `application/rf_tomography_gt.py`, a `rf-tomo-gt` command, tests.
- Implement:
  - VS clustering;
  - `vs_rho_eff` with G_b(d_dep) and P_rx;
  - `vs_theta_inc`;
  - `los_phase_model_error` and `los_amp_model_error_db`;
  - mechanism labels from `interactions`;
  - wrap flags;
  - surface sampling and observability.
- Tests: synthetic mirror plane (spread < 1e-6 m); observability on a box; ε_LoS = 0 on L0f with the iso pattern.
- Deps: T01, T02; **path-gt-rich, image-sources, multi-bs, rich-mock-scene merged**.

**T18 Surface and plane metrics.** Extend `rt/metrics.py` and `application/rf_tomography_gt.py` with P/R/F@d, Chamfer, plane error and stratification. Deps: T10, T17.

**T19 Impairments extension.**
- Files: `domain/rf_camera/impairments.py`, tests.
- Add:
  - a `bs` axis;
  - `noise_var_abs` (absolute σ²);
  - `apply_hardware_impairments` (gauge-free);
  - `apply_gauge`;
  - per-(v, b) GT;
  - pose perturbation;
  - `calibration_capture(element_gains, snr_db, rng)`.
- Wire T06 to use them for the observed track.
- Tests: σ² is independent of signal power when `noise_var_abs` is set; the calibration estimate is within 0.1 dB / 1° at 40 dB.
- Deps: **observed-impairments merged**, T06.

**T20 Tomography dataset profile.**
- Files: `adapters/sionna/rf_camera_dataset.py`, `rf_tracing.py`, `Makefile`, CLI.
- Adds: pose bank, multi-height, N = 128, BS set, `los=False` trace, variants, splits and `nested_view_order` seeds, placement, pattern metadata, P_ref and σ², hashes.
- Tests: configuration unit tests; heavy-CI manifest check.
- Deps: **multi-bs, solver-profiles, rich-mock-scene merged**; T02, T06.

**T21 Pattern and convention validation on real traces.**
- Files: `scripts/ci/check_tomography_resynthesis.py`, a `run-heavy.sh` step.
- Checks: L0f NMSE < 1e-3; tr38901 direct-path ratios within 0.5 dB; |ε_LoS| P90 reported.
- Deps: T03, T17, T20.

**T22 Heavy CI smoke.** `scripts/ci/check_tomography_smoke.py` implementing §6.7. Deps: T16, T17, T20.

### Phase 3: sparse/parametric, hybrids, analysis

**T23 E3 sparse with continuous refinement.**
- Files: `rt/solvers/sparse.py`, tests.
- Implement CLEAN, OMP, NOMP and LM on the exact operator (≤ 100 atoms), and `gaussian_init(mean[K,3], cov[K,3,3], amp[K])`.
- Tests: L0c K = 16 recall ≥ 0.9 at 1 m for IDP-S at 30 dB; median error ≤ 0.1 m.
- Deps: T13.

**T24a ESPRIT and MDL.**
- Files: `rt/solvers/esprit.py`, tests.
- Implement:
  - `esprit_2d(Yc_n[8,8], sub=(5,5), order=None) -> (u[K,2], a[K])`;
  - `esprit_3d(Yc[8,8,N], sub=(5,5,48), order=None) -> (u[K,2], t[K], a[K])`;
  - `mdl_order(eigs, n_snap)`.
- Tests: two paths 0.5 cell apart resolved within 0.02 in u at 30 dB; MDL correct for K ∈ {1..4} in ≥ 95 of 100 trials at 20 dB.
- Deps: T04.

**T24b VS triangulation.**
- Files: `rt/solvers/triangulate.py`, tests.
- Implement `triangulate_vs(returns: list[(v, b, u[K,2], t[K])], geom, gate_m=1.0) -> (s[K',3], membership)`, using gated clustering of p_v + cτ·R_v u.
- Tests: L0e VS within 0.1 m at 30 dB.
- Deps: T01, T24a.

**T24c Pseudorange solver.**
- Files: `rt/solvers/pseudorange.py`, tests.
- Implement `pseudorange_solve(returns, geom, bias_model={"per_capture","N_sep"}, init) -> (s[K,3], bias[V,B], cov)`, Gauss–Newton over the directions and delays of the returns, with LoS atoms excluded in blind mode.
- Used by D-N, ID-N, DP-N and IDP-N.
- Tests: L0e with ≥ 3 VS shared over 8 views gives bias within 0.1 ns; the (1 VS, 2 views) case is flagged by T15b.
- Deps: T15b, T24b.

**T25 Likelihood library and discrepancy weights.**
- Files: `rt/likelihoods.py`, tests.
- Implement the §3.6 exact and robust losses with gradients, the profiled gauge loss, and the type-II-ML s²_m update.
- Tests: gradient checks to 1e-6; s² converges within 10 % on synthetic mismatch.
- Deps: T07c, T08.

**T26 Hybrids F0, F1, F4, F7.**
- Files: `rt/fusion.py`, tests.
- Tests: pooling of identical maps is idempotent; R2 with all-complex equals IDP to 1e-10.
- Deps: T12, T13, T25; partial-observations.

**T27a F3 stage A (support).** `stage_a_support(Y, geom, grid_1m) -> points[P,3]`. Accept: L0c K = 16, true points inside the support ≥ 0.95. Deps: T12.

**T27b F3 stage B (gauges).** `stage_b_gauges(Y, geom, support, strategy={"los","blind","pseudorange"}) -> (phi, tau)`. Accept at 30 dB: LoS within 2° + |ε_LoS| and 0.05 ns; blind within 0.2 ns. Deps: T14, T24c.

**T27c F3 stage C (coherent LS on support).** `stage_c(Y, geom, support, gauges, mode={"S","N"}, beta_model) -> x`. Accept: complex NMSE ≤ E2 IDP-S NMSE + 0.5 dB on L0c. Deps: T13.

**T27d F3 stage D (gridless continuation).** `stage_d(Y, geom, x0, schedule) -> atoms`, with the λ/4 local search. Accept: L0c K = 4 median error ≤ λ/4 at 30 dB with pose jitter 0. Deps: T23, T25.

**T27e F3 stage E and wrapper.** `stage_e_polish(...)` and `f3_cascade(Y, geom, cfg) -> (atoms, gauges, gaussian_init)`. Accept: F3 IDP-S localisation error ≤ E2 IDP-S in ≥ 18 of 20 L0c seeds. Deps: T27a–d.

**T28 Hybrid F5 coherence groups.**
- Files: extend `rt/fusion.py`.
- Implement group specs (per capture / BS / view / sub-band / all) with per-group τ̂_G.
- Tests:
  - the all-group equals S;
  - the per-capture group with true τ_c applied equals N-mode BP with the same τ_c, and differs from S when τ_c ≠ 0 is not applied.
- Deps: T11, T13.

**T29 Hybrid F6 dual space and cross-BS facets.** `rt/surfaces.py`. Tests: L0e walls within 2° and 0.2 m. Deps: T17, T23.

**T30 Resolution and exact CRB.**
- Files: `rt/information.py`, tests.
- Fisher information per observable (Jacobians J_μ by T15b's `numeric_jacobian`, or analytically):
  - complex Gaussian (IDP, IP, IP_W, IDP-o): J = (2/σ²)·Re(J_μᴴ J_μ);
  - noncentral χ² (ID, I, I@n0, ID-o, I-o): J = Σ_bins ∇λ ∇λᵀ · F_ncχ²(λ; dof), with F tabulated by 1-D quadrature;
  - projected normal (P, P_W, DP): J = Σ [∇ψ∇ψᵀ F_ψ(γ) + ∇γ∇γᵀ F_γ(γ)], tabulated;
  - D lists: J = Σ_k J_kᵀ Σ_k⁻¹ J_k, with Σ_k the single-path complex-Gaussian CRB, conditional on detection;
  - N/A: PHAT BP, pooling, T(u,t).
- Also: gauges by Schur complement, operator SVD on the ROI, and PSF widths / PSLR / ISLR.
- Tests:
  - the single-tone delay CRB matches the closed form within 5 %;
  - the Rician F → 2/σ²-equivalent limit at high SNR;
  - F_ψ(γ) → 2γ at high SNR.
- Deps: T07c, T08, T15b, T25.

**T31 Ablation analysis and report.**
- Files: `application/rf_tomography_report.py`, tests.
- Implement Shapley and interactions per sublattice (NB, WB, omni), synergy, joint − stack, Δ_sync, bootstrap, Wilcoxon, `report.md` and figures.
- Tests: Shapley on a synthetic additive metric returns the known values per sublattice.
- Deps: T16, T18, T30.

### Phase 4: robustness, scale, extras

**T32 Robustness sweeps.** Presets for the observed track, pose jitter, element-gain residuals, F/B collapse, masks and sub-bands, the omni column and L0-mm. Deps: T19, T26.

**T33 Accelerated backend (optional).** `adapters/accel/rf_tomography_torch.py`, with the same API as T07b/T07c/T08. It is the only route for full-grid E2 and unpruned VS MMV. Equivalence test on CPU torch. Deps: T07b, T07c, T08; Q5.

**T34 L3/L4 runs.** ROI and G_vs extent logic, wrap flags, full-suite configuration. Deps: T20, T27e, T29.

**T35 I-T transmission tomography (optional).** `rt/solvers/rti.py`. Deps: T02, T20; Q2.

**T36 F8 optical oracle and overlays.** Deps: T31; feature/11-optical-reference.

**T37 RF-GS hand-off.** Freeze `gaussian_init`, the splits, `metrics.py` and the M4 prediction rules as the shared scorer. Deps: T23, T31.

**Critical path:** T01 → T03 → T04 → T07a → T07b/T07c → T08 → T11/T12/T13 → T14 → T15/T15b → T16. The L0 core results need only master plus gauge-alignment. Sionna-scored results wait on the merges listed in T17 and T20.

---

## 9. Risks and open questions

### 9.1 Main risks

- **Sparse specular regime.** BV maps of the current profiles are mostly empty space. VS and point metrics are primary there; BV needs the diffuse profile or many more views and BSs.
- **Model mismatch.** Specular anisotropy, multi-bounce, refraction, G_b and polarisation, and diffuse noise. Mitigated by two scene spaces, the three β options, discrepancy variances, the mechanism axis and the L0-mm floor.
- **Overlap in one angle-delay cell** (multi-bounce clusters, diffuse patches; LoS plus ground bounce in omni). It biases power-domain estimates, including omni delay anchoring by up to about 1 m. Coherent joint fitting mitigates it.
- **LoS phase-model error** limits φ_c recovery. It is measured (ε_LoS) and reported.
- **Cycle skipping and pose sensitivity** for coherent S. Handled by continuation, the λ/4 search and the pose-jitter sweep.
- **Compute.** CPU E2 is bounded by pruning (§4.1). Full-grid runs depend on T33.
- **Hyperparameter sensitivity** across about 70 rows. Handled by the frozen validation protocol and equal budgets.

### 9.2 Decisions

Decided 2026-09-25: every default below (in brackets) was adopted.

1. **Letter semantics and the lattice.** Read I = amplitude, D = time of flight, P = carrier phase, with the camera's angle resolution always available. Add three lattice-completion nodes (I@n0, P_W, IP_W) so the "value of each type" analysis is not confounded by data volume. [Yes.]
2. **Elevated UEs.** Without elevated UEs, camera configurations still localise in 3D, but vertical precision is limited to about r·Δu_z (≈ 10 m at 40 m), and omni range-only configurations have a z-mirror ambiguity. [Measure this with M7/M8 on L2 first; add 10/25 m UEs only if the vertical error dominates. I-T deferred.]
3. **Headline RF-GS comparison.** Virtual-source results, or physical-space (F6) results. [Report both; F6 physical space is the headline.]
4. **"Non-simultaneous".** Clocks and phase only, or also scene changes. [Clocks and phase only.]
5. **Compute.** An optional GPU backend (torch or CuPy) for full-grid E2 and the nightly full suite, with CI gating only on the CPU unit tests and the ≤ 10-minute heavy smoke. [Yes.]