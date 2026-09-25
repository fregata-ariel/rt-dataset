# Multi-BS / multi-UE RF Camera Dataset

This milestone extends the validated 1-BS / 1-UE RF camera into a multi-BS,
multi-view dataset while keeping the observation model image-like and compact.

## Camera convention

Each UE is an RF camera.

- UE local **+x** is camera forward.
- The receive aperture lies in local **y-z**.
- Each UE uses Sionna's `look_at` orientation convention.
- The developed image is a direction-cosine disk `(ky/k, kz/k)`.
- A front-hemisphere ray is reconstructed as

```text
kx/k = +sqrt(1 - (ky/k)^2 - (kz/k)^2)
```

### Front / back hemispheres

The 2-D planar aperture alone cannot tell `+kx` from `-kx`: a wave arriving
from behind lands on the same `(ky/k, kz/k)` pixel as its mirror image in
front. (With the earlier directional `tr38901` element, whose front-to-back
ratio is only 30 dB, the BS behind a camera still produced the brightest pixel
of its image as such a mirrored ghost.)

The Rx element is therefore the `rf_camera_split` pattern
(`src/plateau_rt/adapters/sionna/rf_patterns.py`). It fills Sionna's two
antenna-pattern slots with the front (`kx >= 0`) and back (`kx < 0`) halves of
a vertically polarized isotropic element, so one `PathSolver` call records both
hemispheres separately and exactly:

- `front + back` equals the isotropic element (checked in the heavy CI);
- a finite front-to-back ratio `g` can be synthesized later as
  `front + g * back`, without re-tracing;
- the developed image uses the **front** hemisphere only. Arrivals from behind
  are excluded like a light source behind an optical camera: the BS behind a
  UE is not imaged, but the scene it illuminates in front is. A view with
  nothing arriving from the front therefore has an empty image.

Because both pattern slots are used, dual polarization is not available with
this pattern.

### Image quantity

The FFT of an isotropic aperture measures the plane-wave spectrum `U(ky, kz)`,
an amplitude per unit direction-cosine area. A pixel covers the solid angle
`dOmega = dky dkz / kx`, so the developed image is the complex amplitude per
unit solid angle

```text
A(ky, kz) = kx * U(ky, kz),   kx = sqrt(1 - ky^2 - kz^2)
```

`|A|^2` is power per steradian (the RF analogue of radiance), the quantity a
Gaussian-Splatting renderer integrates over solid angle. The weight `kx` is a
known per-pixel function stored as `solid_angle_weight` in `camera_model.npz`;
the raw aperture CFR is kept unweighted.

## Mock geometry

The included mock building's CityJSON vertices span `[0,10] x [0,10] x [0,10]`
m, but the scene builder recenters `x`/`y` on the bounding-box centre and
keeps `z` from its minimum (`src/plateau_rt/adapters/plateau/cityjson_parser.py`),
so in scene coordinates the building occupies `x, y in [-5, 5]`, `z in [0, 10]`
m. The smoke test uses:

- target / look-at: `(5, 5, 5)` m -- this sits on the building's `+x +y`
  vertical edge (`x = y = 5`, mid-height), not its centre
- 8 UE views on a 30 m radius ring
- UE height: 1.5 m
- BS `bs_000` at `(-50, -50, 30)` m
- BS `bs_001` at `(60, 35, 25)` m
- each BS panel looks at `(5, 5, 5)` m by default
  (`--bs-look-at` can override per BS)
- 8 x 8 Rx aperture, 0.5 lambda spacing
- 3.5 GHz carrier
- 100 MHz bandwidth
- 64 uniformly spaced baseband frequency bins

All eight receivers and both transmitters are solved in **one** Sionna
`PathSolver` call.

The mock has a single building and no ground. Each (view, BS) pair carries
energy in one hemisphere only: `bs_000` is in front for `ue_000000`–`ue_000003`
and `ue_000007` (behind for `ue_000004`–`ue_000006`), while `bs_001` is in
front for `ue_000002`–`ue_000007` (behind for `ue_000000`/`ue_000001`). Energy
then sits only in that hemisphere, and the 5 back-hemisphere pairs (3 for
`bs_000`, 2 for `bs_001`) have empty front images (`dominant_delay_s` all NaN,
`dominant_delay_power` 0 over the full disk).

`bs_000` reaches `ue_000001` only through a weak through-edge residual
at the LOS delay (single path at 371.88 ns, front energy 6.478e-07,
~150x below a typical direct pair, dominant delay 370.0 ns over
12849 valid pixels) because its LOS grazes the box edge. `bs_001`'s
LOS to `ue_000005` is blocked by the box: that pair receives only a
weak residual with two through-box paths, 316.16 ns at the LOS delay
transmitted straight through the box and 371.92 ns reflected inside
the box (total front energy 5.461e-07, dominant delays 320.0-370.0
ns over 12849 valid pixels).

All other front pairs show a single direct path whose dominant delay matches
the geometric LOS within the 10 ns delay resolution (e.g. `ue_000000`/`bs_000`:
350.83 ns LOS, 350.0 ns image, front energy 9.752e-05; `ue_000002`/`bs_001`:
199.51 ns LOS, 200.0 ns image, 1.906e-04; `ue_000004`/`bs_001`: 310.72 ns LOS,
310.0 ns image, 1.297e-04). `ue_000007`/`bs_001` receives its direct path
(219.15 ns) plus a clean reflection off the box's +x wall (316.16 ns)
and a weaker through-box path (371.92 ns) that is reflected inside
the box; front energy 1.873e-04. It is the only pair with an exterior
wall reflection in this mock. For the richer 4-building scene with a ground
plane, see [Mock city (ground plane)](#mock-city-ground-plane) below.

## Mock city (ground plane)

The mock city (`data/raw/mock_city.city.json`) has 4 concrete box buildings
around a plaza centred on the origin:

- nw: x[-26,-12] y[10,24] h18
- ne: x[12,24] y[10,24] h30
- se: x[10,26] y[-24,-12] h12
- sw: x[-24,-12] y[-24,-10] h24

`make build-mock-city` adds a 200 m square ground plane at z = -0.01 m
(placed there to avoid coplanar faces with the building GroundSurface at
z = 0), made of `itu_medium_dry_ground`.

`make rf-camera-multiview-mock-city` uses 12 views on a 40 m radius ring,
UE height 1.5 m, target (0, 0, 8) and BS at (-70, 5, 25).

The BS is behind the camera (camera-local kx < 0) in `ue_000005` (kx = -0.4),
`ue_000006` (kx = -0.7) and `ue_000007` (kx = -0.3), all three with unblocked
line of sight. In these three views the **back**-hemisphere energy is mainly
the direct LoS path and the **front** energy comes from building and ground
reflections, which explains the large back/front ratios (e.g. +46.0 dB in
`ue_000006`). `ue_000004` sees the direct path nearly grazing, just in front
(kx = +0.07).

Per-view hemisphere energies from one GPU run (rounded to 0.1 dB; GPU
tracing is not bit-reproducible, so reruns will differ slightly):

```text
view        bs_front  kx_bs    front_dB   back_dB   back-front_dB
ue_000000  True      +1.0      -40.2      -inf      -inf
ue_000001  True      +0.9      -79.8      -inf      -inf
ue_000002  True      +0.8      -67.3      -inf      -inf
ue_000003  True      +0.5      -40.2      -inf      -inf
ue_000004  True      +0.1      -38.9      -51.4     -12.5
ue_000005  False     -0.4      -44.2      -34.8       9.4
ue_000006  False     -0.7      -78.0      -32.0      46.0
ue_000007  False     -0.3      -44.7      -36.0       8.7
ue_000008  True      +0.2      -39.2      -inf      -inf
ue_000009  True      +0.6      -40.4      -inf      -inf
ue_000010  True      +0.8      -61.7      -inf      -inf
ue_000011  True      +1.0      -61.1      -inf      -inf
```

(`-inf` means no traced energy in that hemisphere.)

- **`ue_000004`:** its back-hemisphere energy (-51.4 dB) is mainly the
  ground-reflected BS path. The mirror image of the BS in the ground plane is
  at camera-local kx of about -0.06, while the direct path is just in front.
- **All other views with the BS in front** (every view except `ue_000004` to
  `ue_000007`) have an empty back hemisphere (`-inf`). No scatterer lies
  behind the cameras: the buildings stay within 35.4 m of the origin, the
  ring radius is 40 m, and every camera looks inward at the target. Their
  ground reflection also arrives from the front. So in this scene,
  back-hemisphere energy appears only when the BS itself, or its ground
  image, is behind the camera.
- **Blocked line of sight:** the direct BS path is geometrically blocked by
  buildings for `ue_000001` (nw and ne), `ue_000002` (nw), `ue_000010` (sw)
  and `ue_000011` (se). This is why their front energy is 20-40 dB below that
  of the unblocked front views, which are around -40 dB.

### Carrier limit of the ground material

The ground material `itu_medium_dry_ground` is only defined for carriers
from 1 to 10 GHz. `rf-camera` and `rf-camera-multiview` fail fast with exit
code 2 and a clear message when the scene contains the ground plane and the
carrier is outside that range, before any Sionna tracing starts:

```text
Error: Invalid value for --carrier-ghz: ITU material 'itu_medium_dry_ground'
is only defined for carriers from 1 to 10 GHz, got 28.000000 GHz. Rebuild
without --ground-plane-size-m or use a carrier inside the range.
```

The scene-build `manifest.json` records a `ground_plane` block with `size_m`,
`z_m`, `material` and `valid_carrier_range_hz`. This key and the extra
`ground_plane.ply` mesh appear only when the plane is enabled
(`--ground-plane-size-m` > 0); a default build is unchanged.

## Run

```bash
make rf-camera-multiview-mock
```

or, for another scene or ring:

```bash
PYTHONPATH=./src uv run python -m plateau_rt.cli.main rf-camera-multiview SCENE.xml OUT \
  --num-views 8 --radius-m 30 --ue-height-m 1.5 --target 5 5 5 \
  --bs-position -50 -50 30 --bs-position 60 35 25 \
  --frequency-bins 64 --bandwidth-mhz 100
```

Repeat `--bs-position` for each base station. A single `--bs-position` keeps
the original single-BS layout with `num_bs = 1`. Each BS looks at `--target`
unless `--bs-look-at` is repeated once per BS.

Poses and the camera model are in `src/plateau_rt/domain/rf_camera/camera.py`
(NumPy only); Sionna tracing is in
`src/plateau_rt/adapters/sionna/rf_camera_dataset.py`.

Unit tests (the Sionna adapter imports fine without a GPU):

```bash
scripts/ci/run-unit-tests.sh
```

## Coverage placement (#16)

`--placement coverage` computes a 2D path-gain map (radio map) at UE height with
Sionna's `RadioMapSolver`, keeps the cells above a threshold that are outside
buildings and far enough from the base stations, and draws UE poses at random
with a dedicated `--placement-seed`. The default `--placement ring` is
unchanged.

```bash
make rf-camera-coverage-mock
```

or manually:

```bash
PYTHONPATH=./src uv run python -m plateau_rt.cli.main rf-camera-multiview SCENE.xml OUT \
  --placement coverage --placement-seed 0 --orientation-policy face_bs \
  --num-views 8 --ue-height-m 1.5 --target 5 5 5 \
  --bs-position -50 -50 30 --bs-position 60 35 25 \
  --rm-center 0 0 --rm-size 80 80 --rm-cell-size 1 1 \
  --pl-threshold-mode relative_to_max_db --pl-threshold 30 \
  --building-clearance-m 1 --min-bs-distance-m 5 --min-ue-spacing-m 5
```

Coverage options:

| option | default | meaning |
|---|---|---|
| `--placement` | `ring` | `ring` or `coverage` |
| `--placement-seed` | `0` | dedicated placement seed |
| `--pl-threshold-mode` | `relative_to_max_db` | `absolute_db`, `relative_to_max_db` or `percentile` |
| `--pl-threshold` | `30` | dB (absolute / below max) or percentile |
| `--bs-aggregation` | `max` | `max`, `sum` or `all` over base stations |
| `--orientation-policy` | `face_bs` | `look_at_target`, `face_bs` or `random_yaw` |
| `--face-bs` | `strongest` | BS index or `strongest` for `face_bs` |
| `--pitch-deg` | `0` | forward elevation for `random_yaw` |
| `--building-clearance-m` | `1` | dilate the building mask [m] |
| `--min-bs-distance-m` | `5` | minimum 3D distance to any BS [m] |
| `--min-ue-spacing-m` | `2` | minimum horizontal UE spacing [m] |
| `--cell-jitter` | `0` | intra-cell jitter fraction in `[0, 1)` |
| `--rm-center` | target x y | radio-map centre x y [m] |
| `--rm-size` | `100 100` | radio-map size x y [m] |
| `--rm-cell-size` | `1 1` | cell size x y [m] |
| `--rm-max-depth` | `5` | `RadioMapSolver` max_depth |
| `--rm-samples-per-tx` | `1000000` | `RadioMapSolver` samples_per_tx |
| `--rm-seed` | `42` | `RadioMapSolver` seed (not the placement seed) |
| `--radio-map` | none | reuse a saved radio map (json, `placement/` dir or dataset dir) |

Building interiors still receive a nonzero path gain (refraction through
walls), so the indoor mask comes from an upward ray test: a cell centre whose
ray straight up hits scene geometry is under a roof and is excluded, together
with a horizontal `--building-clearance-m` dilation. Cells that are invalid
(aggregated gain `<= 0` or non-finite), too close to a BS, below the threshold
or excluded are never chosen.

Invalid cells include Monte Carlo sampling holes, not only true shadow. On the
mock city (1 BS, 120 m x 120 m, 1 m cells, 1e6 samples per transmitter), 22% of
the cells get no ray; the share of outdoor cells left invalid grows with the
distance to the BS (0% within 40 m, 15% at 70-100 m, 55% beyond 100 m), so
the candidates, and hence the drawn UEs, lean towards the BS and towards
line of sight. For large maps with small cells, raise `--rm-samples-per-tx`
(1e7 cut the invalid cells from 3119 to 667) or coarsen `--rm-cell-size`.

### Reproducibility

Sionna GPU tracing is **not** bit-reproducible run to run (measured ~`6e-7`
relative jitter in `path_gain`), so the radio map is saved as an artifact:

```text
OUT/placement/
  radio_map.json
  radio_map_path_gain.npy     # float32 [num_bs, ny, nx], linear
  radio_map_indoor_mask.npy   # bool [ny, nx]
```

`radio_map.json` records the scene, carrier, base stations, grid and solver
settings plus the sha256 of both `.npy` files. Passing the saved map back with
`--radio-map` reuses the exact float32 arrays, so `saved map + placement_seed`
reproduces the poses exactly; the placement is recorded in the manifest
`placement` section, from which the views can be rebuilt:

```python
from plateau_rt.application.ue_placement import replan_from_manifest
placement = replan_from_manifest("OUT")   # saved map + seed -> identical poses
```

Reusing a map requires the same carrier, base stations and UE height (checked
before tracing); `--rm-*` options are ignored then. The scene itself is not
hashed: a map computed on a different scene file only prints a warning, so keep
the map with the scene it was computed on. `--radio-map` is rejected with
`--placement ring`. Changing only
`--placement-seed` changes only the placement (and the traced views): every
other manifest section stays equal.

## Expected Sionna CFR shape

For the default 8-view, 2-BS mock:

```text
(8, 128, 2, 1, 1, 64)
 ^   ^    ^  ^  ^   ^
 UE rxant tx txant t freq
```

`rxant = 2 x 64`: Sionna fuses the pattern axis pattern-major, so channels
`0..63` are the front hemisphere and `64..127` the back hemisphere (each in
PlanarArray column-first order). The `tx` axis follows the order of
`--bs-position` (`bs_000`, `bs_001`, ...).

## Output layout

```text
rf_camera_multiview/
  dataset_manifest.json
  camera_model.npz
  path_geometry_gt.npz
  path_schema.json
  views/
    ue_000000/
      pose.json
      rf/
        aperture_cfr.npy
        bs_000/
          angular_cfr_center.npy
          angular_power_center.npy
          phase_valid_mask.npy
          dominant_delay_s.npy
          dominant_delay_power.npy
          angular_power_center.png
        bs_001/
          ...
      optical/            # only after rf-camera-optical (one render per view)
    ue_000001/
      ...
```

### Canonical vs derived data

`aperture_cfr.npy` is the canonical compact RF observation,
`[bs, hemisphere, row, col, frequency]` with hemispheres `(front, back)` and
BS ids `(bs_000, bs_001, ...)`. It preserves the complex CFR on the physical
UE aperture over frequency for every base station.

The following files, stored per BS under `rf/bs_XXX/`, are derived from that
BS's front hemisphere (as `A = kx * U`) and can be regenerated from the
aperture CFR:

- center-frequency calibrated angular complex image
- center-frequency power image
- phase-valid mask
- dominant delay map (`dominant_delay_s`: NaN outside the propagating disk and
  wherever `dominant_delay_power` is 0, i.e. no energy from that BS in that
  direction)
- dominant-delay power (`dominant_delay_power`: 0 there)

The full `[kz, ky, frequency]` or `[kz, ky, delay]` volume is intentionally not
stored for every production view. This avoids a large storage multiplier while
keeping all information needed to regenerate it.

## Path-level ground truth

`path_geometry_gt.npz` stores the per-path Sionna ray data behind
`aperture_cfr.npy`, so the observation can be resynthesised and analysed per
path. `path_schema.json` is the single canonical description of it; the
manifest records only the two file names.

The canonical (synthetic-array) arrays use manifest order for views and BSs:

```text
valid, tau, theta_t, phi_t, theta_r, phi_r   [view, bs, path]
a_baseband                                   [view, bs, hemisphere, row, col, path]
interactions, object_index, primitives       [view, bs, path, depth]
vertices                                     [view, bs, path, depth, xyz]
num_interactions                             [view, bs, path]
```

`a_baseband` is split into `[hemisphere, row, col]` exactly like
`aperture_cfr`. Paths are put in a canonical order (valid paths
first, sorted by delay quantised to `tau_quantum_s=1e-12 s`, ties broken by
descending power); the schema's `ordering` text documents that quantisation
only reduces, not prevents, run-to-run reordering. Invalid paths have
`tau < 0`. `object_index` indexes into the schema's `object_names` (sorted
scene object names; -1 for none or unknown). Sionna's `load_scene` merges
shapes that share a material by default, so the mock scenes report a single
`merged-shapes` object; `primitives` and `vertices` still locate each
interaction.

The stored aperture CFR resynthesises as

```text
H[..., f] = sum_p a_baseband[..., p] * exp(-j*2*pi*frequency_offsets_hz[f]*tau[p])
```

Only `a_baseband` is stored; the passband coefficient follows from
`a = a_baseband * exp(+j*2*pi*carrier_frequency_hz*tau)` for valid paths.
`a_baseband` is `views*bs*2*rows*cols*P*8` bytes, comparable to `aperture_cfr`
once the number of paths `P` approaches the number of frequency bins.

With `--explicit-array` (`config.synthetic_array == false`) Sionna's receive
array is not synthesised, so there are no per-element coefficients. The schema
mode is `sionna_native` and only the six geometry arrays above are stored
unchanged in Sionna's native `[view, rx_ant, bs, tx_ant, path]` order.

Read it through the typed reader (no Sionna needed):

```python
path_gt = dataset.path_geometry_gt
schema = path_gt.load_schema()   # mode, axes, ordering, resynthesis
arrays = path_gt.load_arrays()   # {'valid': ..., 'tau': ..., 'a_baseband': ..., ...}
path_gt.array_axes("a_baseband")  # ('view', 'bs', 'hemisphere', 'row', 'col', 'path')
```

## Optical reference renders

`rf-camera-optical` adds a co-registered optical render (pinhole photo/depth
and a render on this dataset's hemisphere grid) to every view -- a reference
for debugging and for a parallel optical Gaussian-Splatting dataset, not an
RF training target. See [docs/optical_reference.md](optical_reference.md).

## Camera model file

`camera_model.npz` contains:

- `ray_directions_local[H,W,3]`
- `valid_mask[H,W]`
- `solid_angle_weight[H,W]` (`kx`, 0 outside the propagating disk)
- `ky_over_k[W]`
- `kz_over_k[H]`

For each view, `pose.json` contains the UE position and the
`world_from_local_rotation` matrix. A local RF-camera ray can therefore be
mapped to world coordinates by

```text
ray_world = world_from_local_rotation @ ray_local
```

This is the intended bridge to a Gaussian-Splatting camera model.

## Manifest (schema version 3)

Besides the configuration and frequency grid, `dataset_manifest.json` records

- `config.tx_positions` (list of 3-vectors) plus optional `tx_look_ats`:
  `config.tx_position` was replaced by `config.tx_positions` and
  `RFMultiViewConfig(tx_position=...)` no longer exists (breaking change);
- `base_stations`: `bs_id`, index, position and look-at of every BS;
- `raw_observation`: axis order (`bs`, `hemisphere`, `row`, `col`,
  `frequency_offset`), BS ids and hemisphere names of `aperture_cfr`, and the
  Rx element pattern;
- `camera_model`: projection, developed hemisphere and the image quantity
  definition;
- `path_geometry_gt`: artifact file name (`path_geometry_gt.npz`) of the
  path-level ground truth;
- `path_schema`: artifact file name (`path_schema.json`) of its single
  canonical schema (array dtypes/shapes/axes/units, mode, ordering and the
  resynthesis formula);
- per view: pose, `artifacts` (pose and `aperture_cfr`), and a `bs` list with
  one entry per BS (`bs_id`, `bs_direction_local`,
  `bs_in_front_hemisphere` and `hemisphere_energy`, the sum of
  `|aperture_cfr|^2` per hemisphere), so views with back-hemisphere energy
  can be found without loading the arrays.

## Reading a dataset

`plateau_rt.application.rf_dataset_manifest` is a small typed reader for
`dataset_manifest.json` that needs only NumPy (no Sionna). New consumers
should use it instead of indexing the JSON by hand:

```python
from plateau_rt.application.rf_dataset_manifest import load_rf_dataset_manifest

dataset = load_rf_dataset_manifest("outputs/rf_camera_multiview")  # dir or manifest file
dataset.bs_ids                 # ("bs_000", "bs_001")
dataset.frequency_offsets_hz   # float64 [N]
dataset.aperture_cfr_axis_order  # ("bs", "hemisphere", "row", "col", "frequency_offset")
for view, bs in dataset.pairs():  # view-major, then BS order
    bs.hemisphere_energy, bs.bs_in_front_hemisphere, bs.artifact("dominant_delay_s")
cfr = dataset.load_aperture_cfr(view)  # [bs, hemisphere, row, col, freq]
```

It validates the layout (supported schema, BS order in every view, axis
order, frequency-grid length) and raises `ManifestError` (a `ValueError`)
otherwise. Artifact paths come back as absolute paths under the dataset
directory. Schema-v2 datasets (single BS) are read as `B = 1`: the BS is
built from `config.tx_position`/`tx_look_at` as `bs_000`, the per-view BS
fields and derived images become that BS's entry, and `load_aperture_cfr`
adds the leading `bs` axis. Unknown extra keys are ignored.

## Observed receiver impairments (`rf-camera-observe`)

`rf-camera-observe` turns the ideal, two-hemisphere, multi-BS aperture CFR into
a single-channel *observed* CFR per `(view, BS)` pair, as a CPU (NumPy-only)
post-processing step. It applies, in this fixed order:

1. `front + g * back`, collapsing the two hemispheres with the front-to-back
   gain `g` (`--front-to-back-db`, `None` = ideal front-only receiver);
2. a per-element complex gain error, constant over frequency;
3. a timing ramp `exp(-1j * 2*pi * f * tau)`, with `tau` the fixed
   `--timing-offset-ns` plus a per-link Gaussian extra;
4. a common phase `exp(1j * phi)`, either fixed (`--common-phase-deg`) or drawn
   uniformly per link (`--random-common-phase`);
5. circular complex AWGN.

The impairment split follows the physical entities: the per-element gain/phase
error is a property of the UE receive array and is drawn **once per view**,
shared by all of that view's BSs; the timing offset, common phase and noise
belong to each **`(UE, BS)` link** and are drawn independently per pair.

Noise is one **dataset-wide floor**, not one per view. `--snr-db X` is relative
to the dataset reference power `P_ref`, the maximum over all `(view, BS)` pairs
of the ideal isotropic mean power `mean(|front + back|^2)`; the resolved complex
noise variance is `P_ref / 10**(X/10)`. Path loss and front-to-back attenuation
therefore show up as a lower achieved SNR rather than as a lower noise floor.
`--noise-variance V` sets that variance directly instead (`--snr-db` and
`--noise-variance` are mutually exclusive). With neither flag no noise is added.
Noise is added whenever the variance is positive, even when a pair's signal is
exactly zero.

Use `--name` (default `observed`) to keep several variants in one dataset;
re-running with the same name overwrites just that variant. Each pair gets
`views/<view_id>/rf/<bs_id>/observed/<name>/aperture_cfr.npy`
(`complex64`, axis order `row, col, frequency_offset`) and an
`impairment_gt.json` (the applied `g`, element `gain`/`phase`, timing and common
phase, plus the pair's `ideal_isotropic_power`, `signal_power`,
`expected_snr_db` and `achieved_snr_db`). The pair's `artifacts` gains the keys
`observed.<name>.aperture_cfr` and `observed.<name>.impairment_gt`, and the
manifest records the variant under `observations[<name>]` (config, seed, RNG
streams, noise definition and per-pair SNRs). Datasets must be schema v3;
anything else raises `ManifestError` before writing.

On the mock dataset, run `make rf-camera-multiview-mock` first, then
`make rf-camera-observe-mock`.

## Delay sampling note

With bandwidth `B` and `N` frequency bins:

```text
delay resolution       = 1 / B
unambiguous delay range = N / B
```

For the mock defaults this is 10 ns resolution and 640 ns unambiguous delay.
The generator checks Sionna path GT and prints a warning if a path exceeds the
unambiguous range. Larger PLATEAU scenes will likely use more frequency bins.

## What to share after the first run

Please keep the console output, especially:

```text
Paths.cfr shape=...
[01/08] ue_000000: ...
...
[08/08] ue_000007: ...
```

Also share a few `views/*/rf/bs_*/angular_power_center.png` images, preferably views
from different sides of the ring. We want to verify that the front-hemisphere
camera and per-view pose rotations produce coherent but genuinely different
observations.

## Not implemented yet

- arbitrary view-list JSON input
- chunk/resume
- train/val/test split
- explicit 3DGS training target normalization
- receiver oscillator / CFO / phase-noise corruption
