# 1 BS / multi-UE RF Camera Dataset

This milestone extends the validated 1-BS / 1-UE RF camera into a multi-view
dataset while keeping the observation model image-like and compact.

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

The included mock building occupies `[0,10] x [0,10] x [0,10]` m. The smoke
test uses:

- target / look-at: `(5, 5, 5)` m
- 8 UE views on a 30 m radius ring
- UE height: 1.5 m
- one BS at `(-50, -50, 30)` m
- BS panel also looks at `(5, 5, 5)` m
- 8 x 8 Rx aperture, 0.5 lambda spacing
- 3.5 GHz carrier
- 100 MHz bandwidth
- 64 uniformly spaced baseband frequency bins

All eight receivers are solved in **one** Sionna `PathSolver` call.

The mock has a single building and no ground, so each ring view receives only
the direct BS path. Each view therefore has energy in one hemisphere only, and
the views with the BS behind them (`ue_000004` to `ue_000006`) have empty
front images. Richer scenes are tracked in #7.

## Run

```bash
make rf-camera-multiview-mock
```

or, for another scene or ring:

```bash
PYTHONPATH=./src uv run python -m plateau_rt.cli.main rf-camera-multiview SCENE.xml OUT \
  --num-views 8 --radius-m 30 --ue-height-m 1.5 --target 5 5 5 \
  --bs-position -50 -50 30 --frequency-bins 64 --bandwidth-mhz 100
```

Poses and the camera model are in `src/plateau_rt/domain/rf_camera/camera.py`
(NumPy only); Sionna tracing is in
`src/plateau_rt/adapters/sionna/rf_camera_dataset.py`.

CPU-only geometry/ray tests (no Sionna import):

```bash
PYTHONPATH=./src uv run pytest tests -q
```

## Expected Sionna CFR shape

For the default 8-view mock:

```text
(8, 128, 1, 1, 1, 64)
 ^   ^    ^  ^  ^   ^
 UE rxant tx txant t freq
```

`rxant = 2 x 64`: Sionna fuses the pattern axis pattern-major, so channels
`0..63` are the front hemisphere and `64..127` the back hemisphere (each in
PlanarArray column-first order).

## Output layout

```text
rf_camera_multiview/
  dataset_manifest.json
  camera_model.npz
  path_geometry_gt.npz
  views/
    ue_000000/
      pose.json
      rf/
        aperture_cfr.npy
        angular_cfr_center.npy
        angular_power_center.npy
        phase_valid_mask.npy
        dominant_delay_s.npy
        dominant_delay_power.npy
        angular_power_center.png
    ue_000001/
      ...
```

### Canonical vs derived data

`aperture_cfr.npy` is the canonical compact RF observation,
`[hemisphere, row, col, frequency]` with hemispheres `(front, back)`. It
preserves the complex CFR on the physical UE aperture over frequency.

The following files are derived from the front hemisphere (as `A = kx * U`)
and can be regenerated from the aperture CFR:

- center-frequency calibrated angular complex image
- center-frequency power image
- phase-valid mask
- dominant delay map
- dominant-delay power

The full `[kz, ky, frequency]` or `[kz, ky, delay]` volume is intentionally not
stored for every production view. This avoids a large storage multiplier while
keeping all information needed to regenerate it.

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

## Manifest (schema version 2)

Besides the configuration and frequency grid, `dataset_manifest.json` records

- `raw_observation`: axis order and hemisphere names of `aperture_cfr`, and the
  Rx element pattern;
- `camera_model`: projection, developed hemisphere and the image quantity
  definition;
- per view: pose, `bs_direction_local`, `bs_in_front_hemisphere` and
  `hemisphere_energy` (sum of `|aperture_cfr|^2` per hemisphere), so views with
  back-hemisphere energy can be found without loading the arrays.

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

Also share a few `views/*/rf/angular_power_center.png` images, preferably views
from different sides of the ring. We want to verify that the front-hemisphere
camera and per-view pose rotations produce coherent but genuinely different
observations.

## Not implemented yet

- arbitrary view-list JSON input
- 2+ BS illumination axes
- chunk/resume
- train/val/test split
- optical co-registered reference render (tracked in issue #11)
- explicit 3DGS training target normalization
- receiver oscillator / CFO / phase-noise corruption
