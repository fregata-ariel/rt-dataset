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

The 2-D planar aperture alone has a front/back ambiguity. For the multi-view
camera dataset the Rx element pattern is therefore changed from the symmetric
`dipole` used in the first validation experiment to the directional
`tr38901` pattern. Its boresight is local +x and its back hemisphere is strongly
attenuated. This makes the +x hemisphere the camera convention, but does not
mathematically remove all back-hemisphere energy.

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

## Run

```bash
git switch feature/1bs-multiue-rf-camera-dataset
git pull
make rf-camera-multiview-mock
```

CPU-only geometry/ray tests:

```bash
PYTHONPATH=./src uv run pytest \
  tests/test_rf_camera_imaging.py \
  tests/test_rf_camera_calibration.py \
  tests/test_rf_camera_delay.py \
  tests/test_rf_camera_dataset.py -q
```

## Expected Sionna CFR shape

For the default 8-view mock:

```text
(8, 64, 1, 1, 1, 64)
 ^   ^   ^  ^  ^   ^
 UE rxant tx txant t freq
```

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

`aperture_cfr.npy` is the canonical compact RF observation. It preserves the
complex CFR on the physical UE aperture over frequency.

The following files are derived and can be regenerated from the aperture CFR:

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
- `ky_over_k[W]`
- `kz_over_k[H]`

For each view, `pose.json` contains the UE position and the
`world_from_local_rotation` matrix. A local RF-camera ray can therefore be
mapped to world coordinates by

```text
ray_world = world_from_local_rotation @ ray_local
```

This is the intended bridge to a Gaussian-Splatting camera model.

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
from different sides of the ring. We want to verify that the directional
`tr38901` camera and per-view pose rotations produce coherent but genuinely
different observations.

## Not implemented yet

- arbitrary view-list JSON input
- 2+ BS illumination axes
- chunk/resume
- train/val/test split
- optical co-registered reference render (tracked in issue #11)
- explicit 3DGS training target normalization
- receiver oscillator / CFO / phase-noise corruption
