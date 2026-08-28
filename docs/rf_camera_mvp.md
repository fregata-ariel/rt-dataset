# 1 BS / 1 UE RF Camera MVP

This milestone develops one coherent RF-camera image from a Sionna-RT scene.
It deliberately stops before Gaussian Splatting.

## Observation model

- one BS / RU illumination
- one UE pose
- one active Tx antenna/port for the MVP
- 8 x 8 receive planar aperture by default
- 0.5 lambda antenna spacing
- 3.5 GHz carrier by default
- 100 MHz baseband frequency grid by default
- ideal coherent Sionna phase

The primary raw observation is

```text
aperture_cfr[row, col, frequency_offset] : complex64
```

The first developed image is a zero-padded 2-D spatial FFT of the receive
aperture:

```text
angular_cfr[vertical_spatial_frequency,
            horizontal_spatial_frequency,
            frequency_offset] : complex64
```

A CPU-only calibration then converts this matrix FFT into a physically oriented
UE-local field:

```text
angular_cfr_calibrated[kz_over_k, ky_over_k, frequency_offset] : complex64
```

The calibration restores the phase origin to the aperture/UE center and corrects
Sionna PlanarArray's row direction so +kz points toward local +z.

Finally, an IFFT along the uniformly sampled frequency axis produces:

```text
angular_delay_cir[kz_over_k, ky_over_k, delay] : complex64
```

This angle-delay tensor is the first representation intended to behave like an
RF image with delay bins as channels. The planar y-z aperture still has a
front/back ambiguity in the sign of local kx.

## Run

Switch to the feature branch:

```bash
git switch feature/1bs-1ue-rf-camera-mvp
git pull
```

The shortest path-tracing smoke test uses the included mock scene:

```bash
make rf-camera-mock
```

This first builds `data/raw/mock_building.city.json`, then develops one
RF-camera view into:

```text
data/generated/mock_results/rf_camera/
```

Calibrate the angular field without re-running Sionna:

```bash
make rf-camera-calibrate-mock
```

Develop the calibrated CFR into the angle-delay volume, also without re-running
Sionna:

```bash
make rf-camera-delay-mock
```

### Run the path-tracing stage manually

This repository currently uses `PYTHONPATH=./src` for its CLI entry point.
Build the included mock CityJSON scene:

```bash
PYTHONPATH=./src uv run python -m plateau_rt.cli.main build \
  data/raw/mock_building.city.json \
  data/generated/rf_camera_mvp
```

Then develop one RF-camera view:

```bash
PYTHONPATH=./src uv run python -m plateau_rt.cli.main rf-camera \
  data/generated/rf_camera_mvp/mock_building.city.xml \
  data/generated/rf_camera_mvp/rf_camera
```

Useful overrides:

```bash
PYTHONPATH=./src uv run python -m plateau_rt.cli.main rf-camera SCENE.xml OUT \
  --ue-position 0 0 1.5 \
  --ue-orientation 0 0 0 \
  --bs-position -50 -50 30 \
  --rx-rows 8 \
  --rx-cols 8 \
  --carrier-ghz 3.5 \
  --bandwidth-mhz 100 \
  --frequency-bins 64 \
  --max-depth 5 \
  --synthetic-array
```

For a high-fidelity comparison, replace `--synthetic-array` with
`--explicit-array`. The explicit mode will be substantially more expensive.

## Expected output

After all three MVP stages:

```text
rf_camera/
  aperture_cfr.npy
  angular_cfr.npy
  angular_power_center.png
  angular_phase_center.png
  path_gt.npz
  rf_camera_metadata.json

  angular_cfr_calibrated.npy
  angular_power_center_calibrated.png
  angular_phase_center_calibrated.png
  angular_calibration.json

  angular_delay_cir.npy
  delay_axis_s.npy
  propagating_direction_mask.npy
  angular_power_strongest_delay.png
  delay_profile_los_direction.png
  dominant_delay_map.png
  angle_delay_report.json
```

The expected Sionna CFR shape for the default MVP is:

```text
[1, 64, 1, 1, 1, 64]
 ^   ^   ^  ^  ^   ^
 rx rxant tx txant t freq
```

The implementation fails loudly if Sionna returns a different shape. This is
intentional for the first hardware/software validation pass.

For the default 100 MHz / 64-bin frequency grid:

- frequency spacing: 1.5625 MHz
- delay resolution: 10 ns
- unambiguous delay period: 640 ns

The saved complex angle-delay target uses the rectangular sampled band as-is.
No delay window is applied to the target; visualization sidelobes are therefore
expected.

## Validation targets

The default mock geometry has a direct BS-to-UE path. The angular calibration
reports the strongest center-frequency bin and compares its y-z projection with
the geometric LoS source direction.

The angle-delay stage also compares the nearest-LoS angular-bin delay profile
with the geometric propagation delay. Quantization on the 10 ns delay grid is
expected.

The square spatial FFT contains samples outside the physical far-field
projection disk. `propagating_direction_mask.npy` marks samples satisfying:

```text
(ky/k)^2 + (kz/k)^2 <= 1
```

The remaining sign of kx is not observable from a single planar aperture.

## Image-formation checks

All calibration and delay-development tests are CPU-only:

```bash
PYTHONPATH=./src uv run pytest \
  tests/test_rf_camera_imaging.py \
  tests/test_rf_camera_calibration.py \
  tests/test_rf_camera_delay.py -q
```

## Optical reference images

A future dataset stage will optionally render a conventional optical
ray-tracing reference for the same RF view/pose. This reference is intended for
geometry/pose/material debugging and possible multi-modal experiments; it is
not the RF training target. Pixel-level co-registration will be defined after
the final RF projection is chosen.

## Current limitations

- one BS and one UE only
- one active Tx port only; no RU beamforming yet
- single V polarization
- no receiver oscillator / CFO / phase-noise model
- direction-cosine angular grid only; no perspective/equirectangular RF projection yet
- planar aperture has front/back ambiguity in local kx
- diffuse reflection is disabled
- Sionna synthetic-array accuracy is not yet characterized for this use case
- calibrated phase visualization masks low-power pixels, but training-target masking/loss is not yet defined
- rectangular frequency sampling produces delay sidelobes; alternative analysis windows are diagnostic-only for now
