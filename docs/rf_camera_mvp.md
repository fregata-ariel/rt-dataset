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

This is intentionally called an angular *spectrum* rather than a calibrated
AoA image. Mapping the FFT coordinates to a projection such as hemisphere or
equirectangular is a later image-formation step.

## Run

Switch to the feature branch and update the environment as usual:

```bash
git switch feature/1bs-1ue-rf-camera-mvp
git pull
```

Build the included mock CityJSON scene:

```bash
uv run python -m plateau_rt.cli.main build \
  data/raw/mock_building.city.json \
  data/generated/rf_camera_mvp
```

Then develop one RF-camera view:

```bash
uv run python -m plateau_rt.cli.main rf-camera \
  data/generated/rf_camera_mvp/mock_building.city.xml \
  data/generated/rf_camera_mvp/rf_camera
```

Useful overrides:

```bash
uv run python -m plateau_rt.cli.main rf-camera SCENE.xml OUT \
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

```text
rf_camera/
  aperture_cfr.npy
  angular_cfr.npy
  angular_power_center.png
  angular_phase_center.png
  path_gt.npz
  rf_camera_metadata.json
```

The expected Sionna CFR shape for the MVP is:

```text
[1, 64, 1, 1, 1, 64]
 ^   ^   ^  ^  ^   ^
 rx rxant tx txant t freq
```

The implementation fails loudly if Sionna returns a different shape. This is
intentional for the first hardware/software validation pass.

## What to share after the first run

Please keep the console output, especially these lines:

```text
Paths.cfr shape=...
aperture_cfr shape=...
angular_cfr shape=...
```

If it fails, share the traceback and the `Paths.cfr shape` line if printed.
If it succeeds, share `angular_power_center.png` and
`angular_phase_center.png` as well. Those are the first visual checks for
array ordering and coherent phase.

## CPU-only image-formation checks

The reshape and spatial FFT helpers have small tests that do not trace a scene:

```bash
uv run pytest tests/test_rf_camera_imaging.py -q
```

## Current limitations

- one BS and one UE only
- one active Tx port only; no RU beamforming yet
- single V polarization
- no receiver oscillator / CFO / phase-noise model
- no calibrated AoA projection yet
- no phase-valid mask for deep fades yet
- diffuse reflection is disabled
- Sionna synthetic-array accuracy is not yet characterized for this use case
- the debug phase PNG includes phase in low-power pixels and should not be used
  directly as a training target
