# Optical Reference Renders (issue #11)

`rf-camera-optical` adds a co-registered optical render to every view of an
existing `rf-camera-multiview` dataset: a pinhole photo/depth pair for a
parallel Gaussian-Splatting dataset, and a render on the RF direction-cosine
grid for a pixel-aligned overlay with the RF debug images.

## Not an RF training target

These renders are **reference artifacts for debugging and for a parallel
optical Gaussian-Splatting dataset only.** The canonical RF observation stays
the receive-aperture CFR (`aperture_cfr.npy`); nothing here feeds back into it,
and no RF stage reads the optical outputs. The manifest records this
explicitly under `optical_reference.purpose`.

## Outputs

Run:

```bash
make rf-camera-optical-mock
```

or, on any `rf-camera-multiview` dataset:

```bash
PYTHONPATH=./src uv run python -m plateau_rt.cli.main rf-camera-optical DATASET_DIR \
  [--scene-xml SCENE.xml] [--width 512] [--height 512] [--fov-x-deg 90] \
  [--spp 64] [--seed 0]
```

`--scene-xml` defaults to the dataset manifest's `source_scene`, which
`rf-camera-multiview` stores as given on its own command line (typically
relative to the repository root, since that is how the `make` targets invoke
it). Pass `--scene-xml` explicitly when running from another working
directory, or the scene has moved.

This adds, per view, under `views/<id>/optical/`:

| file | contents |
|---|---|
| `pinhole_rgba.png` | 8-bit RGBA pinhole photo, alpha = hit |
| `pinhole_depth_m.npy` | `float32 [H, W]` z-depth along the optical axis, `0.0` where no hit |
| `pinhole_range_m.npy` | `float32 [H, W]` distance from the camera position, `NaN` where no hit |
| `hemisphere_rgba.png` | 8-bit RGBA render on the RF direction-cosine grid, rows flipped for display |
| `hemisphere_range_m.npy` | `float32 [fft_rows, fft_cols]` distance from the camera position, `NaN` where no hit or outside `valid_mask` |

and, at the dataset root:

- `transforms.json`: a `transforms.json` in nerfstudio's own
  `nerfstudio-data` format covering every view's pinhole frame (see below).
  It is **not** the classic Blender/NeRF-synthetic layout.
- `dataset_manifest.json` gains a top-level `optical_reference` block
  (renderer, `spp`, `seed`, the pinhole intrinsics/orientation/definitions,
  and the hemisphere grid/orientation/definitions) and, per view, five new
  `artifacts` entries: `optical_pinhole_rgba`, `optical_pinhole_depth_m`,
  `optical_pinhole_range_m`, `optical_hemisphere_rgba`,
  `optical_hemisphere_range_m`.

Re-running `rf-camera-optical` overwrites all of the above cleanly
(idempotent): it never needs the RF tracing stage to be re-run.

## Conventions

### Camera-local frame

Every RF view and its optical render share the same camera-local frame used
throughout this project: **x = forward, y = left, z = up**, with
`world_from_local = pose.json["world_from_local_rotation"]`. The receive
aperture and the pinhole image plane both lie in the local y-z plane.

### Pinhole pixel model

`PinholeIntrinsics.from_horizontal_fov(width, height, fov_x_deg)` builds a
pinhole camera with square pixels and

```text
fx = fy = (width / 2) / tan(fov_x_deg / 2 in radians)
cx = width / 2, cy = height / 2
```

Pixel `(row r, col c)` has its centre at `(c + 0.5, r + 0.5)`; the centre of
that pixel maps to the camera-local direction proportional to
`(1, -right, up)` with

```text
right = (c + 0.5 - cx) / fx
up    = (cy - (r + 0.5)) / fy
```

**Row 0 is the top of the image (+z). Column 0 is the left (+y).** This is
standard photo orientation and was verified against Mitsuba's perspective
sensor as built by Sionna (see
`src/plateau_rt/domain/rf_camera/optical.py`).

### `transforms.json`: OpenGL/NeRF convention

`camera_to_world_opengl` maps the OpenGL/NeRF camera convention (camera looks
along `-Z`, `+Y` up, `+X` right) onto the RF camera-local axes, producing the
4x4 `camera_to_world` (`transform_matrix`) that nerfstudio's
`nerfstudio-data` parser expects (the same OpenGL axis convention the
original NeRF datasets use). The translation column is the view position; `camera_model` is
`"OPENCV"` in name only (nerfstudio's generic pinhole convention) -- there is
no lens distortion (`k1=k2=p1=p2=0`).

### z-depth vs range

- **z-depth** (`pinhole_depth_m.npy`, and `depth_file_path` in
  `transforms.json`) is the distance projected onto the optical axis (local
  `+x`): `depth = range * dir_local_x`, where `dir_local_x` is the ray's
  camera-local x-component (the cosine to the boresight). This is the
  convention nerfstudio and most NeRF-style depth supervision expect, and it
  uses `0.0` for missing depth (the nerfstudio convention), not `NaN`.
- **range** (`pinhole_range_m.npy`, `hemisphere_range_m.npy`) is the plain
  Euclidean distance from the camera position to the hit point, with `NaN`
  where there is no hit (or, for the hemisphere, outside `valid_mask`).

### Hemisphere grid alignment (and the mirror)

The hemisphere render uses exactly the front-hemisphere ray grid of
`camera_model.npz` (`ray_directions_local[fft_rows, fft_cols, 3]`,
`valid_mask[fft_rows, fft_cols]`) built by
`build_direction_cosine_camera_model` -- the same grid the RF arrays
(`views/<id>/rf/angular_power_center.npy`, and its debug PNG) are indexed on:
**row = kz index, increasing upward in kz; column = ky index, increasing
toward +ky.**

`hemisphere_range_m.npy` keeps that exact indexing, so it is pixel-aligned
with the RF `.npy` arrays with no transform needed.

`hemisphere_rgba.png`, for display only, has its rows flipped (`np.flipud`)
so that +kz is at the top of the image, matching the RF debug PNGs (which
plot with `matplotlib`'s `origin="lower"`, so +kz also renders upward). Columns
stay in +ky order in both.

**Mirror note:** in the camera-local frame, +y is the camera's **left**. So
`ky` increasing to the right in the RF/hemisphere images means those images
are **left-right mirrored** relative to the pinhole photo (where the camera's
left, +y, is column 0, the photo's left). This is expected and matches how the
RF debug images have always been oriented; it is not a bug in the optical
renderer.

## Training a 3D Gaussian Splatting model

`transforms.json` follows nerfstudio's own `nerfstudio-data` transforms
format: a single file with top-level `camera_model`, `w`/`h`, `fl_x`/`fl_y`,
`cx`/`cy`, and per-frame `file_path` (with extension, relative to the dataset
root) + `transform_matrix` + `depth_file_path`. nerfstudio's default
dataparser (`nerfstudio-data`) consumes it directly:

```bash
ns-train splatfacto --data DATASET_DIR
```

Other tools that read the nerfstudio `transforms.json` format can use it the
same way.

**Not supported as-is:** the classic Blender/NeRF-synthetic layout (nerfstudio's
`blender-data` parser and most third-party "NeRF-synthetic" loaders). Those
loaders look for per-split `transforms_train.json` / `transforms_val.json` /
`transforms_test.json` (never a single `transforms.json`) and append `.png` to
each extension-less `file_path` themselves, so pointing them at this dataset
fails (no split files, and `pinhole_rgba.png` would become
`pinhole_rgba.png.png`). Using such a loader needs a separate export step that
writes the split files with extension-less `file_path`s (`camera_angle_x` is
already in `transforms.json`); none is provided here.

## Limitations

- The mock scene is a single grey box with one radio material; every surface
  renders with the same diffuse colour, so the optical images are not
  photorealistic and are only useful for geometric/pose sanity checks.
- There is no ground/terrain in the mock scene, so rays that miss the
  building simply miss (alpha 0), even when pointed downward.
- Lighting is a constant white environment (see
  `src/plateau_rt/adapters/sionna/optical_render.py`); there is no sun,
  shadow catcher or scene-specific lighting model.
- Radiance comes from a Monte-Carlo path tracer (`spp` samples per ray); very
  low `--spp` will be visibly noisy.
