"""Render co-registered optical reference images for an RF-camera dataset.

For every view of an existing ``rf-camera-multiview`` output directory, this
module renders two optical reference products with
:class:`plateau_rt.adapters.sionna.optical_render.RayRenderer` (or any
duck-typed ``renderer`` object): a pinhole photo/depth pair for a parallel
Gaussian-Splatting dataset, and a render on the RF direction-cosine grid for
pixel-aligned overlay with the RF debug images.

These are **reference artifacts for debugging and for a parallel optical
dataset -- not an RF training target.** The RF observation stays the aperture
CFR; nothing here feeds back into it.

This module must stay importable without Sionna installed: the default
renderer is built with a lazy import inside :func:`render_optical_references`,
never at module scope (see ``tests/test_rf_camera_boundaries.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from plateau_rt.application.rf_dataset_manifest import (
    RFDatasetManifest,
    load_rf_dataset_manifest,
)
from plateau_rt.domain.rf_camera.optical import (
    PinholeIntrinsics,
    camera_to_world_opengl,
    local_to_world_rays,
    nerf_transforms,
    pinhole_ray_directions_local,
    to_rgba8,
    z_depth_from_range,
)

RENDERER_DESCRIPTION = (
    "Mitsuba path tracer on Sionna's visual scene copy (RayRenderer): "
    "constant white environment light, diffuse radio-material colours"
)

TRANSFORMS_FILE_NAME = "transforms.json"


def render_optical_references(
    dataset_dir: Path,
    *,
    renderer: Any | None = None,
    scene_xml: Path | None = None,
    width: int = 512,
    height: int = 512,
    fov_x_deg: float = 90.0,
    spp: int = 64,
    seed: int = 0,
) -> Path:
    """Render pinhole + hemisphere optical references for every dataset view.

    ``dataset_dir`` is an existing ``rf-camera-multiview`` output directory
    (``dataset_manifest.json``, ``camera_model.npz``, ``views/<id>/pose.json``).
    The manifest is read with
    :func:`plateau_rt.application.rf_dataset_manifest.load_rf_dataset_manifest`,
    so schema v3 (multi-BS) and v2 (single-BS) datasets are accepted and any
    other schema raises :class:`~plateau_rt.application.rf_dataset_manifest.ManifestError`
    before anything is rendered or written. The render depends only on each
    view's pose, so every view gets one render however many BSs it has.
    ``renderer`` is any object exposing ``.render(origins, directions, spp=,
    seed=)`` returning an object with ``rgb``/``hit``/``range_m`` attributes
    (see :class:`plateau_rt.adapters.sionna.optical_render.RayRenderResult`);
    tests pass a fake renderer here. When ``renderer`` is ``None``, a
    :class:`~plateau_rt.adapters.sionna.optical_render.RayRenderer` is built
    from ``scene_xml`` (defaulting to the manifest's ``source_scene``).

    Writes, for every view, a pinhole RGBA PNG + z-depth + range (see
    :mod:`plateau_rt.domain.rf_camera.optical` for the pixel convention) and a
    hemisphere RGBA PNG + range on the RF direction-cosine grid of
    ``camera_model.npz``. Writes ``transforms.json`` at the dataset root and
    updates ``dataset_manifest.json`` in place (adding ``optical_reference``
    and view-level artifact paths under ``views[].artifacts``, never under the
    per-BS ``views[].bs[]`` entries). Re-running overwrites all of the above
    cleanly. Returns the path of ``transforms.json``.
    """
    dataset_dir = Path(dataset_dir)
    dataset = load_rf_dataset_manifest(dataset_dir)
    # The typed reader validates the layout; the raw dict (the reader's own,
    # not a copy) is what gets updated and written back so unknown keys survive.
    manifest = dataset.raw
    manifest_path = dataset.manifest_path

    if renderer is None:
        renderer = _build_default_renderer(dataset, scene_xml)

    intrinsics = PinholeIntrinsics.from_horizontal_fov(int(width), int(height), float(fov_x_deg))
    pinhole_dirs_local = pinhole_ray_directions_local(intrinsics)

    camera_model = np.load(dataset.camera_model_path)
    valid_mask = camera_model["valid_mask"]
    hemisphere_dirs_local = camera_model["ray_directions_local"]
    fft_rows, fft_cols = valid_mask.shape

    pinhole_total = int(pinhole_dirs_local.shape[0] * pinhole_dirs_local.shape[1])
    hemisphere_total = int(np.count_nonzero(valid_mask))

    frames: list[dict[str, Any]] = []
    for dataset_view in dataset.views:
        view_index = dataset_view.index
        view_id = dataset_view.view_id
        view = manifest["views"][view_index]
        view_dir = dataset_dir / "views" / view_id
        pose = json.loads(dataset_view.pose_path.read_text(encoding="utf-8"))
        rotation = np.asarray(pose["world_from_local_rotation"], dtype=np.float64)
        position = np.asarray(pose["position_m"], dtype=np.float64)

        optical_dir = view_dir / "optical"
        optical_dir.mkdir(parents=True, exist_ok=True)

        pinhole_hits, pinhole_paths = _render_pinhole(
            optical_dir,
            renderer,
            pinhole_dirs_local,
            rotation,
            position,
            spp=spp,
            seed=seed,
        )
        frames.append(
            {
                "file_path": str(pinhole_paths["rgba"].relative_to(dataset_dir)),
                "depth_file_path": str(pinhole_paths["depth"].relative_to(dataset_dir)),
                "camera_to_world": camera_to_world_opengl(rotation, position),
            }
        )

        hemisphere_hits, hemisphere_paths = _render_hemisphere(
            optical_dir,
            renderer,
            hemisphere_dirs_local,
            valid_mask,
            rotation,
            position,
            spp=spp,
            seed=seed,
        )

        view.setdefault("artifacts", {})
        view["artifacts"]["optical_pinhole_rgba"] = str(
            pinhole_paths["rgba"].relative_to(dataset_dir)
        )
        view["artifacts"]["optical_pinhole_depth_m"] = str(
            pinhole_paths["depth"].relative_to(dataset_dir)
        )
        view["artifacts"]["optical_pinhole_range_m"] = str(
            pinhole_paths["range"].relative_to(dataset_dir)
        )
        view["artifacts"]["optical_hemisphere_rgba"] = str(
            hemisphere_paths["rgba"].relative_to(dataset_dir)
        )
        view["artifacts"]["optical_hemisphere_range_m"] = str(
            hemisphere_paths["range"].relative_to(dataset_dir)
        )

        print(
            f"  [{view_index + 1:02d}/{dataset.num_views:02d}] {view_id}: "
            f"pinhole hits={pinhole_hits}/{pinhole_total}, "
            f"hemisphere hits={hemisphere_hits}/{hemisphere_total}"
        )

    transforms = nerf_transforms(intrinsics, frames)
    transforms_path = dataset_dir / TRANSFORMS_FILE_NAME
    transforms_path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")

    manifest["optical_reference"] = {
        "purpose": (
            "Reference optical renders for debugging and for a parallel optical "
            "Gaussian-Splatting dataset; NOT an RF training target."
        ),
        "renderer": RENDERER_DESCRIPTION,
        "spp": int(spp),
        "seed": int(seed),
        "pinhole": {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fov_x_deg": float(fov_x_deg),
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "image_orientation": (
                "standard photo orientation: row 0 = top (+z), column 0 = camera left (+y)"
            ),
            "depth_definition": (
                "z-depth along the optical axis (local +x), metres; 0.0 where no "
                "hit (nerfstudio convention for missing depth)"
            ),
            "range_definition": (
                "distance from the camera position to the hit point, metres; NaN where no hit"
            ),
            "transforms": TRANSFORMS_FILE_NAME,
        },
        "hemisphere": {
            "grid": (
                "RF direction-cosine grid of camera_model.npz (front hemisphere), "
                "pixel-aligned with the RF arrays: row = kz index increasing "
                "upward in kz, column = ky index increasing toward +ky"
            ),
            "png_orientation": (
                "hemisphere_rgba.png rows are flipped (np.flipud) so +kz is at "
                "the top, keeping columns in +ky order, to match the RF debug "
                "PNGs (matplotlib origin='lower'); the underlying .npy array is "
                "NOT flipped and keeps the RF grid row/column order. Because +ky "
                "is the camera's LEFT, the RF/hemisphere images are left-right "
                "mirrored relative to the pinhole photo."
            ),
            "range_definition": (
                "distance from the camera position to the hit point along each "
                "direction-cosine ray, metres; NaN where no hit or outside "
                "valid_mask"
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"transforms: {transforms_path}")
    print(f"manifest updated: {manifest_path}")
    return transforms_path


def _render_pinhole(
    optical_dir: Path,
    renderer: Any,
    pinhole_dirs_local: np.ndarray,
    rotation: np.ndarray,
    position: np.ndarray,
    *,
    spp: int,
    seed: int,
) -> tuple[int, dict[str, Path]]:
    """Render, and save, one view's pinhole RGBA/depth/range. Returns (hits, paths)."""
    height, width = pinhole_dirs_local.shape[:2]
    origins, directions = local_to_world_rays(pinhole_dirs_local, rotation, position)
    result = renderer.render(origins.reshape(-1, 3), directions.reshape(-1, 3), spp=spp, seed=seed)

    hit = np.asarray(result.hit, dtype=bool).reshape(height, width)
    rgb = np.asarray(result.rgb, dtype=np.float64).reshape(height, width, 3)
    range_m = np.asarray(result.range_m, dtype=np.float64).reshape(height, width)

    rgba = to_rgba8(rgb, hit.astype(np.float64))
    depth = z_depth_from_range(range_m, pinhole_dirs_local)
    depth = np.where(hit, depth, 0.0).astype(np.float32)

    paths = {
        "rgba": optical_dir / "pinhole_rgba.png",
        "depth": optical_dir / "pinhole_depth_m.npy",
        "range": optical_dir / "pinhole_range_m.npy",
    }
    _save_rgba_png(paths["rgba"], rgba)
    np.save(paths["depth"], depth)
    np.save(paths["range"], range_m.astype(np.float32))
    return int(np.count_nonzero(hit)), paths


def _render_hemisphere(
    optical_dir: Path,
    renderer: Any,
    hemisphere_dirs_local: np.ndarray,
    valid_mask: np.ndarray,
    rotation: np.ndarray,
    position: np.ndarray,
    *,
    spp: int,
    seed: int,
) -> tuple[int, dict[str, Path]]:
    """Render, and save, one view's hemisphere RGBA/range. Returns (hits, paths)."""
    fft_rows, fft_cols = valid_mask.shape
    valid_dirs_local = hemisphere_dirs_local[valid_mask]
    origins, directions = local_to_world_rays(valid_dirs_local, rotation, position)
    result = renderer.render(origins, directions, spp=spp, seed=seed)

    hit_valid = np.asarray(result.hit, dtype=bool)
    hit = np.zeros((fft_rows, fft_cols), dtype=bool)
    hit[valid_mask] = hit_valid

    rgb = np.zeros((fft_rows, fft_cols, 3), dtype=np.float64)
    rgb[valid_mask] = np.asarray(result.rgb, dtype=np.float64)

    range_m = np.full((fft_rows, fft_cols), np.nan, dtype=np.float32)
    range_m[valid_mask] = np.asarray(result.range_m, dtype=np.float32)

    # alpha = hit & valid: `hit` is already False outside valid_mask.
    rgba = to_rgba8(rgb, hit.astype(np.float64))
    # PNG only: flip rows so +kz is at the top, matching the RF debug PNGs
    # (matplotlib origin="lower"). The .npy range stays in RF grid order.
    rgba_png = np.flipud(rgba)

    paths = {
        "rgba": optical_dir / "hemisphere_rgba.png",
        "range": optical_dir / "hemisphere_range_m.npy",
    }
    _save_rgba_png(paths["rgba"], rgba_png)
    np.save(paths["range"], range_m)
    return int(np.count_nonzero(hit_valid)), paths


def _save_rgba_png(path: Path, rgba: np.ndarray) -> None:
    """Write an 8-bit RGBA array as a PNG, row 0 = top (standard image order)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(path, rgba)


def _build_default_renderer(dataset: RFDatasetManifest, scene_xml: Path | None) -> Any:
    """Build a :class:`RayRenderer` from ``scene_xml`` or the manifest's scene."""
    # Lazy imports: this module must stay importable without Sionna/Mitsuba.
    from sionna.rt import load_scene

    from plateau_rt.adapters.sionna.optical_render import RayRenderer

    if scene_xml is not None:
        path = Path(scene_xml)
    else:
        source_scene = dataset.source_scene
        if not source_scene:
            raise FileNotFoundError(
                "dataset_manifest.json has no 'source_scene'; pass --scene-xml explicitly"
            )
        path = Path(source_scene)

    if not path.is_absolute() and not path.exists():
        raise FileNotFoundError(
            f"scene XML not found: {path} (relative paths are resolved against the "
            "current working directory, typically the repository root); pass "
            "--scene-xml to point at it explicitly"
        )

    scene = load_scene(str(path))
    return RayRenderer(scene)
