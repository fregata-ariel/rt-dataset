"""Optical ray renderer on a visual copy of a Sionna scene.

The RF-camera dataset pairs every RF view with an optical reference render
that lines up exactly with the RF view. This adapter is the low-level piece:
it takes arbitrary world-space rays (origins plus directions) and returns,
per ray, a Monte-Carlo RGB radiance, a hit flag, the hit range and the
surface normal. Radiance is sampled with a path integrator on a ``visual``
RGB copy of the Sionna scene, while geometry (hit flag, range, normal) comes
from deterministic ray intersections of the exact input rays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np

# Sionna-internal helpers, pinned to Sionna-RT 2.0.1. They are imported only
# in this adapter; domain code must stay NumPy-only.
from sionna.rt.renderer import visual_scene_from_wireless_scene
from sionna.rt.utils.render import make_render_sensor

CHUNK_SIZE = 2**18

_CHUNK_SEED_STRIDE = 0x9E3779B9
_UINT32_MOD = 2**32
_MIN_DIRECTION_NORM = 1e-12


@dataclass(frozen=True)
class RayRenderResult:
    """Per-ray optical render of a visual copy of a Sionna scene."""

    rgb: np.ndarray  # float32 [N, 3], linear radiance (mean over spp), exactly 0 where no hit
    hit: np.ndarray  # bool [N]
    range_m: np.ndarray  # float32 [N], |hit_point - origin| in metres, NaN where no hit
    normal_world: np.ndarray  # float32 [N, 3], geometric normal facing origin, NaN where no hit


class RayRenderer:
    """Render arbitrary world-space rays with Mitsuba on a visual scene copy.

    The constructor builds an RGB ``visual`` copy of the Sionna scene once;
    :meth:`render` then intersects explicit rays against it and samples
    radiance with the visual scene's path integrator.
    """

    def __init__(self, scene: Any, *, max_depth: int = 8, lighting_scale: float = 1.0) -> None:
        """Build the visual scene copy used by :meth:`render`.

        The Sionna scene is only read; the visual copy holds cloned meshes,
        so the input scene is left unchanged.
        """
        if isinstance(max_depth, bool) or not isinstance(max_depth, (int, np.integer)):
            raise ValueError(f"max_depth must be an int >= 1, got {max_depth!r}")
        if int(max_depth) < 1:
            raise ValueError(f"max_depth must be an int >= 1, got {max_depth!r}")
        if (
            isinstance(lighting_scale, bool)
            or not isinstance(lighting_scale, (int, float, np.integer, np.floating))
            or not np.isfinite(float(lighting_scale))
            or float(lighting_scale) <= 0.0
        ):
            raise ValueError(f"lighting_scale must be a finite float > 0, got {lighting_scale!r}")
        # Evaluated while Sionna's own variant is active, i.e. outside any
        # scoped block, mirroring sionna/rt/renderer.py.
        rendering_variant = (
            "cuda_ad_rgb" if dr.backend_v(mi.Float) == dr.JitBackend.CUDA else "llvm_ad_rgb"
        )
        with mi.util.scoped_set_variant(rendering_variant, "cuda_ad_rgb", "llvm_ad_rgb"):
            # Dummy sensor: required by visual_scene_from_wireless_scene but
            # not used by this ray renderer.
            sensor = make_render_sensor(
                scene, camera=mi.ScalarTransform4f(), resolution=(1, 1), fov=45.0
            )
            scene_dict = visual_scene_from_wireless_scene(
                scene,
                sensor=sensor,
                max_depth=int(max_depth),
                lighting_scale=float(lighting_scale),
            )
            self._visual = mi.load_dict(scene_dict)
            self._integrator = self._visual.integrator()
        self._variant = rendering_variant

    def render(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        spp: int = 64,
        seed: int = 0,
    ) -> RayRenderResult:
        """Render world-space rays, returning per-ray radiance and geometry.

        Directions are normalised internally. Rays are processed in chunks of
        at most :data:`CHUNK_SIZE` to bound GPU memory; chunk ``k`` (0-based)
        uses sampler seed ``(seed + k * 0x9E3779B9) % 2**32``, so chunk 0 uses
        exactly ``seed``. The same inputs and seed give bit-identical output.
        """
        origin_array = np.asarray(origins, dtype=np.float64)
        direction_array = np.asarray(directions, dtype=np.float64)
        if origin_array.ndim != 2 or origin_array.shape[1] != 3:
            raise ValueError(f"origins must have shape [N, 3], got {origin_array.shape}")
        if direction_array.ndim != 2 or direction_array.shape[1] != 3:
            raise ValueError(f"directions must have shape [N, 3], got {direction_array.shape}")
        if origin_array.shape[0] != direction_array.shape[0]:
            raise ValueError(
                "origins and directions must hold the same number of rays, "
                f"got {origin_array.shape[0]} and {direction_array.shape[0]}"
            )
        num_rays = origin_array.shape[0]
        if num_rays < 1:
            raise ValueError(f"at least one ray is required (N >= 1), got N={num_rays}")
        if not np.all(np.isfinite(origin_array)):
            raise ValueError("origins must be finite")
        if not np.all(np.isfinite(direction_array)):
            raise ValueError("directions must be finite")
        lengths = np.linalg.norm(direction_array, axis=1)
        if np.any(lengths <= _MIN_DIRECTION_NORM):
            raise ValueError("directions must have non-zero length")
        if isinstance(spp, bool) or not isinstance(spp, (int, np.integer)):
            raise ValueError(f"spp must be an int >= 1, got {spp!r}")
        spp = int(spp)
        if spp < 1:
            raise ValueError(f"spp must be an int >= 1, got {spp!r}")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be an int with 0 <= seed < 2**32, got {seed!r}")
        seed = int(seed)
        if not 0 <= seed < _UINT32_MOD:
            raise ValueError(f"seed must be an int with 0 <= seed < 2**32, got {seed!r}")

        unit_directions = direction_array / lengths[:, None]

        rgb_parts: list[np.ndarray] = []
        hit_parts: list[np.ndarray] = []
        range_parts: list[np.ndarray] = []
        normal_parts: list[np.ndarray] = []
        for chunk_index, start in enumerate(range(0, num_rays, CHUNK_SIZE)):
            stop = min(start + CHUNK_SIZE, num_rays)
            chunk_seed = (seed + chunk_index * _CHUNK_SEED_STRIDE) % _UINT32_MOD
            rgb, hit, range_m, normal_world = self._render_chunk(
                origin_array[start:stop],
                unit_directions[start:stop],
                spp=spp,
                seed=chunk_seed,
            )
            rgb_parts.append(rgb)
            hit_parts.append(hit)
            range_parts.append(range_m)
            normal_parts.append(normal_world)
        return RayRenderResult(
            rgb=np.ascontiguousarray(np.concatenate(rgb_parts, axis=0), dtype=np.float32),
            hit=np.ascontiguousarray(np.concatenate(hit_parts, axis=0), dtype=bool),
            range_m=np.ascontiguousarray(np.concatenate(range_parts, axis=0), dtype=np.float32),
            normal_world=np.ascontiguousarray(
                np.concatenate(normal_parts, axis=0), dtype=np.float32
            ),
        )

    def _render_chunk(
        self, origins: np.ndarray, directions: np.ndarray, *, spp: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Render one chunk of rays with unit directions.

        Returns ``(rgb, hit, range_m, normal_world)`` with the dtypes and
        shapes documented in :class:`RayRenderResult`.
        """
        num_rays = origins.shape[0]
        with mi.util.scoped_set_variant(self._variant, "cuda_ad_rgb", "llvm_ad_rgb"):
            ray = mi.Ray3f(
                mi.Point3f(origins[:, 0], origins[:, 1], origins[:, 2]),
                mi.Vector3f(directions[:, 0], directions[:, 1], directions[:, 2]),
            )
            si = self._visual.ray_intersect(ray)
            hit = si.is_valid().numpy().astype(bool)
            # Dr.Jit 3-vector .numpy() has shape (3, N); transpose to [N, 3].
            # Misses come back as 0 and are overwritten with NaN/0 below.
            points = si.p.numpy().T.astype(np.float64)
            normals = si.n.numpy().T.astype(np.float64)

            range_m = np.linalg.norm(points - origins, axis=1)
            range_m[~hit] = np.nan

            # Geometric normal (si.n), flipped to face the ray origin.
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            unit = normals / np.where(lengths > 0.0, lengths, 1.0)
            facing_away = np.sum(unit * directions, axis=1) > 0.0
            unit[facing_away] *= -1.0
            unit[~hit] = np.nan
            degenerate = hit & (lengths[:, 0] <= 0.0)
            unit[degenerate] = np.nan

            sampler = mi.load_dict({"type": "independent"})
            sampler.seed(seed, num_rays)
            ray_differential = mi.RayDifferential3f(ray)
            acc = mi.Color3f(0.0)
            for _ in range(spp):
                radiance, _valid, _aov = self._integrator.sample(
                    self._visual, sampler, ray_differential
                )
                acc += radiance
                sampler.advance()
                # Evaluate the RNG state every iteration; otherwise the
                # traced graph grows and runtime explodes quadratically.
                sampler.schedule_state()
                dr.eval(acc)
            rgb = (acc / spp).numpy().T.astype(np.float32)
            rgb[~hit] = 0.0

        return (
            np.ascontiguousarray(rgb, dtype=np.float32),
            np.ascontiguousarray(hit, dtype=bool),
            np.ascontiguousarray(range_m, dtype=np.float32),
            np.ascontiguousarray(unit, dtype=np.float32),
        )
