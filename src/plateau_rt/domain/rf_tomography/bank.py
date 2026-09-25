"""Pose bank, splits and noise reference of the tomography dataset profile (NumPy only)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from plateau_rt.domain.rf_camera.camera import generate_ring_views, look_at_orientation
from plateau_rt.domain.rf_tomography import sync, views

SPLIT_STREAM_TAG: int = 0x53504C54
JITTER_STREAM_TAG: int = 0x4A495454
DEFAULT_VIEW_SUBSET_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class MechanismVariant:
    """One cumulative tracing-mechanism variant (the LoS is always traced)."""

    name: str
    specular_reflection: bool
    refraction: bool
    diffraction: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable mechanism flags (fixed keys and order)."""
        return {
            "name": self.name,
            "los": True,
            "specular_reflection": self.specular_reflection,
            "refraction": self.refraction,
            "diffraction": self.diffraction,
            "diffuse_reflection": False,
        }


MECHANISM_VARIANTS: dict[str, MechanismVariant] = {
    "specular": MechanismVariant("specular", True, False, False),
    "refraction": MechanismVariant("refraction", True, True, False),
    "diffraction": MechanismVariant("diffraction", True, True, True),
}


@dataclass(frozen=True)
class RingPose:
    """One candidate pose on a ring around the target."""

    position: tuple[float, float, float]
    radius_m: float
    height_m: float
    azimuth_deg: float
    ring_index: int


def _as_target(target: Sequence[float], *, name: str = "target") -> tuple[float, float, float]:
    """Return ``target`` as 3 finite Python floats, else raise ``ValueError``."""
    try:
        items = tuple(target)
    except TypeError:
        raise ValueError(f"{name} must be a length-3 sequence of finite numbers") from None
    if len(items) != 3:
        raise ValueError(f"{name} must be a length-3 sequence of finite numbers")
    try:
        point = (float(items[0]), float(items[1]), float(items[2]))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a length-3 sequence of finite numbers") from None
    if not all(math.isfinite(v) for v in point):
        raise ValueError(f"{name} must be finite, got {target!r}")
    return point


def ring_poses(
    target: Sequence[float],
    rings: Sequence[tuple[float, float]],
    per_ring: int,
    start_azimuth_deg: float = 0.0,
) -> list[RingPose]:
    """Return the ring poses in ``(ring, k)`` order (positions from ``generate_ring_views``)."""
    point = _as_target(target)
    try:
        ring_list = list(rings)
    except TypeError:
        raise ValueError("rings must be a non-empty sequence of (radius_m, height_m)") from None
    if len(ring_list) == 0:
        raise ValueError("rings must not be empty")
    if isinstance(per_ring, bool) or not isinstance(per_ring, (int, np.integer)):
        raise ValueError(f"per_ring must be an int >= 1, got {per_ring!r}")
    if int(per_ring) < 1:
        raise ValueError(f"per_ring must be >= 1, got {per_ring!r}")
    per_ring = int(per_ring)
    try:
        start = float(start_azimuth_deg)
    except (TypeError, ValueError):
        raise ValueError(f"start_azimuth_deg must be finite, got {start_azimuth_deg!r}") from None
    if not math.isfinite(start):
        raise ValueError(f"start_azimuth_deg must be finite, got {start_azimuth_deg!r}")
    parsed: list[tuple[float, float]] = []
    for entry in ring_list:
        try:
            radius, height = float(entry[0]), float(entry[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError(f"rings entries must be (radius_m, height_m), got {entry!r}") from None
        if not math.isfinite(radius) or not math.isfinite(height):
            raise ValueError(f"ring (radius_m, height_m) must be finite, got {entry!r}")
        if radius <= 0.0:
            raise ValueError(f"ring radius_m must be > 0, got {entry!r}")
        parsed.append((radius, height))
    poses: list[RingPose] = []
    for radius, height in parsed:
        ring_views = generate_ring_views(
            target=point,
            radius_m=radius,
            ue_height_m=height,
            num_views=per_ring,
            start_azimuth_deg=start,
        )
        for k in range(per_ring):
            px, py, pz = (float(v) for v in ring_views[k].position)
            poses.append(
                RingPose(
                    position=(px, py, pz),
                    radius_m=radius,
                    height_m=height,
                    azimuth_deg=start + 360.0 * k / per_ring,
                    ring_index=k,
                )
            )
    return poses


def jitter_look_at(
    positions: np.ndarray,
    target: Sequence[float],
    *,
    max_deg: float,
    rng: np.random.Generator | None,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]], np.ndarray]:
    """Return (look_ats, orientations, jitter_deg[K, 2]) around the target direction."""
    try:
        limit = float(max_deg)
    except (TypeError, ValueError):
        raise ValueError(f"max_deg must satisfy 0 <= max_deg < 90, got {max_deg!r}") from None
    if not math.isfinite(limit) or not 0.0 <= limit < 90.0:
        raise ValueError(f"max_deg must satisfy 0 <= max_deg < 90, got {max_deg!r}")
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] < 1:
        raise ValueError(f"positions must have shape [K, 3] with K >= 1, got {pos.shape}")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions must be finite")
    point = _as_target(target)
    tgt = np.asarray(point, dtype=np.float64)
    deltas = tgt[None, :] - pos
    dist = np.linalg.norm(deltas, axis=1)
    if np.any(dist == 0.0):
        raise ValueError("no position may equal the target")
    count = int(pos.shape[0])
    if limit == 0.0:
        look_ats = [point for _ in range(count)]
        orientations = []
        for i in range(count):
            pos_tuple = (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))
            orientations.append(look_at_orientation(pos_tuple, point))
        return look_ats, orientations, np.zeros((count, 2), dtype=np.float64)
    if rng is None or not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a np.random.Generator when max_deg > 0, got {type(rng)!r}")
    jitter_deg = np.asarray(rng.uniform(-limit, limit, size=(count, 2)), dtype=np.float64)
    look_ats = []
    orientations = []
    cap = math.radians(89.9)
    for i in range(count):
        d = deltas[i]
        dist_i = float(dist[i])
        az0 = math.atan2(float(d[1]), float(d[0]))
        el0 = math.atan2(float(d[2]), math.hypot(float(d[0]), float(d[1])))
        az = az0 + math.radians(float(jitter_deg[i, 0]))
        el = el0 + math.radians(float(jitter_deg[i, 1]))
        if abs(el) >= cap:
            raise ValueError(f"jittered elevation {math.degrees(el):.3f} deg is out of range")
        look_at = (
            float(pos[i, 0]) + dist_i * math.cos(el) * math.cos(az),
            float(pos[i, 1]) + dist_i * math.cos(el) * math.sin(az),
            float(pos[i, 2]) + dist_i * math.sin(el),
        )
        look_ats.append(look_at)
        pos_tuple = (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]))
        orientations.append(look_at_orientation(pos_tuple, look_at))
    return look_ats, orientations, jitter_deg


def _as_nonnegative_int(name: str, value: Any) -> int:
    """Return ``value`` as an int >= 0, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")
    if int(value) < 0:
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")
    return int(value)


def split_bank(
    n_bank: int, *, holdout_fraction: float, min_holdout: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split the bank into sorted ``(train_views, held_out_views)`` int64 arrays."""
    if isinstance(n_bank, bool) or not isinstance(n_bank, (int, np.integer)):
        raise ValueError(f"n_bank must be an int >= 1, got {n_bank!r}")
    n_bank = int(n_bank)
    if n_bank < 1:
        raise ValueError(f"n_bank must be >= 1, got {n_bank!r}")
    try:
        fraction = float(holdout_fraction)
    except (TypeError, ValueError):
        raise ValueError(
            f"holdout_fraction must satisfy 0 <= holdout_fraction < 1, got {holdout_fraction!r}"
        ) from None
    if not math.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError(
            f"holdout_fraction must satisfy 0 <= holdout_fraction < 1, got {holdout_fraction!r}"
        )
    min_holdout = _as_nonnegative_int("min_holdout", min_holdout)
    seed = _as_nonnegative_int("seed", seed)
    n_hold = max(math.ceil(fraction * n_bank - 1e-9), min_holdout)
    if n_hold >= n_bank:
        raise ValueError(f"holdout {n_hold} leaves no training view in a bank of {n_bank}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, SPLIT_STREAM_TAG]))
    perm = np.asarray(rng.permutation(n_bank), dtype=np.int64)
    held = np.sort(perm[:n_hold]).astype(np.int64)
    train = np.sort(perm[n_hold:]).astype(np.int64)
    return train, held


def bs_split(num_bs: int, num_held_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Split BS indices; the held-out BSs are the last ``num_held_out`` indices."""
    num_bs = _as_nonnegative_int("num_bs", num_bs)
    if num_bs < 1:
        raise ValueError(f"num_bs must be >= 1, got {num_bs!r}")
    num_held_out = _as_nonnegative_int("num_held_out", num_held_out)
    if num_held_out >= num_bs:
        raise ValueError(f"num_held_out must be < num_bs, got {num_held_out} >= {num_bs}")
    train = np.arange(num_bs - num_held_out, dtype=np.int64)
    held = np.arange(num_bs - num_held_out, num_bs, dtype=np.int64)
    return train, held


def bs_subsets(train_bs: np.ndarray, sizes: Sequence[int]) -> dict[int, np.ndarray]:
    """Return ``{B: train_bs[:B]}`` for the requested nested BS subset sizes."""
    train = np.asarray(train_bs, dtype=np.int64).ravel()
    try:
        size_list = list(sizes)
    except TypeError:
        raise ValueError(f"sizes must be positive strictly increasing ints, got {sizes!r}") from (
            None
        )
    for size in size_list:
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or int(size) < 1:
            raise ValueError(f"sizes must be positive strictly increasing ints, got {sizes!r}")
    ints = [int(size) for size in size_list]
    if any(b - a <= 0 for a, b in zip(ints, ints[1:])):
        raise ValueError(f"sizes must be positive strictly increasing ints, got {sizes!r}")
    return {size: train[:size].copy() for size in ints if size <= len(train)}


def nested_training_orders(train_views: np.ndarray, seeds: Sequence[int]) -> dict[int, np.ndarray]:
    """Return ``{seed: train_views[nested_view_order(len(train_views), seed)]}``."""
    train = np.asarray(train_views, dtype=np.int64).ravel()
    if train.size == 0:
        raise ValueError("train_views must not be empty")
    try:
        seed_list = list(seeds)
    except TypeError:
        raise ValueError("seeds must be a non-empty sequence without duplicates") from None
    if len(seed_list) == 0 or len(set(int(s) for s in seed_list)) != len(seed_list):
        raise ValueError("seeds must be a non-empty sequence without duplicates")
    for seed in seed_list:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
            raise ValueError(f"order seeds must be ints >= 0, got {seed!r}")
    return {
        int(seed): np.asarray(train[views.nested_view_order(len(train), int(seed))], dtype=np.int64)
        for seed in seed_list
    }


def view_subset_sizes(
    n_train: int, sizes: Sequence[int] = DEFAULT_VIEW_SUBSET_SIZES
) -> tuple[int, ...]:
    """Return the requested view subset sizes ``<= n_train``, in order."""
    n_train = _as_nonnegative_int("n_train", n_train)
    return tuple(int(size) for size in sizes if int(size) <= n_train)


@dataclass(frozen=True)
class NoiseReference:
    """The absolute noise reference of one dataset (P_ref, sigma2, per-capture SNRs)."""

    snr_db: float
    p_ref: float
    c_ref: tuple[int, int]
    sigma2: float
    los_fallback: bool
    capture_power: np.ndarray
    expected_snr_db: np.ndarray
    scatter_power: np.ndarray | None = None
    expected_scatter_snr_db: np.ndarray | None = None


def noise_reference(
    Y_clean: np.ndarray,
    los_visible: np.ndarray,
    capture_mask: np.ndarray,
    snr_db: float,
    *,
    Y_los_free: np.ndarray | None = None,
) -> NoiseReference:
    """Return the absolute noise reference (lower-median LoS-visible training capture)."""
    arr = np.asarray(Y_clean, dtype=np.complex128)
    if arr.ndim != 6 or arr.shape[2] != 2:
        raise ValueError(f"Y_clean must have shape [V, B, 2, R, C, N], got {arr.shape}")
    los = np.asarray(los_visible, dtype=bool)
    if los.shape != arr.shape[:2]:
        raise ValueError(f"los_visible must have shape {arr.shape[:2]}, got {los.shape}")
    mask = np.asarray(capture_mask, dtype=bool)
    if mask.shape != arr.shape[:2]:
        raise ValueError(f"capture_mask must have shape {arr.shape[:2]}, got {mask.shape}")
    if not np.any(mask):
        raise ValueError("capture_mask must select at least one capture")
    try:
        snr = float(snr_db)
    except (TypeError, ValueError):
        raise ValueError(f"snr_db must be finite, got {snr_db!r}") from None
    if not math.isfinite(snr):
        raise ValueError(f"snr_db must be finite, got {snr_db!r}")
    free: np.ndarray | None = (
        np.asarray(Y_los_free, dtype=np.complex128) if Y_los_free is not None else None
    )
    if free is not None and free.shape != arr.shape:
        raise ValueError(f"Y_los_free shape {free.shape} does not match Y_clean shape {arr.shape}")
    candidates = los & mask
    if np.any(candidates):
        p_ref, c_ref = sync.reference_power(arr, candidates)
        los_fallback = False
    else:
        p_ref, c_ref = sync.reference_power(arr, mask)
        los_fallback = True
    if not p_ref > 0.0:
        raise ValueError(f"reference power must be > 0, got {p_ref!r}")
    sigma2 = float(p_ref) / 10.0 ** (snr / 10.0)
    powers = sync.capture_power(arr)
    with np.errstate(divide="ignore"):
        expected = 10.0 * np.log10(powers / sigma2)
    expected = np.asarray(expected, dtype=np.float64)
    scatter: np.ndarray | None = None
    expected_scatter: np.ndarray | None = None
    if free is not None:
        scatter = sync.capture_power(free)
        with np.errstate(divide="ignore"):
            expected_scatter = 10.0 * np.log10(scatter / sigma2)
        expected_scatter = np.asarray(expected_scatter, dtype=np.float64)
    return NoiseReference(
        snr_db=snr,
        p_ref=float(p_ref),
        c_ref=(int(c_ref[0]), int(c_ref[1])),
        sigma2=float(sigma2),
        los_fallback=bool(los_fallback),
        capture_power=np.asarray(powers, dtype=np.float64),
        expected_snr_db=expected,
        scatter_power=scatter,
        expected_scatter_snr_db=expected_scatter,
    )
