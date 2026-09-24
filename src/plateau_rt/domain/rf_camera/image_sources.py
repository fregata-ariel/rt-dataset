"""Image-source (virtual-source) geometry for specular RF-camera multipath.

Each specular multipath component reaches the UE as if it came from a point
source at the mirror image of the transmitter, unfolded at every reflection.
This module builds those virtual sources from Sionna path geometry, projects
them into the UE-local direction-cosine image and provides small helpers to
match virtual sources against measured image peaks.

The module is NumPy only. Sionna angles are inputs (``theta_r``/``phi_r``),
never imported, so the geometry can be tested without a GPU scene.
"""

from __future__ import annotations

import numpy as np

from plateau_rt.domain.rf_camera.calibration import rotation_matrix
from plateau_rt.domain.rf_camera.delay import SPEED_OF_LIGHT_M_S


def arrival_unit_vectors(theta_r: np.ndarray, phi_r: np.ndarray) -> np.ndarray:
    """Convert Sionna world-frame arrival angles to unit vectors.

    Sionna's ``theta_r``/``phi_r`` describe the direction *from the UE toward
    where the wave comes from* (the last interaction point or the BS), so the
    returned unit vector points along the back-traced arrival direction.

    Returns an array of shape ``(..., 3)``.
    """
    theta, phi = np.broadcast_arrays(
        np.asarray(theta_r, dtype=np.float64), np.asarray(phi_r, dtype=np.float64)
    )
    sin_theta = np.sin(theta)
    return np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)], axis=-1)


def virtual_source_positions(
    ue_position: np.ndarray | tuple[float, float, float],
    tau_s: np.ndarray,
    theta_r: np.ndarray,
    phi_r: np.ndarray,
) -> np.ndarray:
    """Return world-frame virtual-source positions ``UE + c*tau*r_hat``.

    For a specular-only chain this point is the mirror image of the transmitter
    unfolded at every reflection, so ``c*tau`` equals the distance from the UE
    to the virtual source.
    """
    ue = np.asarray(ue_position, dtype=np.float64)
    tau = np.asarray(tau_s, dtype=np.float64)
    r_hat = arrival_unit_vectors(theta_r, phi_r)
    tau_b = np.broadcast_to(tau[..., None], r_hat.shape[:-1] + (1,))
    return ue + SPEED_OF_LIGHT_M_S * tau_b * r_hat


def world_to_local_directions(
    directions_world: np.ndarray,
    orientation: tuple[float, float, float],
) -> np.ndarray:
    """Rotate world-frame directions into the UE-local frame.

    ``rotation_matrix(orientation)`` has the local axes as columns, so world to
    local is ``R.T @ v`` for every stacked vector of shape ``(..., 3)``.
    """
    directions = np.asarray(directions_world, dtype=np.float64)
    rotation = rotation_matrix(orientation)
    return directions @ rotation


def local_direction_to_image_coords(
    direction_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project UE-local directions onto the front-image coordinates.

    Returns ``(ky, kz, in_front)`` where ``in_front`` is ``local kx >= 0``. A
    planar y-z aperture cannot resolve the sign of local kx on its own, so the
    hemisphere split supplies ``in_front``.
    """
    local = np.asarray(direction_local, dtype=np.float64)
    return local[..., 1], local[..., 2], local[..., 0] >= 0.0


def mirror_point(
    point: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """Return the mirror image of ``point`` across a plane."""
    point = np.asarray(point, dtype=np.float64)
    plane_point = np.asarray(plane_point, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    return point - 2.0 * np.dot(point - plane_point, normal) * normal


def polyline_length(points: np.ndarray) -> float:
    """Return the total length of a polyline given as ``(num_vertices, 3)``."""
    vertices = np.asarray(points, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("points must have shape (num_vertices, 3)")
    if vertices.shape[0] < 2:
        return 0.0
    segments = np.linalg.norm(np.diff(vertices, axis=0), axis=-1)
    return float(np.sum(segments))


def point_to_ray_distance(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Return the perpendicular distance from ``point`` to a ray."""
    point = np.asarray(point, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    offset = point - origin
    along = np.dot(offset, direction)
    return float(np.linalg.norm(offset - along * direction))


def unfold_specular_chain(
    tx_position: np.ndarray,
    vertices: np.ndarray,
    is_reflection: np.ndarray,
    rx_position: np.ndarray,
) -> np.ndarray:
    """Return the image of ``tx_position`` through a specular vertex chain.

    The vertices are walked in TX->RX order. At every reflection vertex ``k``
    the facet normal is estimated from the local geometry

    ``n = u_out - u_in``, with ``u_in = unit(v_k - prev)`` and
    ``u_out = unit(next - v_k)``, and the *current* image is mirrored across the
    plane ``(v_k, n)`` with :func:`mirror_point`.

    Refraction vertices are straight-through, so they carry no direction change
    of their own and are ignored completely: ``prev`` is the nearest *preceding
    reflection* vertex (or the TX when there is none) and ``next`` is the
    nearest *following reflection* vertex (or the RX when there is none). Using
    the nearest reflection neighbour rather than the immediate vertex avoids
    amplifying sub-millimetre intersection jitter on the short segment next to a
    refraction vertex. A reflection whose incoming and outgoing directions
    coincide (``|u_out - u_in| < 1e-12``) is skipped. With no vertices the TX
    position is returned unchanged.
    """
    tx = np.asarray(tx_position, dtype=np.float64)
    rx = np.asarray(rx_position, dtype=np.float64)
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    reflections = np.asarray(is_reflection, dtype=bool).reshape(-1)

    if points.shape[0] != reflections.shape[0]:
        raise ValueError("vertices and is_reflection must have the same length")

    image = tx.copy()
    num_vertices = points.shape[0]
    if num_vertices == 0:
        return image

    reflection_indices = [index for index in range(num_vertices) if bool(reflections[index])]
    last = len(reflection_indices) - 1
    for order, index in enumerate(reflection_indices):
        vertex = points[index]
        if order == 0:
            previous = tx
        else:
            previous = points[reflection_indices[order - 1]]
        if order == last:
            following = rx
        else:
            following = points[reflection_indices[order + 1]]

        incoming = vertex - previous
        outgoing = following - vertex
        incoming = incoming / np.linalg.norm(incoming)
        outgoing = outgoing / np.linalg.norm(outgoing)

        normal = outgoing - incoming
        if np.linalg.norm(normal) < 1e-12:
            continue
        image = mirror_point(image, vertex, normal)

    return image


def hann_taper(rows: int, cols: int) -> np.ndarray:
    """Return a separable Hann taper of shape ``(rows, cols)`` with non-zero edges.

    The taper is ``outer(np.hanning(rows + 2)[1:-1], np.hanning(cols + 2)[1:-1])``,
    which removes the zero endpoints of the ordinary Hann window. Raises
    ``ValueError`` if ``rows < 1`` or ``cols < 1``.
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    return np.outer(np.hanning(rows + 2)[1:-1], np.hanning(cols + 2)[1:-1])


def pattern_summed_element_power(
    rx_values: np.ndarray,
    *,
    rows: int,
    cols: int,
    num_patterns: int = 2,
    element: int = 0,
    axis: int = 0,
) -> np.ndarray:
    """Return ``|sum_p rx_values[p*rows*cols + element]|^2`` along a fused axis.

    ``rx_values`` is indexed along ``axis`` by Sionna's pattern-major fused
    receive axis, so channel ``p*rows*cols + element`` is the same element seen
    through pattern ``p``. The complex values of the requested element are
    summed over the patterns and squared, and the fused ``axis`` is removed from
    the output. Complex input is accepted.

    Raises ``ValueError`` if the axis length is not ``num_patterns*rows*cols``
    or if ``element`` is out of range.
    """
    values = np.asarray(rx_values)
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    if num_patterns < 1:
        raise ValueError("num_patterns must be >= 1")
    if values.ndim == 0:
        raise ValueError("rx_values must have at least one axis")
    if axis < 0:
        axis += values.ndim
    if axis < 0 or axis >= values.ndim:
        raise ValueError("axis out of range")
    size = rows * cols
    if values.shape[axis] != num_patterns * size:
        raise ValueError(
            f"axis length must be num_patterns*rows*cols = {num_patterns * size}, "
            f"got {values.shape[axis]}"
        )
    if element < 0 or element >= size:
        raise ValueError(f"element must be in [0, {size})")

    indices = [pattern * size + element for pattern in range(num_patterns)]
    selected = np.take(values, indices, axis=axis)
    return np.abs(np.sum(selected, axis=axis)) ** 2


def match_nearest(
    peaks_kykz: np.ndarray,
    sources_kykz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match each peak to its nearest virtual source in direction cosine.

    Returns ``(indices, distances)`` with one entry per peak. If there are no
    sources, indices are ``-1`` and distances are ``inf``.
    """
    peaks = np.asarray(peaks_kykz, dtype=np.float64).reshape(-1, 2)
    sources = np.asarray(sources_kykz, dtype=np.float64).reshape(-1, 2)
    if peaks.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=np.float64)
    if sources.size == 0:
        return np.full(peaks.shape[0], -1, dtype=int), np.full(peaks.shape[0], np.inf)

    distances = np.linalg.norm(peaks[:, None, :] - sources[None, :, :], axis=-1)
    indices = np.argmin(distances, axis=1)
    return indices, distances[np.arange(peaks.shape[0]), indices]


def match_peaks_to_sources(
    peaks_kykz: np.ndarray,
    sources_kykz: np.ndarray,
    *,
    source_ids: np.ndarray,
    candidate_mask: np.ndarray,
    max_distance: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Match each peak to its nearest *candidate* source in direction cosine.

    Only sources with ``candidate_mask == True`` participate. The returned
    ``ids`` are taken from ``source_ids`` (for example ``path_index`` values),
    and ``distances`` hold the nearest-candidate distance. A peak with no
    candidate, or whose nearest candidate is farther than ``max_distance``,
    gets id ``-1``; its distance is still the nearest-candidate distance (``inf``
    when there are no candidates at all). Empty peaks give empty ``int`` and
    ``float`` arrays.
    """
    peaks = np.asarray(peaks_kykz, dtype=np.float64).reshape(-1, 2)
    sources = np.asarray(sources_kykz, dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(source_ids, dtype=int).reshape(-1)
    candidate = np.asarray(candidate_mask, dtype=bool).reshape(-1)

    if ids.shape[0] != sources.shape[0] or candidate.shape[0] != sources.shape[0]:
        raise ValueError("source_ids and candidate_mask must match the sources")

    if peaks.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=np.float64)

    selected = candidate
    if not np.any(selected):
        return np.full(peaks.shape[0], -1, dtype=int), np.full(peaks.shape[0], np.inf)

    candidates = sources[selected]
    candidate_ids = ids[selected]
    distances = np.linalg.norm(peaks[:, None, :] - candidates[None, :, :], axis=-1)
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(peaks.shape[0]), nearest]
    nearest_ids = candidate_ids[nearest].astype(int)
    nearest_ids = np.where(nearest_distance > max_distance, -1, nearest_ids)
    return nearest_ids, nearest_distance


def source_recall(
    peaks_kykz: np.ndarray,
    sources_kykz: np.ndarray,
    *,
    max_distance: float,
) -> np.ndarray:
    """Return a boolean per source: is any peak within ``max_distance``?

    Distances are Euclidean in direction-cosine space. With no peaks every
    source is ``False``.
    """
    peaks = np.asarray(peaks_kykz, dtype=np.float64).reshape(-1, 2)
    sources = np.asarray(sources_kykz, dtype=np.float64).reshape(-1, 2)
    if sources.size == 0:
        return np.empty(0, dtype=bool)
    if peaks.size == 0:
        return np.zeros(sources.shape[0], dtype=bool)

    distances = np.linalg.norm(sources[:, None, :] - peaks[None, :, :], axis=-1)
    return np.any(distances <= max_distance, axis=1)


def find_image_peaks(
    power: np.ndarray,
    *,
    mask: np.ndarray,
    threshold_db: float = -20.0,
    size: int = 5,
) -> list[tuple[int, int]]:
    """Find local maxima of ``power`` inside ``mask``, strongest first.

    A peak is the maximum of its ``size`` x ``size`` neighbourhood (computed
    with :func:`numpy.lib.stride_tricks.sliding_window_view` and ``-inf``
    padding, matching a constant ``-inf`` maximum filter for odd ``size``) and
    lies within ``threshold_db`` of the strongest peak. Overlapping maxima of a
    flat-topped lobe are greedily suppressed to one point.
    """
    values = np.asarray(power, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("power and mask must have the same shape")
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be an odd integer >= 1")

    work = np.where(valid, values, -np.inf)
    half = size // 2
    padded = np.pad(work, half, mode="constant", constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    local_max = np.max(windows, axis=(-1, -2))
    candidates = valid & np.isfinite(work) & (work >= local_max)
    if not np.any(candidates):
        return []

    peak_value = float(np.max(work[candidates]))
    if peak_value <= 0.0:
        return []
    floor = peak_value * 10.0 ** (threshold_db / 10.0)
    candidates &= work >= floor

    rows, cols = np.nonzero(candidates)
    order = np.argsort(work[rows, cols])[::-1]
    rows, cols = rows[order], cols[order]

    keep: list[tuple[int, int]] = []
    min_separation = max(size // 2, 1)
    for row, col in zip(rows.tolist(), cols.tolist()):
        if all(
            max(abs(row - kept_row), abs(col - kept_col)) > min_separation
            for kept_row, kept_col in keep
        ):
            keep.append((row, col))
    return keep
