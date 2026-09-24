"""Exact spherical-wave forward model and least-squares recovery of point scatterers.

This is an exploratory, NumPy/SciPy-only prototype that seeds a later "RF
Gaussian Splatting" renderer. It simulates the complex channel frequency
response (CFR) observed by a planar receive aperture from a small number of
isotropic point scatterers, and reconstructs their complex reflectivities on a
voxel grid by regularised (Tikhonov) least squares.

Model conventions (matching Sionna's ``Paths.cfr()``)

* Time-harmonic sign: ``H(f) = sum a * exp(-j 2 pi f tau)``.
* Direct path (isotropic element, free space): ``a = lambda_c / (4 pi r)`` with
  ``lambda_c = c / fc`` the *carrier* wavelength used for every frequency bin.
* A point scatterer of complex reflectivity ``rho`` contributes

  ``h = rho * lambda_c / ((4 pi)**1.5 * r1 * r2) * exp(-j 2 pi f (r1 + r2) / c)``

  where ``r1 = |s - bs|``, ``r2 = |p_elem - s|`` and ``|rho|**2`` is the
  bistatic RCS in m^2.
* A far source seen from the aperture centre in unit direction ``u``
  (UE toward source) adds the plane-wave element phase
  ``exp(+j k_c u . p_local)`` with ``k_c = 2 pi fc / c`` the *carrier*
  wavenumber (identical for every frequency bin) and ``p_local`` the element
  offset in UE-local coordinates. This reproduces Sionna
  ``synthetic_array=True``, which applies the element phase with the carrier
  wavelength for every bin. The exact spherical model (``plane_wave=False``)
  instead disperses the element phase per bin; it differs from Sionna
  synthetic-array data by about ``1e-3`` in normalised correlation, which is a
  model floor when fitting such data.

The direct path is assumed known. :func:`system_matrix` describes only the
scatterer response for unit reflectivity, so observations must have the direct
path removed ("background subtraction") before calling :func:`reconstruct`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.sparse.linalg import lsqr

from plateau_rt.domain.rf_camera.calibration import rotation_matrix
from plateau_rt.domain.rf_camera.camera import RFViewSpec
from plateau_rt.domain.rf_camera.delay import SPEED_OF_LIGHT_M_S
from plateau_rt.domain.rf_camera.imaging import frequency_offsets

_TWO_PI = 2.0 * np.pi
_FOUR_PI = 4.0 * np.pi
_SPHERICAL_AMPLITUDE = _FOUR_PI**1.5


@dataclass(frozen=True)
class ApertureSpec:
    """Sampling plan of one planar receive aperture.

    Rows run top to bottom (local +z to -z); columns run along local +y, i.e.
    from the camera's right to its left. Element ``(0, 0)`` is at local
    ``(+z, -y)``, the top-right corner as seen by the camera looking along +x.
    Element spacing is ``spacing_lambda`` carrier wavelengths both ways and does
    not scale with frequency.
    """

    rows: int
    cols: int
    carrier_hz: float = 3.5e9
    bandwidth_hz: float = 100e6
    num_freq: int = 16
    spacing_lambda: float = 0.5

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if self.carrier_hz <= 0.0:
            raise ValueError("carrier_hz must be > 0")
        if self.bandwidth_hz < 0.0:
            raise ValueError("bandwidth_hz must be >= 0")
        if self.num_freq < 1:
            raise ValueError("num_freq must be >= 1")
        if self.spacing_lambda <= 0.0:
            raise ValueError("spacing_lambda must be > 0")


def carrier_wavelength_m(spec: ApertureSpec) -> float:
    """Return the carrier wavelength ``c / fc`` in metres."""
    return SPEED_OF_LIGHT_M_S / spec.carrier_hz


def element_positions_world(
    spec: ApertureSpec,
    ue_position: tuple[float, float, float],
    ue_orientation: tuple[float, float, float],
) -> np.ndarray:
    """Return element centres in world coordinates, shape ``[rows, cols, 3]``.

    Element ``(row i, col j)`` has UE-local offset ``y = d * j - (cols - 1) * d
    / 2`` and ``z = -d * i + (rows - 1) * d / 2`` with ``d = spacing * lambda_c``,
    so index ``(0, 0)`` sits at local ``(+z, -y)``: the top-right corner as seen
    by the camera looking along +x (columns run from the camera's right to its
    left).
    """
    spacing = spec.spacing_lambda * carrier_wavelength_m(spec)
    row = np.arange(spec.rows, dtype=np.float64)[:, None]
    col = np.arange(spec.cols, dtype=np.float64)[None, :]
    y = np.broadcast_to(spacing * col - (spec.cols - 1) * spacing / 2.0, (spec.rows, spec.cols))
    z = np.broadcast_to(-spacing * row + (spec.rows - 1) * spacing / 2.0, (spec.rows, spec.cols))
    local = np.stack([np.zeros((spec.rows, spec.cols)), y, z], axis=-1)

    rotation = rotation_matrix(ue_orientation)
    return np.asarray(ue_position, dtype=np.float64) + local @ rotation.T


def frequencies_hz(spec: ApertureSpec) -> np.ndarray:
    """Return absolute frequencies ``fc + offsets`` in Hz as float64."""
    offsets = frequency_offsets(spec.bandwidth_hz, spec.num_freq).astype(np.float64)
    return spec.carrier_hz + offsets


def direct_path_cfr(
    spec: ApertureSpec,
    bs: tuple[float, float, float],
    ue_position: tuple[float, float, float],
    ue_orientation: tuple[float, float, float],
    *,
    plane_wave: bool = False,
) -> np.ndarray:
    """Return the direct-path CFR sampled by the aperture, ``[rows, cols, F]``.

    With ``plane_wave=True`` the aperture-centre distance drives the amplitude
    and the absolute per-bin delay, while the element phase uses the carrier
    wavenumber ``k_c = 2 pi fc / c`` for every bin:
    ``exp(+j k_c u_local . p_local)``. The ramp is therefore identical in every
    column, which mimics Sionna's ``synthetic_array=True``. With
    ``plane_wave=False`` the exact spherical model drives both amplitude and
    delay per element and disperses the element phase across bins.
    """
    frequencies = frequencies_hz(spec)
    wavelength = carrier_wavelength_m(spec)
    bs_arr = np.asarray(bs, dtype=np.float64)
    ue_arr = np.asarray(ue_position, dtype=np.float64)
    elements = element_positions_world(spec, ue_position, ue_orientation)

    if plane_wave:
        centre_distance = float(np.linalg.norm(bs_arr - ue_arr))
        if centre_distance == 0.0:
            raise ValueError("bs and ue_position must differ")
        direction_world = (bs_arr - ue_arr) / centre_distance
        rotation = rotation_matrix(ue_orientation)
        direction_local = rotation.T @ direction_world
        # World offsets are rotated into the UE-local frame before the dot
        # product, so this is exactly exp(+j k_c u_local . p_local).
        local_offsets = (elements - ue_arr).reshape(-1, 3) @ rotation
        projection = local_offsets @ direction_local
        base = (wavelength / _FOUR_PI / centre_distance) * np.exp(
            -1j * _TWO_PI * frequencies * (centre_distance / SPEED_OF_LIGHT_M_S)
        )
        # Sionna synthetic_array applies the element phase at the carrier
        # wavelength for every frequency bin.
        carrier_wavenumber = _TWO_PI * spec.carrier_hz / SPEED_OF_LIGHT_M_S
        element_phase = np.exp(1j * carrier_wavenumber * projection)
        return (base[None, :] * element_phase[:, None]).reshape(
            spec.rows, spec.cols, frequencies.size
        )

    distances = np.linalg.norm(elements - bs_arr, axis=-1)
    amplitude = wavelength / _FOUR_PI / distances
    delay = distances / SPEED_OF_LIGHT_M_S
    return amplitude[..., None] * np.exp(-1j * _TWO_PI * delay[..., None] * frequencies)


def scatterer_channels(
    points: np.ndarray,
    transmitter_position: np.ndarray,
    receiver_positions: np.ndarray,
    frequencies: np.ndarray,
    carrier_wavelength: float,
) -> np.ndarray:
    """Return per-scatterer unit-reflectivity channels, shape ``[R, S, F]``.

    ``points`` has shape ``[S, 3]`` and ``receiver_positions`` shape ``[R, 3]``.
    Each channel already carries the ``lambda_c / ((4 pi)**1.5 r1 r2)``
    amplitude and the ``exp(-j 2 pi f (r1 + r2) / c)`` phase, so multiplying by
    ``rho`` and summing over ``S`` gives :func:`scatterer_cfr`.
    """
    points_arr = np.atleast_2d(np.asarray(points, dtype=np.float64))
    transmitter = np.asarray(transmitter_position, dtype=np.float64).reshape(3)
    receivers = np.asarray(receiver_positions, dtype=np.float64).reshape(-1, 3)
    freqs = np.asarray(frequencies, dtype=np.float64).reshape(-1)

    r1 = np.linalg.norm(points_arr - transmitter, axis=-1)
    r2 = np.linalg.norm(receivers[:, None, :] - points_arr[None, :, :], axis=-1)
    amplitude = carrier_wavelength / (_SPHERICAL_AMPLITUDE * r1[None, :] * r2)
    delay = (r1[None, :] + r2) / SPEED_OF_LIGHT_M_S
    phase = np.exp(-1j * _TWO_PI * delay[:, :, None] * freqs[None, None, :])
    return amplitude[:, :, None] * phase


def scatterer_cfr(
    spec: ApertureSpec,
    bs: tuple[float, float, float],
    ue_position: tuple[float, float, float],
    ue_orientation: tuple[float, float, float],
    points: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """Return the summed scatterer CFR, ``[rows, cols, F]``.

    ``points`` has shape ``[S, 3]`` and ``rho`` shape ``[S]`` (complex, with
    ``|rho|**2`` the bistatic RCS in m^2).
    """
    frequencies = frequencies_hz(spec)
    wavelength = carrier_wavelength_m(spec)
    elements = element_positions_world(spec, ue_position, ue_orientation).reshape(-1, 3)
    channels = scatterer_channels(points, bs, elements, frequencies, wavelength)
    rho_arr = np.asarray(rho, dtype=np.complex128).reshape(-1)
    response = np.sum(channels * rho_arr[None, :, None], axis=1)
    return response.reshape(spec.rows, spec.cols, frequencies.size)


def system_matrix(
    spec: ApertureSpec,
    bs: tuple[float, float, float],
    views: Sequence[RFViewSpec],
    voxels: np.ndarray,
) -> np.ndarray:
    """Return the unit-reflectivity response matrix, ``[sum_views R*C*F, V]``.

    Row order matches flattening each view's ``[rows, cols, F]`` CFR in C order
    (row, then column, then frequency); columns are voxels. Observations must
    have the direct path subtracted before these columns are fitted.
    """
    frequencies = frequencies_hz(spec)
    wavelength = carrier_wavelength_m(spec)
    voxels_arr = np.atleast_2d(np.asarray(voxels, dtype=np.float64))
    num_voxels = voxels_arr.shape[0]
    rows_per_view = spec.rows * spec.cols * frequencies.size

    blocks = []
    for view in views:
        elements = element_positions_world(spec, view.position, view.orientation).reshape(-1, 3)
        channels = scatterer_channels(voxels_arr, bs, elements, frequencies, wavelength)
        block = np.moveaxis(channels, 1, 2).reshape(rows_per_view, num_voxels)
        blocks.append(block)
    return np.concatenate(blocks, axis=0)


def reconstruct(
    A: np.ndarray,
    y: np.ndarray,
    *,
    damp: float,
    method: Literal["lsqr", "normal"] = "lsqr",
) -> np.ndarray:
    """Solve the damped least-squares problem ``min ||A x - y||^2 + damp^2 ||x||^2``.

    ``method="lsqr"`` calls :func:`scipy.sparse.linalg.lsqr`, which accepts
    complex inputs. ``method="normal"`` forms and solves the normal equations
    ``(A^H A + damp^2 I) x = A^H y`` directly.
    """
    A_arr = np.asarray(A)
    y_arr = np.asarray(y).reshape(-1)
    if damp < 0.0:
        raise ValueError("damp must be >= 0")

    if method == "lsqr":
        return lsqr(A_arr, y_arr, damp=damp)[0]
    if method == "normal":
        gram = A_arr.conj().T @ A_arr + (damp**2) * np.eye(A_arr.shape[1])
        return np.linalg.solve(gram, A_arr.conj().T @ y_arr)
    raise ValueError(f"unknown reconstruction method: {method!r}")


def _thin_svd(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the economy SVD ``(U, s, Vh)`` of ``A``."""
    return np.linalg.svd(np.asarray(A), full_matrices=False)


def tikhonov_path(
    A: np.ndarray,
    y: np.ndarray,
    damps: np.ndarray,
    *,
    svd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return the Tikhonov solutions for every damp as ``[len(damps), V]``.

    Computed from a single economy SVD ``U, s, Vh = svd(A)`` as
    ``x(d) = Vh^H @ ((s / (s**2 + d**2)) * (U^H y))``. Pass a precomputed
    ``svd`` to reuse the decomposition across calls.
    """
    A_arr = np.asarray(A)
    y_arr = np.asarray(y).reshape(-1)
    damps_arr = np.asarray(damps, dtype=np.float64).reshape(-1)
    if damps_arr.size == 0:
        raise ValueError("damps must be non-empty")
    if np.any(damps_arr < 0.0):
        raise ValueError("damps must be >= 0")

    if svd is None:
        svd = _thin_svd(A_arr)
    U, s, Vh = svd
    projected = U.conj().T @ y_arr
    filters = s[None, :] / (s[None, :] ** 2 + damps_arr[:, None] ** 2)
    # Vh^H @ z == z @ Vh.conj() for a row vector z.
    return (filters * projected[None, :]) @ Vh.conj()


def _svd_residual_norms(
    U: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    damps: np.ndarray,
) -> np.ndarray:
    """Return ``||A x(d) - y||`` for every damp using one economy SVD."""
    y_arr = np.asarray(y).reshape(-1)
    projected = U.conj().T @ y_arr
    norm_y_sq = float(np.vdot(y_arr, y_arr).real)
    norm_projected_sq = float(np.vdot(projected, projected).real)
    orthogonal_sq = max(norm_y_sq - norm_projected_sq, 0.0)
    filters = damps[:, None] ** 2 / (s[None, :] ** 2 + damps[:, None] ** 2)
    residual_sq = np.sum((filters * np.abs(projected)[None, :]) ** 2, axis=1) + orthogonal_sq
    return np.sqrt(np.maximum(residual_sq, 0.0))


def discrepancy_damp(
    A: np.ndarray,
    y: np.ndarray,
    noise_sigma: float,
    damps: np.ndarray,
    tau: float = 1.0,
    *,
    svd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> float:
    """Return the Morozov discrepancy-principle damp from a damp grid.

    The complex noise is assumed to have ``E|n|^2 = noise_sigma**2`` per entry,
    so the target residual norm is ``tau * noise_sigma * sqrt(m)`` with
    ``m = len(y)``. The returned damp is the largest grid value whose residual
    ``||A x(d) - y||`` satisfies the target (residuals grow with damp). If
    no grid value does, the smallest damp is returned. One economy SVD drives
    all residuals; pass ``svd`` to reuse it.
    """
    if noise_sigma < 0.0:
        raise ValueError("noise_sigma must be >= 0")
    damps_arr = np.asarray(damps, dtype=np.float64).reshape(-1)
    if damps_arr.size == 0:
        raise ValueError("damps must be non-empty")

    if svd is None:
        svd = _thin_svd(np.asarray(A))
    U, s, _ = svd
    residuals = _svd_residual_norms(U, s, y, damps_arr)
    target = tau * noise_sigma * np.sqrt(np.asarray(y).size)

    for index in np.argsort(damps_arr)[::-1]:
        if residuals[index] <= target:
            return float(damps_arr[index])
    return float(np.min(damps_arr))


def voxel_axes(
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three 1-D voxel-centre axes of a regular grid."""
    if spacing <= 0.0:
        raise ValueError("spacing must be > 0")
    axes = []
    for low, high in zip(bounds_min, bounds_max):
        count = int(np.floor((high - low) / spacing + 1e-9)) + 1
        axes.append(low + spacing * np.arange(count, dtype=np.float64))
    return axes[0], axes[1], axes[2]


def voxel_grid(
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    spacing: float,
) -> np.ndarray:
    """Return voxel centres as ``[V, 3]`` in C order (x, then y, then z)."""
    x, y, z = voxel_axes(bounds_min, bounds_max, spacing)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")
    return np.stack(
        [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()],
        axis=-1,
    )


def relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Return ``||estimate - truth|| / ||truth||`` for complex vectors."""
    estimate_arr = np.asarray(estimate, dtype=np.complex128).reshape(-1)
    truth_arr = np.asarray(truth, dtype=np.complex128).reshape(-1)
    if estimate_arr.shape != truth_arr.shape:
        raise ValueError("estimate and truth must have the same shape")
    denominator = float(np.linalg.norm(truth_arr))
    if denominator == 0.0:
        raise ValueError("truth must be non-zero")
    return float(np.linalg.norm(estimate_arr - truth_arr) / denominator)
