"""Unit tests for :mod:`plateau_rt.domain.rf_tomography.observables` (T04)."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
from scipy.constants import c as SPEED_OF_LIGHT

from plateau_rt.domain.rf_camera import calibration, delay, imaging
from plateau_rt.domain.rf_tomography import observables
from plateau_rt.domain.rf_tomography.observables import NODE_ALIASES, NODE_NAMES

SPACING = 0.5


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _wave(
    shape: tuple[int, ...],
    u: np.ndarray,
    tau: float,
    alpha: complex,
    *,
    delta_f: float,
    spacing: float = SPACING,
) -> np.ndarray:
    """One plane wave using the centred aperture offsets and the data signs."""
    num_views, num_bs, num_hemispheres, rows, cols, num_bins = shape
    hemisphere = 0 if u[0] >= 0.0 else 1
    row = np.arange(rows)[:, None, None]
    col = np.arange(cols)[None, :, None]
    n = np.arange(num_bins)[None, None, :]
    spatial = np.exp(
        1j
        * 2.0
        * np.pi
        * spacing
        * (u[1] * (col - (cols - 1) / 2.0) + u[2] * ((rows - 1) / 2.0 - row))
    )
    delay_term = np.exp(-1j * 2.0 * np.pi * (n - num_bins // 2) * delta_f * tau)
    wave = alpha * spatial * delay_term
    out = np.zeros(shape, dtype=np.complex128)
    out[:, :, hemisphere] += wave
    return out


def _sum_waves(
    shape: tuple[int, ...],
    waves: list[tuple[np.ndarray, float, complex]],
    *,
    delta_f: float,
    spacing: float = SPACING,
) -> np.ndarray:
    out = np.zeros(shape, dtype=np.complex128)
    for u, tau, alpha in waves:
        out += _wave(shape, u, tau, alpha, delta_f=delta_f, spacing=spacing)
    return out


def _noise(shape: tuple[int, ...], sigma2: float, rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(sigma2 / 2.0)
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def _gauge(Y: np.ndarray, phi: np.ndarray, tau: np.ndarray, delta_f: float) -> np.ndarray:
    num_bins = Y.shape[-1]
    offsets = (np.arange(num_bins) - num_bins // 2) * delta_f
    factor = np.exp(1j * phi)[..., None] * np.exp(
        -1j * 2.0 * np.pi * offsets[None, None, :] * tau[..., None]
    )
    return Y * factor[:, :, None, None, None, :]


def _windows(window: str | None, rows: int, cols: int, num_bins: int) -> tuple[np.ndarray, ...]:
    if window is None:
        return (np.ones(rows), np.ones(cols), np.ones(num_bins))
    return (
        observables.taylor_window(rows),
        observables.taylor_window(cols),
        observables.taylor_window(num_bins),
    )


def _largest_sidelobe_db(profile: np.ndarray, peak: int) -> float:
    values = np.asarray(profile, dtype=np.float64)
    previous = np.roll(values, 1)
    following = np.roll(values, -1)
    local = (values > previous) & (values >= following)
    local[peak] = False
    sidelobes = values[local]
    if sidelobes.size == 0:
        return -np.inf
    return float(10.0 * np.log10(np.max(sidelobes) / values[peak]))


def _circular_parabola(nt: int, k0: float, amplitude: float) -> np.ndarray:
    k = np.arange(nt, dtype=np.float64)
    distance = np.mod(k - k0 + nt / 2.0, float(nt)) - nt / 2.0
    return amplitude - distance**2


def _direct_i(Y: np.ndarray) -> np.ndarray:
    num_bins = Y.shape[-1]
    power = np.zeros(Y.shape[:4] + (Y.shape[4],), dtype=np.float64)
    for n in range(num_bins):
        beam = observables.angle_delay_volume(Y[..., n : n + 1])[..., 0]
        power += np.abs(beam) ** 2
    return power


# --------------------------------------------------------------------------------------
# 1. Unitarity of the unwindowed native transform
# --------------------------------------------------------------------------------------
def test_unitary_volume_is_orthonormal() -> None:
    rows, cols, num_bins = 3, 4, 6
    size = rows * cols * num_bins
    matrix = np.empty((size, size), dtype=np.complex128)
    for index in range(size):
        basis = np.zeros((1, 1, 1, rows, cols, num_bins), dtype=np.complex128)
        basis.flat[index] = 1.0
        matrix[:, index] = observables.angle_delay_volume(basis)[0, 0, 0].ravel()
    np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(size), atol=1e-12)

    rng = _rng(1)
    data = rng.standard_normal((1, 1, 1, 8, 8, 16)) + 1j * rng.standard_normal((1, 1, 1, 8, 8, 16))
    volume = observables.angle_delay_volume(data)
    np.testing.assert_allclose(np.linalg.norm(volume), np.linalg.norm(data), rtol=1e-12)


# --------------------------------------------------------------------------------------
# 2. Volume against the physical triple sum and the axis definitions
# --------------------------------------------------------------------------------------
def test_volume_matches_physical_definition() -> None:
    rows, cols, num_bins = 4, 6, 8
    delta_f = 100e6 / num_bins
    rng = _rng(2)
    data = rng.standard_normal((1, 1, 1, rows, cols, num_bins)) + 1j * rng.standard_normal(
        (1, 1, 1, rows, cols, num_bins)
    )
    angle_factor, delay_factor = 2, 3
    qy, qz, nt = angle_factor * cols, angle_factor * rows, delay_factor * num_bins
    u_y, u_z, t = observables.volume_axes(qy, qz, nt, delta_f=delta_f, spacing_lambda=SPACING)
    np.testing.assert_allclose(u_y, np.fft.fftshift(np.fft.fftfreq(qy)) / SPACING, atol=1e-15)
    np.testing.assert_allclose(u_z, np.fft.fftshift(np.fft.fftfreq(qz)) / SPACING, atol=1e-15)
    np.testing.assert_allclose(t, np.arange(nt) / (nt * delta_f), atol=1e-18)

    centred_y = np.arange(cols) - (cols - 1) / 2.0
    centred_z = (rows - 1) / 2.0 - np.arange(rows)
    delay_matrix = np.exp(
        1j * 2.0 * np.pi * np.outer(t, (np.arange(num_bins) - num_bins // 2) * delta_f)
    )
    phase = np.exp(
        -1j
        * 2.0
        * np.pi
        * SPACING
        * (
            u_y[:, None, None, None] * centred_y[None, None, None, :]
            + u_z[None, :, None, None] * centred_z[None, None, :, None]
        )
    )

    for window in (None, "taylor"):
        w_r, w_c, w_n = _windows(window, rows, cols, num_bins)
        volume = observables.angle_delay_volume(data, window, (angle_factor, delay_factor))
        weighted = data[0, 0, 0] * w_r[:, None, None] * w_c[None, :, None] * w_n[None, None, :]
        direct = np.einsum("ijrc,rcn,tn->ijt", phase, weighted, delay_matrix) / np.sqrt(
            rows * cols * num_bins
        )
        centre = observables.aperture_centre_phase(
            u_y, u_z, rows=rows, cols=cols, spacing_lambda=SPACING
        )
        centred = volume[0, 0, 0] * centre[..., None]
        np.testing.assert_allclose(centred, direct, rtol=0, atol=1e-12 * np.max(np.abs(direct)))


# --------------------------------------------------------------------------------------
# 3. On-grid plane wave collapses to a single centred cell
# --------------------------------------------------------------------------------------
def test_on_grid_plane_wave_is_a_delta() -> None:
    rows = cols = 8
    num_bins = 16
    delta_f = 100e6 / num_bins
    alpha = 0.7 - 0.3j
    tau = 5.0 / (num_bins * delta_f)
    for u_y, u_z in ((0.25, -0.5), (-0.75, 0.25)):
        u_x = np.sqrt(1.0 - u_y**2 - u_z**2)
        direction = np.array([u_x, u_y, u_z])
        data = _wave((1, 1, 2, rows, cols, num_bins), direction, tau, alpha, delta_f=delta_f)
        volume = observables.angle_delay_volume(data)
        axis_y, axis_z, axis_t = observables.volume_axes(
            cols, rows, num_bins, delta_f=delta_f, spacing_lambda=SPACING
        )
        centre = observables.aperture_centre_phase(
            axis_y, axis_z, rows=rows, cols=cols, spacing_lambda=SPACING
        )
        centred = volume[0, 0, 0] * centre[..., None]

        iy = int(np.argmin(np.abs(axis_y - u_y)))
        iz = int(np.argmin(np.abs(axis_z - u_z)))
        it = int(np.argmin(np.abs(axis_t - tau)))
        assert abs(axis_y[iy] - u_y) < 1e-15
        assert abs(axis_z[iz] - u_z) < 1e-15
        np.testing.assert_allclose(
            centred[iy, iz, it], alpha * np.sqrt(rows * cols * num_bins), atol=1e-12
        )

        mask = np.ones(centred.shape, dtype=bool)
        mask[iy, iz, it] = False
        assert np.max(np.abs(centred[mask])) < 1e-12 * abs(alpha)
        np.testing.assert_array_equal(volume[0, 0, 1], np.zeros_like(volume[0, 0, 1]))


# --------------------------------------------------------------------------------------
# 4. Off-grid plane wave peaks within half an oversampled cell
# --------------------------------------------------------------------------------------
def test_off_grid_plane_wave_peaks_at_true_direction() -> None:
    rows = cols = 8
    num_bins = 32
    delta_f = 100e6 / num_bins
    oversample = 8
    u_y, u_z, tau = 0.31, -0.17, 123.4e-9
    direction = _unit((np.sqrt(1.0 - u_y**2 - u_z**2), u_y, u_z))
    data = _wave((1, 1, 1, rows, cols, num_bins), direction, tau, 1.0 + 0.0j, delta_f=delta_f)
    volume = observables.angle_delay_volume(data, "taylor", (oversample, oversample))
    axis_y, axis_z, axis_t = observables.volume_axes(
        oversample * cols, oversample * rows, oversample * num_bins, delta_f=delta_f
    )
    iy, iz, it = np.unravel_index(np.argmax(np.abs(volume[0, 0, 0]) ** 2), volume.shape[3:])
    assert abs(axis_y[iy] - u_y) <= 0.5 / (oversample * cols * SPACING)
    assert abs(axis_z[iz] - u_z) <= 0.5 / (oversample * rows * SPACING)
    period = 1.0 / delta_f
    delay_error = abs(axis_t[it] - tau) % period
    delay_error = min(delay_error, period - delay_error)
    assert delay_error <= 0.5 / (oversample * num_bins * delta_f)


# --------------------------------------------------------------------------------------
# 5. Relation to the master rf_camera pipeline
# --------------------------------------------------------------------------------------
def test_relation_to_master_pipeline() -> None:
    rows = cols = 8
    num_bins = 16
    delta_f = 100e6 / num_bins
    rng = _rng(5)
    data = rng.standard_normal((1, 1, 1, rows, cols, num_bins)) + 1j * rng.standard_normal(
        (1, 1, 1, rows, cols, num_bins)
    )
    freqs = imaging.frequency_offsets(100e6, num_bins)
    for angle_factor in (1, 8):
        qy = qz = angle_factor * rows
        volume = observables.angle_delay_volume(data, oversample=(angle_factor, 1))
        raw = imaging.aperture_to_angular_fft(data[0, 0, 0], fft_rows=qz, fft_cols=qy)
        axis_y, axis_z, _ = observables.volume_axes(qy, qz, num_bins, delta_f=delta_f)
        centre = observables.aperture_centre_phase(axis_y, axis_z, rows=rows, cols=cols)
        centred = volume[0, 0, 0] * centre[..., None]

        ky, kz = calibration.direction_cosine_axes(
            fft_rows=qz,
            fft_cols=qy,
            horizontal_spacing_lambda=SPACING,
            vertical_spacing_lambda=SPACING,
        )
        np.testing.assert_allclose(ky, axis_y, atol=1e-15)
        np.testing.assert_allclose(kz[:-1], axis_z[1:], atol=1e-15)

        for iy in range(qy):
            for iz in range(qz):
                raw_slice = raw[(qz - iz) % qz, iy, :]
                delay_ifft = (
                    num_bins
                    * np.fft.ifft(raw_slice, n=num_bins, axis=-1)
                    * np.exp(-2j * np.pi * (num_bins // 2) * np.arange(num_bins) / num_bins)
                )
                np.testing.assert_allclose(
                    volume[0, 0, 0, iy, iz, :],
                    delay_ifft / np.sqrt(rows * cols * num_bins),
                    rtol=0,
                    atol=1e-12 * np.max(np.abs(volume)),
                )

        calibrated = calibration.calibrate_angular_cfr(
            raw,
            aperture_rows=rows,
            aperture_cols=cols,
            horizontal_spacing_lambda=SPACING,
            vertical_spacing_lambda=SPACING,
        )
        cir = delay.angular_cfr_to_delay(calibrated.cfr, freqs).cir
        scale = np.sqrt(num_bins / (rows * cols))
        for iz in range(1, qz):
            np.testing.assert_allclose(
                centred[:, iz, :],
                scale * cir[iz - 1, :, :],
                rtol=0,
                atol=1e-12 * np.max(np.abs(centred)),
            )


# --------------------------------------------------------------------------------------
# 6. Taylor window sidelobes and noise gain
# --------------------------------------------------------------------------------------
def test_taylor_window_sidelobes_and_gain() -> None:
    rows = cols = 8
    num_bins = 128
    delta_f = 100e6 / num_bins
    direction = _unit((np.sqrt(1.0 - 0.29**2 - 0.19**2), 0.29, -0.19))
    data = _wave((1, 1, 1, rows, cols, num_bins), direction, 37.0e-9, 1.0 + 0.0j, delta_f=delta_f)
    oversample = 8

    for window, bound in (("taylor", -34.0), (None, -14.0)):
        volume = observables.angle_delay_volume(data, window, (oversample, oversample))
        power = np.abs(volume[0, 0, 0]) ** 2
        iy, iz, it = np.unravel_index(np.argmax(power), power.shape)
        sidelobe_db = _largest_sidelobe_db(power[iy, iz, :], it)
        if window == "taylor":
            assert sidelobe_db <= bound
        else:
            assert sidelobe_db >= bound

    volume = observables.angle_delay_volume(data, "taylor", (oversample, oversample))
    power = np.abs(volume[0, 0, 0]) ** 2
    iy, iz, it = np.unravel_index(np.argmax(power), power.shape)
    assert _largest_sidelobe_db(power[:, iz, it], iy) <= -30.0

    w_r, w_c, w_n = _windows("taylor", rows, cols, num_bins)
    expected = float(np.mean(w_r**2) * np.mean(w_c**2) * np.mean(w_n**2))
    assert observables.volume_noise_gain(rows, cols, num_bins, "taylor") == pytest.approx(
        expected, abs=1e-15
    )
    assert observables.volume_noise_gain(rows, cols, num_bins, None) == 1.0


# --------------------------------------------------------------------------------------
# 7. Parseval: I equals the delay-integrated ID and the IP_W-derived I
# --------------------------------------------------------------------------------------
def test_parseval_i_from_id_equals_i_from_ip_w() -> None:
    shape = (2, 2, 2, 8, 8, 16)
    delta_f = 100e6 / shape[-1]
    rng = _rng(7)
    waves = [
        (_unit((0.9, 0.2, -0.3)), 40e-9, 1.0 + 0.4j),
        (_unit((0.8, -0.5, 0.4)), 71e-9, 0.6 - 0.2j),
    ]
    data = _sum_waves(shape, waves, delta_f=delta_f) + _noise(shape, 1e-3, rng)

    delay_power = observables.extract(data, "ID").data.sum(-1)
    intensity = observables.extract(data, "I").data
    phase_only = observables.extract(data, "IP_W").data
    intensity_phase = observables.extract(phase_only, "I").data
    scale = np.max(np.abs(intensity))
    np.testing.assert_allclose(delay_power, intensity, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(intensity_phase, intensity, rtol=0, atol=1e-12 * scale)

    direct = _direct_i(data)
    np.testing.assert_allclose(intensity, direct, rtol=0, atol=1e-12 * np.max(np.abs(direct)))


# --------------------------------------------------------------------------------------
# 8. Sync-invariant nodes agree between S and N
# --------------------------------------------------------------------------------------
def test_sync_invariant_nodes_equal_in_s_and_n() -> None:
    shape = (2, 2, 2, 8, 8, 16)
    delta_f = 100e6 / shape[-1]
    sigma2 = 1e-3
    rng = _rng(8)
    waves = [
        (_unit((0.9, 0.3, -0.2)), 55e-9, 1.2 + 0.1j),
        (_unit((0.4, -0.6, 0.5)), 90e-9, 0.5 - 0.3j),
    ]
    data = _sum_waves(shape, waves, delta_f=delta_f) + _noise(shape, sigma2, rng)
    num_views, num_bs = shape[0], shape[1]
    phi = rng.uniform(0.0, 2.0 * np.pi, (num_views, num_bs))
    tau = rng.uniform(0.0, 1.0 / delta_f, (num_views, num_bs))
    shifted = _gauge(data, phi, tau, delta_f)

    for name, params in (
        ("I", {}),
        ("I_n0", {}),
        ("P_W", {"noise_var": sigma2}),
        ("IP_W", {}),
        ("I-o", {}),
    ):
        first = observables.extract(data, name, params)
        second = observables.extract(shifted, name, params)
        np.testing.assert_array_equal(first.mask, second.mask)
        scale = max(1.0, float(np.max(np.abs(first.data))))
        np.testing.assert_allclose(first.data, second.data, rtol=0, atol=1e-12 * scale)


# --------------------------------------------------------------------------------------
# 9. D and ID shift circularly with the timing gauge
# --------------------------------------------------------------------------------------
def test_d_n_is_circular_shift_of_d_s() -> None:
    shape = (2, 2, 2, 8, 8, 32)
    num_bins = shape[-1]
    delta_f = 100e6 / num_bins
    period = 1.0 / delta_f
    delay_oversample = 8
    num_out = delay_oversample * num_bins
    sigma2 = 1e-3
    rng = _rng(9)
    alpha = np.sqrt(1000.0 * sigma2)
    directions = [
        (_unit((0.95, 0.25, 0.0)), 45e-9, alpha),
        (_unit((0.9, -0.25, 0.25)), 90e-9, alpha * 0.8),
        (_unit((0.85, 0.0, -0.5)), 150e-9, alpha * 0.6),
    ]
    data = _sum_waves(shape, directions, delta_f=delta_f) + _noise(shape, sigma2, rng)
    params = {
        "delta_f": delta_f,
        "delay_oversample": delay_oversample,
        "pfa": 1e-3,
        "max_returns": 2,
    }

    num_views, num_bs = shape[0], shape[1]
    shifts = rng.integers(0, num_out, size=(num_views, num_bs))
    gauge_shift = shifts * (period / num_out)
    shifted = _gauge(data, np.zeros((num_views, num_bs)), gauge_shift, delta_f)
    source = observables.extract(data, "D", params)
    target = observables.extract(shifted, "D", params)
    np.testing.assert_array_equal(source.mask, target.mask)
    difference = np.mod(
        target.data - (source.data + gauge_shift[:, :, None, None, None, None]), period
    )
    difference = np.minimum(difference, period - difference)
    assert np.max(difference[source.mask]) <= 1e-9 * period / num_out

    native_shift = rng.integers(0, num_bins, size=(num_views, num_bs))
    shifted_id = _gauge(
        data, np.zeros((num_views, num_bs)), native_shift / (num_bins * delta_f), delta_f
    )
    source_id = observables.extract(data, "ID", {}).data
    target_id = observables.extract(shifted_id, "ID", {}).data
    scale = float(np.max(np.abs(source_id)))
    for view in range(num_views):
        for bs in range(num_bs):
            np.testing.assert_allclose(
                target_id[view, bs],
                np.roll(source_id[view, bs], int(native_shift[view, bs]), axis=-1),
                rtol=0,
                atol=1e-12 * scale,
            )

    random_tau = rng.uniform(0.0, period, (num_views, num_bs))
    shifted_random = _gauge(data, np.zeros((num_views, num_bs)), random_tau, delta_f)
    source = observables.extract(data, "D", params)
    target = observables.extract(shifted_random, "D", params)
    axis_y, axis_z, _ = observables.volume_axes(shape[4], shape[3], num_bins, delta_f=delta_f)
    tolerance = 0.01 / (num_bins * delta_f)
    for direction, _, _ in directions:
        iy = int(np.argmin(np.abs(axis_y - direction[1])))
        iz = int(np.argmin(np.abs(axis_z - direction[2])))
        hemisphere = 0 if direction[0] >= 0.0 else 1
        for view in range(num_views):
            for bs in range(num_bs):
                s_value = source.data[view, bs, hemisphere, iy, iz, 0]
                n_value = target.data[view, bs, hemisphere, iy, iz, 0]
                delta = np.mod(n_value - (s_value + random_tau[view, bs]), period)
                delta = min(delta, period - delta)
                assert delta < tolerance


# --------------------------------------------------------------------------------------
# 10. P and IP are tau-invariant and rotate with the phase gauge
# --------------------------------------------------------------------------------------
def test_p_and_ip_are_tau_invariant_and_rotate_with_phi() -> None:
    shape = (2, 2, 1, 4, 4, 16)
    delta_f = 100e6 / shape[-1]
    rng = _rng(10)
    data = _sum_waves(shape, [(_unit((1.0, 0.3, -0.2)), 30e-9, 1.0 + 0.5j)], delta_f=delta_f)
    data = data + _noise(shape, 1e-3, rng)
    num_views, num_bs = shape[0], shape[1]
    params = {"mask_k": 0.0}
    period = 1.0 / delta_f

    tau = rng.uniform(0.0, period, (num_views, num_bs))
    tau_shifted = _gauge(data, np.zeros((num_views, num_bs)), tau, delta_f)
    for name in ("P", "IP"):
        first = observables.extract(data, name, params).data
        second = observables.extract(tau_shifted, name, params).data
        np.testing.assert_allclose(
            second, first, rtol=0, atol=1e-12 * max(1.0, np.max(np.abs(first)))
        )

    phi = rng.uniform(0.0, 2.0 * np.pi, (num_views, num_bs))
    phi_shifted = data * np.exp(1j * phi)[:, :, None, None, None, None]
    rotation = np.exp(1j * phi)[:, :, None, None, None]
    for name in ("P", "IP"):
        first = observables.extract(data, name, params).data
        second = observables.extract(phi_shifted, name, params).data
        np.testing.assert_allclose(
            second, first * rotation, rtol=0, atol=1e-12 * max(1.0, np.max(np.abs(first)))
        )


# --------------------------------------------------------------------------------------
# 11. Power observables are phase-invariant
# --------------------------------------------------------------------------------------
def test_power_observables_are_phi_invariant() -> None:
    shape = (2, 1, 2, 4, 4, 16)
    num_bins = shape[-1]
    delta_f = 100e6 / num_bins
    period = 1.0 / delta_f
    rng = _rng(11)
    data = _sum_waves(
        shape,
        [(_unit((0.9, 0.3, -0.2)), 40e-9, 1.0 + 0.5j), (_unit((0.5, -0.4, 0.6)), 120e-9, 0.4)],
        delta_f=delta_f,
    )
    data = data + _noise(shape, 1e-3, rng)
    num_views, num_bs = shape[0], shape[1]
    phi = rng.uniform(0.0, 2.0 * np.pi, (num_views, num_bs))
    shifted = data * np.exp(1j * phi)[:, :, None, None, None, None]
    params = {"delta_f": delta_f, "pfa": 1e-3, "max_returns": 2}

    for name in ("I", "I_n0", "ID", "T", "I-o", "ID-o", "IP-o"):
        first = observables.extract(data, name, params).data
        second = observables.extract(shifted, name, params).data
        scale = max(1.0, float(np.max(np.abs(first))))
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-12 * scale)

    for name in ("D", "D-o"):
        first = observables.extract(data, name, params)
        second = observables.extract(shifted, name, params)
        np.testing.assert_array_equal(first.mask, second.mask)
        difference = np.mod(first.data - second.data, period)
        difference = np.minimum(difference, period - difference)
        assert np.max(difference[first.mask]) <= 1e-9 * period / (8 * num_bins)


# --------------------------------------------------------------------------------------
# 12. Omni nodes derive from IDP-o
# --------------------------------------------------------------------------------------
def test_omni_nodes_derive_from_idp_o() -> None:
    shape = (2, 1, 2, 4, 4, 16)
    delta_f = 100e6 / shape[-1]
    rng = _rng(12)
    data = _sum_waves(shape, [(_unit((0.9, 0.3, -0.2)), 50e-9, 1.0 + 0.5j)], delta_f=delta_f)
    data = data + _noise(shape, 1e-3, rng)
    phases = rng.uniform(0.0, 2.0 * np.pi, shape[:5])[..., None]
    twisted = data * np.exp(1j * phases)

    for name in ("IDP-o", "IP-o", "ID-o", "I-o"):
        first = observables.extract(data, name, {}).data
        second = observables.extract(twisted, name, {}).data
        scale = max(1.0, float(np.max(np.abs(first))))
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-12 * scale)

    idp_o = observables.extract(data, "IDP-o", {}).data
    reference = observables.extract(data, "ID-o", {}).data
    np.testing.assert_allclose(
        observables.extract(idp_o, "ID-o", {}).data,
        reference,
        rtol=0,
        atol=1e-12 * np.max(np.abs(reference)),
    )
    reference_scalar = observables.extract(data, "I-o", {}).data
    np.testing.assert_allclose(
        observables.extract(idp_o, "I-o", {}).data,
        reference_scalar,
        rtol=0,
        atol=1e-12 * np.max(np.abs(reference_scalar)),
    )
    np.testing.assert_allclose(
        reference_scalar, reference.sum(-1), rtol=0, atol=1e-12 * np.max(np.abs(reference_scalar))
    )
    np.testing.assert_allclose(
        reference_scalar,
        (np.abs(data) ** 2).sum(axis=(3, 4, 5)),
        rtol=0,
        atol=1e-12 * np.max(np.abs(reference_scalar)),
    )
    assert observables.extract(data, "P-o", {}).data.shape == shape[:3] + (0,)


# --------------------------------------------------------------------------------------
# 13. D returns resolve two paths in one pixel
# --------------------------------------------------------------------------------------
def test_d_returns_locate_paths() -> None:
    rows = cols = 8
    num_bins = 64
    delta_f = 100e6 / num_bins
    sigma2 = 1e-3
    rng = _rng(13)
    alpha = np.sqrt(1000.0 * sigma2)
    direction = _unit((np.sqrt(1.0 - 0.25**2), 0.25, 0.0))
    delays = [60.3e-9, 181.7e-9]
    data = _sum_waves(
        (1, 1, 1, rows, cols, num_bins),
        [(direction, delays[0], alpha), (direction, delays[1], alpha / np.sqrt(2.0))],
        delta_f=delta_f,
    )
    data = data + _noise(data.shape, sigma2, rng)
    axis_y, axis_z, _ = observables.volume_axes(cols, rows, num_bins, delta_f=delta_f)
    iy = int(np.argmin(np.abs(axis_y - direction[1])))
    iz = int(np.argmin(np.abs(axis_z - direction[2])))
    tolerance = 0.02 / (num_bins * delta_f)
    params = {"delta_f": delta_f}

    two = observables.extract(data, "D", {**params, "max_returns": 2}).data[0, 0, 0, iy, iz]
    assert abs(two[0] - delays[0]) < tolerance
    assert abs(two[1] - delays[1]) < tolerance

    one = observables.extract(data, "D", {**params, "max_returns": 1}).data[0, 0, 0, iy, iz]
    assert abs(one[0] - delays[0]) < tolerance

    three = observables.extract(data, "D", params).data[0, 0, 0, iy, iz]
    assert abs(three[0] - delays[0]) < tolerance
    assert abs(three[1] - delays[1]) < tolerance


# --------------------------------------------------------------------------------------
# 14. CFAR threshold, parabolic refinement, ordering and OS noise level
# --------------------------------------------------------------------------------------
def test_cfar_threshold_and_parabolic_refinement() -> None:
    threshold = float(np.log(1000.0))
    profile = np.zeros(32, dtype=np.float64)
    profile[10] = 1.01 * threshold
    detected = observables.cfar_returns(profile, pfa=1e-3, noise_power=1.0)
    assert bool(detected.mask[0])
    np.testing.assert_allclose(detected.position[0], 10.0, atol=1e-12)
    np.testing.assert_allclose(detected.power[0], 1.01 * threshold, rtol=0, atol=1e-12)

    profile[10] = 0.99 * threshold
    missed = observables.cfar_returns(profile, pfa=1e-3, noise_power=1.0)
    assert not bool(missed.mask[0])
    assert np.isnan(missed.position[0])

    for k0 in (3.7, 16.0 - 0.3):
        nt = 16
        parabola = _circular_parabola(nt, k0, 300.0)
        refined = observables.cfar_returns(parabola, pfa=1e-3, noise_power=1.0)
        np.testing.assert_allclose(refined.position[0], k0, atol=1e-12)
        np.testing.assert_allclose(refined.power[0], 300.0, rtol=0, atol=1e-12)

    ordered = np.zeros(16, dtype=np.float64)
    ordered[3] = 10.0
    ordered[9] = 8.0
    result = observables.cfar_returns(ordered, pfa=1e-3, max_returns=3, noise_power=1.0)
    assert result.mask.tolist() == [True, True, False]
    np.testing.assert_allclose(result.power[:2], [10.0, 8.0], rtol=0, atol=1e-12)
    assert np.isnan(result.position[2]) and np.isnan(result.power[2])

    default = observables.cfar_returns(ordered, pfa=1e-3, max_returns=2)
    np.testing.assert_allclose(default.noise_power, np.median(ordered) / np.log(2.0), atol=1e-15)


# --------------------------------------------------------------------------------------
# 15. Noise variance estimate within 10 percent
# --------------------------------------------------------------------------------------
def test_noise_var_estimate_within_10_percent() -> None:
    rows = cols = 8
    num_bins = 128
    delta_f = 100e6 / num_bins
    sigma2 = 1e-3
    rng = _rng(15)
    shape = (4, 1, 2, rows, cols, num_bins)
    peak = 10.0**6.9 * sigma2 / (rows * cols * num_bins)
    alpha = np.sqrt(peak)
    waves: list[tuple[np.ndarray, float, complex]] = []
    for view in range(shape[0]):
        distance = 30.0 + 5.0 * view
        los = _unit((1.0, 0.2, 0.1))
        bounce = _unit((1.0, 0.2, -0.1))
        waves.append((los, distance / SPEED_OF_LIGHT, alpha))
        waves.append((bounce, (distance + 1.0) / SPEED_OF_LIGHT, alpha / np.sqrt(2.0)))
    waves.append((_unit((0.6, 0.6, 0.4)), 90.0 / SPEED_OF_LIGHT, alpha * 10.0 ** (-20.0 / 20.0)))
    waves.append((_unit((-0.7, 0.5, 0.3)), 150.0 / SPEED_OF_LIGHT, alpha * 10.0 ** (-26.0 / 20.0)))
    data = _sum_waves(shape, waves, delta_f=delta_f) + _noise(shape, sigma2, rng)

    estimate = observables.noise_var_estimate(data, 180.0, delta_f=delta_f)
    assert abs(estimate - sigma2) / sigma2 < 0.1
    with pytest.raises(ValueError):
        observables.noise_var_estimate(data, SPEED_OF_LIGHT / delta_f, delta_f=delta_f)


# --------------------------------------------------------------------------------------
# 16. PHAT mask and the noise-variance requirement
# --------------------------------------------------------------------------------------
def test_phat_mask_and_noise_var_required() -> None:
    shape = (1, 1, 2, 4, 4, 16)
    sigma2 = 1e-3
    rng = _rng(16)
    data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    data[0, 0, 0, 0, 0, 0] = 0.5 * np.sqrt(sigma2) * np.exp(0.7j)
    data[0, 0, 0, 1, 1, 1] = 2.0 * np.sqrt(sigma2) * np.exp(1.1j)

    with pytest.raises(ValueError):
        observables.extract(data, "DP", {})

    result = observables.extract(data, "DP", {"noise_var": sigma2})
    magnitude = np.abs(data)
    expected = (magnitude > 0.0) & (magnitude >= np.sqrt(sigma2))
    np.testing.assert_array_equal(result.mask, expected)
    np.testing.assert_allclose(np.abs(result.data[expected]), 1.0, atol=1e-12)
    assert np.all(result.data[~expected] == 0.0)
    assert not bool(result.mask[0, 0, 0, 0, 0, 0])
    assert bool(result.mask[0, 0, 0, 1, 1, 1])

    free = observables.extract(data, "DP", {"mask_k": 0.0})
    assert bool(free.mask[0, 0, 0, 1, 1, 1])
    assert bool(free.mask[0, 0, 0, 0, 0, 0])


# --------------------------------------------------------------------------------------
# 17. Every node: shapes, dtypes, finite data and validation
# --------------------------------------------------------------------------------------
_DOMAIN_LABELS: dict[str, str] = {
    "I": "beam",
    "I_n0": "beam",
    "D": "returns",
    "D_PHAT": "returns",
    "T": "beam_delay",
    "P": "element",
    "P_W": "element_freq",
    "ID": "beam_delay",
    "IP": "element",
    "IP_W": "element_freq",
    "DP": "element_freq",
    "IDP": "element_freq",
    "IDP-o": "element_freq",
    "DP-o": "element_freq",
    "IP-o": "element_freq",
    "P-o": "empty",
    "ID-o": "delay",
    "I-o": "scalar",
    "D-o": "returns",
    "IDP-1el": "freq",
    "DP-1el": "freq",
    "PxK": "element_freq",
    "IPxK": "element_freq",
}


def _expected_node(name: str) -> tuple[np.dtype, tuple[int, ...]]:
    views, bs, hemispheres = 2, 1, 2
    rows = cols = 4
    num_bins = 16
    returns = 3
    bins = 4
    if name in ("I", "I_n0"):
        return np.dtype(np.float64), (views, bs, hemispheres, cols, rows)
    if name in ("D", "D_PHAT"):
        return np.dtype(np.float64), (views, bs, hemispheres, cols, rows, returns)
    if name == "T":
        return np.dtype(np.float64), (views, bs, hemispheres, cols, rows, num_bins)
    if name == "P":
        return np.dtype(np.complex128), (views, bs, hemispheres, rows, cols)
    if name == "IP":
        return np.dtype(np.complex128), (views, bs, hemispheres, rows, cols)
    if name == "ID":
        return np.dtype(np.float64), (views, bs, hemispheres, cols, rows, num_bins)
    if name in ("P_W", "IP_W", "DP", "IDP", "IDP-o", "DP-o"):
        return np.dtype(np.complex128), (views, bs, hemispheres, rows, cols, num_bins)
    if name == "IP-o":
        return np.dtype(np.float64), (views, bs, hemispheres, rows, cols, num_bins)
    if name == "P-o":
        return np.dtype(np.float64), (views, bs, hemispheres, 0)
    if name == "ID-o":
        return np.dtype(np.float64), (views, bs, hemispheres, num_bins)
    if name == "I-o":
        return np.dtype(np.float64), (views, bs, hemispheres)
    if name in ("D-o",):
        return np.dtype(np.float64), (views, bs, returns)
    if name in ("IDP-1el", "DP-1el"):
        return np.dtype(np.complex128), (views, bs, hemispheres, num_bins)
    if name in ("IPxK", "PxK"):
        return np.dtype(np.complex128), (views, bs, hemispheres, rows, cols, bins)
    raise AssertionError(f"unexpected node {name}")


@pytest.mark.parametrize("name", list(NODE_NAMES) + list(NODE_ALIASES))
def test_every_node_shapes_dtypes_and_finite(name: str) -> None:
    shape = (2, 1, 2, 4, 4, 16)
    delta_f = 100e6 / shape[-1]
    rng = _rng(17)
    data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    params = {
        "noise_var": 1e-3,
        "delta_f": delta_f,
        "pfa": 1e-3,
        "max_returns": 3,
        "delay_oversample": 8,
        "bins": [2, 6, 10, 14],
        "element": (1, 1),
        "spacing_lambda": 0.5,
        "mask_k": 1.0,
    }
    canonical = NODE_ALIASES.get(name, name)
    expected_dtype, expected_shape = _expected_node(canonical)
    result = observables.extract(data, name, params)
    assert result.data.dtype == expected_dtype
    assert result.data.shape == expected_shape
    assert result.mask.dtype == np.bool_
    assert result.mask.shape == expected_shape
    if result.data.size:
        assert bool(np.all(np.isfinite(result.data[result.mask])))
    assert result.meta["node"] == canonical
    assert result.meta["domain"] == _DOMAIN_LABELS[canonical]
    assert result.noise_var == pytest.approx(1e-3)

    with pytest.raises(ValueError):
        observables.extract(data, "not-a-node", params)
    with pytest.raises(ValueError):
        observables.extract(data, "I", {"unknown_key": 1})


# --------------------------------------------------------------------------------------
# 18. Partial-D bins and the one-element nodes
# --------------------------------------------------------------------------------------
def test_partial_d_and_one_element_nodes() -> None:
    shape = (2, 1, 2, 4, 4, 16)
    rng = _rng(18)
    data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    partial = observables.extract(data, "IPxK", {})
    assert partial.meta["bins"] == [2, 6, 10, 14]
    np.testing.assert_array_equal(partial.data, data[..., [2, 6, 10, 14]])

    element = observables.extract(data, "IDP-1el", {})
    np.testing.assert_array_equal(element.data, data[..., 3, 3, :])

    distribution = observables.extract(data, "T", {})
    totals = distribution.data.sum(-1)
    valid = distribution.mask.any(-1)
    np.testing.assert_allclose(totals[valid], 1.0, atol=1e-12)
    assert np.all(distribution.data[~distribution.mask] == 0.0)


# --------------------------------------------------------------------------------------
# 19. IP_W / P_W share one unknown phase per (v, b, n) across both hemispheres
# --------------------------------------------------------------------------------------
def test_ip_w_keeps_relative_phases_within_each_bin() -> None:
    shape = (2, 1, 2, 4, 4, 16)
    num_bins = shape[-1]
    delta_f = 100e6 / num_bins
    noise_var = 1e-3
    rng = _rng(20)
    waves = [
        (_unit((0.9, 0.3, -0.2)), 40e-9, 1.0 + 0.5j),
        (_unit((-0.8, -0.4, 0.5)), 90e-9, 0.7 - 0.2j),
        (_unit((0.7, -0.5, -0.3)), 130e-9, 0.4 + 0.3j),
        (_unit((-0.6, 0.6, 0.4)), 70e-9, 0.5 + 0.1j),
    ]
    data = _sum_waves(shape, waves, delta_f=delta_f) + _noise(shape, noise_var, rng)
    scale = float(np.max(np.abs(data)))

    ip_w = observables.extract(data, "IP_W", {}).data
    np.testing.assert_allclose(np.abs(ip_w), np.abs(data), rtol=1e-12, atol=1e-12 * scale)

    product = ip_w * np.conj(data)
    reference = product[:, :, 0, 0, 0, :]
    reference = reference / np.abs(reference)
    phase = np.angle(product * np.conj(reference)[:, :, None, None, None, :])
    np.testing.assert_allclose(phase, 0.0, atol=1e-12)

    num_views, num_bs = shape[0], shape[1]
    for view in range(num_views):
        for bs in range(num_bs):
            for n in range(num_bins):
                magnitude = np.abs(data[view, bs, :, :, :, n])
                hemisphere, row, col = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
                sample = ip_w[view, bs, hemisphere, row, col, n]
                assert abs(sample.imag) <= 1e-12 * scale
                assert sample.real > 0.0

    p_w = observables.extract(data, "P_W", {"noise_var": noise_var, "mask_k": 0.0}).data
    np.testing.assert_allclose(p_w, ip_w / np.abs(ip_w), rtol=0, atol=1e-12 * scale)

    phases = rng.uniform(0.0, 2.0 * np.pi, (num_views, num_bs, num_bins))
    twisted = data * np.exp(1j * phases)[:, :, None, None, None, :]
    ip_w_twisted = observables.extract(twisted, "IP_W", {}).data
    p_w_twisted = observables.extract(twisted, "P_W", {"noise_var": noise_var, "mask_k": 0.0}).data
    np.testing.assert_allclose(ip_w_twisted, ip_w, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(p_w_twisted, p_w, rtol=0, atol=1e-12 * scale)


# --------------------------------------------------------------------------------------
# 20. I_n0 is the DC (n0 = N // 2) beam power
# --------------------------------------------------------------------------------------
def test_i_n0_is_the_dc_beam_power() -> None:
    rows, cols, num_bins = 4, 6, 16
    shape = (1, 1, 1, rows, cols, num_bins)
    rng = _rng(21)
    data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    n0 = num_bins // 2
    fy = np.fft.fftshift(np.fft.fftfreq(cols))
    fz = np.fft.fftshift(np.fft.fftfreq(rows))
    column = np.arange(cols, dtype=np.float64)
    row = np.arange(rows, dtype=np.float64)
    col_matrix = np.exp(-2j * np.pi * np.outer(fy, column))
    row_matrix = np.exp(2j * np.pi * np.outer(fz, row))
    beam = np.einsum("yc,rc,zr->yz", col_matrix, data[0, 0, 0, :, :, n0], row_matrix) / np.sqrt(
        rows * cols
    )

    result = observables.extract(data, "I_n0", {})
    expected = np.abs(beam) ** 2
    np.testing.assert_allclose(
        result.data[0, 0, 0], expected, rtol=1e-12, atol=1e-12 * float(np.max(expected))
    )
    assert result.meta["n0"] == n0

    dc_only = np.zeros_like(data)
    dc_only[..., n0] = data[..., n0]
    np.testing.assert_allclose(
        observables.extract(dc_only, "I_n0", {}).data,
        result.data,
        rtol=0,
        atol=1e-12 * float(np.max(expected)),
    )
    without_dc = data.copy()
    without_dc[..., n0] = 0.0
    np.testing.assert_allclose(observables.extract(without_dc, "I_n0", {}).data, 0.0, atol=1e-24)


# --------------------------------------------------------------------------------------
# 21. D, D_PHAT and D-o follow their definitions
# --------------------------------------------------------------------------------------
def test_d_nodes_match_their_definitions() -> None:
    shape = (2, 1, 2, 4, 4, 16)
    num_bins = shape[-1]
    delta_f = 100e6 / num_bins
    sigma2 = 1e-3
    delay_oversample = 8
    num_out = delay_oversample * num_bins
    period = 1.0 / delta_f
    tolerance = 1e-12 * period
    rng = _rng(22)
    alpha = np.sqrt(1000.0 * sigma2)
    waves = [
        (_unit((0.9, 0.3, -0.2)), 40e-9, alpha),
        (_unit((-0.8, -0.4, 0.5)), 90e-9, alpha * 0.7),
        (_unit((0.7, -0.5, -0.3)), 130e-9, alpha * 0.5),
    ]
    data = _sum_waves(shape, waves, delta_f=delta_f) + _noise(shape, sigma2, rng)
    params = {
        "delta_f": delta_f,
        "noise_var": sigma2,
        "pfa": 1e-3,
        "max_returns": 3,
        "delay_oversample": delay_oversample,
    }

    volume = observables.angle_delay_volume(data, "taylor", (1, delay_oversample))
    expected_d = observables.cfar_returns(np.abs(volume) ** 2, 1e-3, 3)
    result_d = observables.extract(data, "D", params)
    np.testing.assert_array_equal(result_d.mask, expected_d.mask)
    np.testing.assert_allclose(
        result_d.data, expected_d.position * (period / num_out), rtol=0, atol=tolerance
    )

    magnitude = np.abs(data)
    unit = np.zeros_like(data)
    nonzero = magnitude > 0.0
    unit[nonzero] = data[nonzero] / magnitude[nonzero]
    phat_data = np.where(magnitude >= np.sqrt(sigma2), unit, 0.0)
    phat_volume = observables.angle_delay_volume(phat_data, "taylor", (1, delay_oversample))
    expected_p = observables.cfar_returns(np.abs(phat_volume) ** 2, 1e-3, 3)
    result_p = observables.extract(data, "D_PHAT", params)
    np.testing.assert_array_equal(result_p.mask, expected_p.mask)
    np.testing.assert_allclose(
        result_p.data, expected_p.position * (period / num_out), rtol=0, atol=tolerance
    )
    if np.array_equal(result_p.mask, result_d.mask):
        shared = result_p.mask & result_d.mask
        position_gap = float(np.max(np.abs(result_p.data[shared] - result_d.data[shared])))
    else:
        position_gap = np.inf
    assert position_gap > 1e-3 * period / num_out

    window = observables.taylor_window(num_bins)
    offsets = np.arange(num_bins) - num_bins // 2
    sample = np.arange(num_out)
    kernel = np.exp(2j * np.pi * np.outer(offsets, sample) / num_out)
    transform = np.einsum("vbhrcn,nk->vbhrck", data * window, kernel) / np.sqrt(num_bins)
    profile = (np.abs(transform) ** 2).sum(axis=(2, 3, 4))
    expected_o = observables.cfar_returns(profile, 1e-3, 3)
    result_o = observables.extract(data, "D-o", params)
    np.testing.assert_array_equal(result_o.mask, expected_o.mask)
    np.testing.assert_allclose(
        result_o.data, expected_o.position * (period / num_out), rtol=0, atol=tolerance
    )


# --------------------------------------------------------------------------------------
# 22. Noise estimate rejects propagating clutter at all delays
# --------------------------------------------------------------------------------------
def test_noise_var_estimate_rejects_propagating_clutter() -> None:
    rows = cols = 8
    num_bins = 128
    delta_f = 100e6 / num_bins
    period = 1.0 / delta_f
    sigma2 = 1e-3
    shape = (4, 1, 2, rows, cols, num_bins)
    rng = _rng(23)
    data = _noise(shape, sigma2, rng)

    # Diffuse clutter in propagating directions at every delay, including beyond
    # max_path, so only the evanescent-angle condition can reject it.
    amplitude = np.sqrt(30.0 * sigma2 / 50.0)
    for wave_index in range(100):
        radius = 0.5 * np.sqrt(rng.uniform())
        theta = rng.uniform(0.0, 2.0 * np.pi)
        u_y, u_z = radius * np.cos(theta), radius * np.sin(theta)
        sign = 1.0 if wave_index % 2 == 0 else -1.0
        direction = _unit((sign * np.sqrt(1.0 - u_y**2 - u_z**2), u_y, u_z))
        delay = rng.uniform(0.0, period)
        alpha = amplitude * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
        data += _wave(shape, direction, delay, alpha, delta_f=delta_f)

    estimate = observables.noise_var_estimate(data, 180.0, delta_f=delta_f)
    assert abs(estimate - sigma2) / sigma2 < 0.1


# --------------------------------------------------------------------------------------
# 23. Optional benchmark
# --------------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("RF_TOMO_BENCH") != "1", reason="set RF_TOMO_BENCH=1")
def test_benchmark_volume() -> None:
    rng = _rng(19)
    data = rng.standard_normal((2, 1, 2, 8, 8, 128)) + 1j * rng.standard_normal(
        (2, 1, 2, 8, 8, 128)
    )
    start = time.perf_counter()
    volume = observables.angle_delay_volume(data, "taylor", (8, 8))
    elapsed = time.perf_counter() - start
    print(f"angle_delay_volume [2,1,2,8,8,128] taylor (8,8): {elapsed:.3f} s, shape {volume.shape}")
