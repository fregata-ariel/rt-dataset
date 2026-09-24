"""Convention tests pinning the tomography and master RF-camera pipelines (Phase 0)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as SPEED_OF_LIGHT

from plateau_rt.domain.rf_camera import calibration, delay, imaging
from plateau_rt.domain.rf_tomography import observables
from plateau_rt.domain.rf_tomography.antenna import bs_orientation
from plateau_rt.domain.rf_tomography.forward_exact import atom_cfr
from plateau_rt.domain.rf_tomography.geometry import (
    CaptureGeometry,
    planar_element_offsets,
    rotations_from_orientations,
)

F_C = 3.5e9
BANDWIDTH = 100e6
NUM_BINS = 16
ROWS = 8
COLS = 8
SPACING = 0.5
Q = 64
ANGLE_OS = 8
DELAY_OS = 8
FIXTURE = Path(__file__).parent / "fixtures" / "rf_tomography" / "sionna_mock_los.npz"

UE_POS = np.array([3.0, -2.0, 1.5], dtype=np.float64)
UE_ORIENTATION = (0.7, -0.2, 0.3)
BS_POS = np.array([40.0, 30.0, 20.0], dtype=np.float64)

DIRECTIONS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (35.0, 20.0),
    (-50.0, -30.0),
    (84.0, 10.0),
    (88.5, -5.0),
    (91.5, 5.0),
    (150.0, 25.0),
    (-120.0, -40.0),
    (180.0, 0.0),
    (10.0, 80.0),
]
DELAY_BINS: list[float] = [1.3071, 5.4219, 15.7071, 18.2963, 57.1271]
CASES: list[tuple[float, float, float]] = [
    (az, el, tb) for az, el in DIRECTIONS for tb in DELAY_BINS
]
CASE_IDS: list[str] = [f"az{az:g}_el{el:g}-t{tb:g}" for az, el, tb in CASES]
BV_CASES: list[tuple[float, float, float, int]] = [
    (25.0, 15.0, 12.0, 1),
    (-140.0, 20.0, 30.0, 2),
    (70.0, -10.0, 70.0, 2),
]
BV_IDS: list[str] = [f"az{az:g}_el{el:g}-r{r2:g}" for az, el, r2, _ in BV_CASES]


def _local_unit(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Return the UE-local unit vector for (azimuth, elevation) in degrees."""
    az = float(np.deg2rad(azimuth_deg))
    el = float(np.deg2rad(elevation_deg))
    return np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        dtype=np.float64,
    )


def _dirichlet(arg: np.ndarray, length: int) -> np.ndarray:
    """Return sin(pi L a)/sin(pi a) with the limit where the denominator vanishes."""
    a = np.asarray(arg, dtype=np.float64)
    numerator = np.sin(np.pi * float(length) * a)
    denominator = np.sin(np.pi * a)
    small = np.abs(denominator) < 1e-12
    out = np.empty_like(a, dtype=np.float64)
    safe = ~small
    out[safe] = numerator[safe] / denominator[safe]
    out[small] = float(length) * np.cos(np.pi * float(length) * a[small]) / np.cos(np.pi * a[small])
    return out


def _delay_kernel(arg: np.ndarray, num_bins: int = NUM_BINS) -> np.ndarray:
    """Return exp(-j pi x) dirichlet(x, N), the closed-form delay kernel."""
    x = np.asarray(arg, dtype=np.float64)
    return np.exp(-1j * np.pi * x) * _dirichlet(x, num_bins)


def _circular_distance(first: float, second: float, period: float) -> float:
    """Return the shortest distance between two scalars on a circle of period."""
    diff = abs((float(first) - float(second)) % float(period))
    return float(min(diff, float(period) - diff))


def _circular_dist_array(axis: np.ndarray, value: float, period: float) -> np.ndarray:
    """Return the circular distance of each axis entry to a scalar value."""
    diff = np.mod(np.asarray(axis, dtype=np.float64) - float(value), float(period))
    return np.minimum(diff, float(period) - diff)


def _nearest_index(axis: np.ndarray, value: float, period: float) -> int:
    """Return the index of the circularly nearest axis entry to a value."""
    return int(np.argmin(_circular_dist_array(axis, value, period)))


def _synthetic_geometry() -> CaptureGeometry:
    """Return the common V=1 B=1 synthetic capture geometry of the brief."""
    return CaptureGeometry.from_orientations(
        UE_POS[None, :],
        np.array([UE_ORIENTATION], dtype=np.float64),
        BS_POS[None, :],
        f_c=F_C,
        bandwidth=BANDWIDTH,
        num_bins=NUM_BINS,
    )


def _synthetic_rotation() -> np.ndarray:
    """Return the world_from_local rotation of the synthetic UE orientation."""
    return calibration.rotation_matrix(UE_ORIENTATION)


def _master_chain(
    aperture: np.ndarray, freq_offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the master chain on one [R, C, N] aperture, returning cir/ky/kz/t."""
    raw = imaging.aperture_to_angular_fft(aperture, fft_rows=Q, fft_cols=Q)
    cal = calibration.calibrate_angular_cfr(
        raw,
        aperture_rows=ROWS,
        aperture_cols=COLS,
        horizontal_spacing_lambda=SPACING,
        vertical_spacing_lambda=SPACING,
    )
    vol = delay.angular_cfr_to_delay(cal.cfr, freq_offsets)
    return vol.cir, cal.ky_over_k, cal.kz_over_k, vol.delay_s


def _expected_master(
    u: np.ndarray,
    tau: float,
    gamma: complex,
    ky: np.ndarray,
    kz: np.ndarray,
    t: np.ndarray,
    delta_f: float,
) -> np.ndarray:
    """Return the closed-form master cir [Q, Q, N] of convention (a)."""
    ay = _dirichlet(SPACING * (float(u[1]) - np.asarray(ky, dtype=np.float64)), COLS)
    az = _dirichlet(SPACING * (float(u[2]) - np.asarray(kz, dtype=np.float64)), ROWS)
    dd = _delay_kernel(delta_f * (np.asarray(t, dtype=np.float64) - float(tau)), NUM_BINS)
    return (
        (complex(gamma) / float(NUM_BINS))
        * az[:, None, None]
        * ay[None, :, None]
        * dd[None, None, :]
    )


def _expected_t04(
    u: np.ndarray,
    tau: float,
    gamma: complex,
    u_y: np.ndarray,
    u_z: np.ndarray,
    t: np.ndarray,
    centre: np.ndarray,
    delta_f: float,
) -> np.ndarray:
    """Return the closed-form T04 volume [Qy, Qz, Nt] of convention (b)."""
    ay = _dirichlet(SPACING * (float(u[1]) - np.asarray(u_y, dtype=np.float64)), COLS)
    az = _dirichlet(SPACING * (float(u[2]) - np.asarray(u_z, dtype=np.float64)), ROWS)
    dd = _delay_kernel(delta_f * (np.asarray(t, dtype=np.float64) - float(tau)), NUM_BINS)
    norm = float(np.sqrt(ROWS * COLS * NUM_BINS))
    field = (complex(gamma) / norm) * ay[:, None, None] * az[None, :, None] * dd[None, None, :]
    return field / np.asarray(centre, dtype=np.complex128)[:, :, None]


def _phase_offset(found_value: complex, expected_value: complex) -> float:
    """Return angle(found / expected) at one cell in radians."""
    found_c = complex(found_value)
    expected_c = complex(expected_value)
    if expected_c == 0j:
        if found_c == 0j:
            return 0.0
        return float(np.angle(found_c))
    return float(np.angle(found_c / expected_c))


def _peak_message(
    label: str,
    expected: tuple[float, float, float],
    found: tuple[float, float, float],
    bins: tuple[float, float, float],
    period_t: float,
    phase: float,
) -> str:
    """Build the peak-location message with expected/found and phase offset."""
    exp_y, exp_z, exp_t = (float(v) for v in expected)
    fnd_y, fnd_z, fnd_t = (float(v) for v in found)
    bin_y, bin_z, bin_t = (float(v) for v in bins)
    off_y = _circular_distance(fnd_y, exp_y, 2.0) / bin_y
    off_z = _circular_distance(fnd_z, exp_z, 2.0) / bin_z
    off_t = _circular_distance(fnd_t, exp_t, float(period_t)) / bin_t
    return (
        f"{label}: expected (u_y, u_z, t) = "
        f"({exp_y:.5f}, {exp_z:.5f}, {exp_t:.4e} s), found "
        f"({fnd_y:.5f}, {fnd_z:.5f}, {fnd_t:.4e} s), "
        f"off by ({off_y:.2f}, {off_z:.2f}, {off_t:.2f}) bins, "
        f"phase offset {float(phase):.1e} rad"
    )


def _peak_bins_master(period_t: float) -> tuple[float, float, float]:
    """Return the (angle, angle, delay) bin widths of the master chain."""
    return (2.0 / float(Q), 2.0 / float(Q), float(period_t) / float(NUM_BINS))


def _peak_bins_t04(period_t: float, num_out: int) -> tuple[float, float, float]:
    """Return the (angle, angle, delay) bin widths of a T04 volume."""
    return (2.0 / float(Q), 2.0 / float(Q), float(period_t) / float(num_out))


def _vs_gamma(distance: float, wavelength: float, wavenumber: float) -> complex:
    """Return the independent VS spreading factor gamma for one range."""
    return complex(wavelength / (4.0 * np.pi * distance) * np.exp(-1j * wavenumber * distance))


def _bv_gamma(r1: float, r2: float, wavelength: float, wavenumber: float) -> complex:
    """Return the independent BV spreading factor gamma for two ranges."""
    return complex(
        wavelength / ((4.0 * np.pi) ** 1.5 * r1 * r2) * np.exp(-1j * wavenumber * (r1 + r2))
    )


@dataclass(frozen=True)
class _VsCase:
    """One synthetic VS atom with its independent truth and capture."""

    geom: CaptureGeometry
    u: np.ndarray
    tau: float
    tau_mod: float
    gamma: complex
    hemi: int
    y: np.ndarray
    label: str


def _vs_case(azimuth_deg: float, elevation_deg: float, delay_bins: float, prefix: str) -> _VsCase:
    """Build one synthetic VS case exactly as the convention tests always did."""
    geom = _synthetic_geometry()
    rot = _synthetic_rotation()
    u = _local_unit(azimuth_deg, elevation_deg)
    tau = float(delay_bins) / BANDWIDTH
    distance = float(tau * SPEED_OF_LIGHT)
    source = UE_POS + distance * (rot @ u)
    hemi = 0 if float(u[0]) >= 0.0 else 1
    y = atom_cfr(source, [1.0], geom, "vs", pattern="iso")
    gamma = _vs_gamma(distance, float(geom.wavelength), float(geom.wavenumber))
    period = float(geom.delay_period)
    tau_mod = float(tau % period)
    label = f"{prefix} az{azimuth_deg:g}_el{elevation_deg:g}-t{delay_bins:g}"
    return _VsCase(
        geom=geom, u=u, tau=tau, tau_mod=tau_mod, gamma=gamma, hemi=hemi, y=y, label=label
    )


def _check_peak_within_half_bin(
    label: str,
    expected_triplet: tuple[float, float, float],
    found_triplet: tuple[float, float, float],
    bins: tuple[float, float, float],
    period_t: float,
    phase: float,
) -> None:
    """Assert the found peak lies within half a bin circularly in each axis."""
    msg = _peak_message(label, expected_triplet, found_triplet, bins, period_t, phase)
    for found_v, exp_v, width, period in (
        (found_triplet[0], expected_triplet[0], bins[0], 2.0),
        (found_triplet[1], expected_triplet[1], bins[1], 2.0),
        (found_triplet[2], expected_triplet[2], bins[2], period_t),
    ):
        off_bins = _circular_distance(float(found_v), float(exp_v), float(period)) / float(width)
        assert off_bins <= 0.5 + 1e-9, msg


def _check_relation_c(
    label: str,
    vol1: np.ndarray,
    vol4: np.ndarray,
    cir: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    u_y: np.ndarray,
    u_z: np.ndarray,
    centre: np.ndarray,
    t_native: np.ndarray,
) -> None:
    """Assert the documented T04-vs-master flip/phase relation of convention (c)."""
    np.testing.assert_allclose(ky, u_y, atol=1e-15, err_msg=f"{label}: ky != u_y exactly")
    np.testing.assert_allclose(kz[:-1], u_z[1:], atol=1e-15, err_msg=f"{label}: kz[:-1] != u_z[1:]")
    assert float(kz[-1]) == pytest.approx(1.0, abs=1e-15), f"{label}: kz[-1] != +1"
    assert float(u_z[0]) == pytest.approx(-1.0, abs=1e-15), f"{label}: u_z[0] != -1"
    centred = vol1 * np.asarray(centre, dtype=np.complex128)[:, :, None]
    scale = float(np.sqrt(NUM_BINS / (ROWS * COLS)))
    t_axis = np.asarray(t_native, dtype=np.float64)
    uy = np.asarray(u_y, dtype=np.float64)
    uz = np.asarray(u_z, dtype=np.float64)
    got_bulk = centred[:, 1:, :]
    ref_bulk = scale * np.transpose(cir[:-1, :, :], (1, 0, 2))
    denom = max(float(np.max(np.abs(got_bulk))), float(np.max(np.abs(ref_bulk))), 1e-300)
    abs_err = np.abs(got_bulk - ref_bulk)
    rel = float(np.max(abs_err) / denom)
    flat = int(np.argmax(abs_err))
    iy, j, n = np.unravel_index(flat, got_bulk.shape)
    iz = int(j) + 1
    phase = _phase_offset(complex(got_bulk[iy, j, n]), complex(ref_bulk[iy, j, n]))
    msg_bulk = (
        f"{label} relation-c bulk: max rel err {rel:.2e} at (iy, iz, n) = "
        f"({int(iy)}, {iz}, {int(n)}) (u_y, u_z, t) = "
        f"({float(uy[int(iy)]):.5f}, {float(uz[iz]):.5f}, {float(t_axis[int(n)]):.4e} s), "
        f"phase offset {float(phase):.2e} rad"
    )
    assert rel <= 1e-12, msg_bulk
    got_edge = centred[:, 0, :]
    ref_edge = ((-1.0) ** (ROWS - 1)) * scale * cir[Q - 1, :, :]
    denom_e = max(float(np.max(np.abs(got_edge))), float(np.max(np.abs(ref_edge))), 1e-300)
    abs_err_e = np.abs(got_edge - ref_edge)
    rel_e = float(np.max(abs_err_e) / denom_e)
    flat_e = int(np.argmax(abs_err_e))
    iy_e, n_e = np.unravel_index(flat_e, got_edge.shape)
    phase_e = _phase_offset(complex(got_edge[iy_e, n_e]), complex(ref_edge[iy_e, n_e]))
    msg_edge = (
        f"{label} relation-c edge: max rel err {rel_e:.2e} at (iy, iz, n) = "
        f"({int(iy_e)}, 0, {int(n_e)}) (u_y, u_z, t) = "
        f"({float(uy[int(iy_e)]):.5f}, {float(uz[0]):.5f}, "
        f"{float(t_axis[int(n_e)]):.4e} s), phase offset {float(phase_e):.2e} rad"
    )
    assert rel_e <= 1e-12, msg_edge
    native = vol4[:, :, ::DELAY_OS]
    denom_n = max(float(np.max(np.abs(native))), float(np.max(np.abs(vol1))), 1e-300)
    abs_err_n = np.abs(native - vol1)
    rel_n = float(np.max(abs_err_n) / denom_n)
    flat_n = int(np.argmax(abs_err_n))
    iy_n, iz_n, n_n = np.unravel_index(flat_n, native.shape)
    phase_n = _phase_offset(complex(native[iy_n, iz_n, n_n]), complex(vol1[iy_n, iz_n, n_n]))
    msg_native = (
        f"{label} relation-c native: max rel err {rel_n:.2e} at (iy, iz, n) = "
        f"({int(iy_n)}, {int(iz_n)}, {int(n_n)}) (u_y, u_z, t) = "
        f"({float(uy[int(iy_n)]):.5f}, {float(uz[int(iz_n)]):.5f}, "
        f"{float(t_axis[int(n_n)]):.4e} s), phase offset {float(phase_n):.2e} rad"
    )
    assert rel_n <= 1e-12, msg_native


def _check_master_chain(
    label: str,
    y_capture: np.ndarray,
    u: np.ndarray,
    tau: float,
    gamma: complex,
    geom: CaptureGeometry,
) -> None:
    """Assert the hemisphere split of one capture ``[2, R, C, N]`` and its master chain."""
    hemi = 0 if float(u[0]) >= 0.0 else 1
    assert np.all(y_capture[1 - hemi] == 0.0), f"{label}: wrong hemisphere not exactly 0"
    assert float(np.max(np.abs(y_capture[hemi]))) > 0.0, f"{label}: expected hemisphere is 0"
    aperture = y_capture[hemi]
    delta_f = float(geom.delta_f)
    period = float(geom.delay_period)
    cir, ky, kz, t = _master_chain(aperture, np.asarray(geom.freq_offsets))
    expected = _expected_master(u, tau, gamma, ky, kz, t, delta_f)
    denom = float(np.max(np.abs(expected)))
    rel = float(np.max(np.abs(cir - expected)) / denom)
    flat = int(np.argmax(np.abs(cir)))
    pi0, pj0, pn0 = np.unravel_index(flat, cir.shape)
    phase0 = _phase_offset(complex(cir[pi0, pj0, pn0]), complex(expected[pi0, pj0, pn0]))
    tau_mod = float(float(tau) % period)
    peak_msg = _peak_message(
        label,
        (float(u[1]), float(u[2]), tau_mod),
        (float(ky[pj0]), float(kz[pi0]), float(t[pn0])),
        _peak_bins_master(period),
        period,
        phase0,
    )
    assert rel <= 1e-10, f"{peak_msg}; master volume max rel err {rel:.2e} > 1e-10"
    _check_peak_within_half_bin(
        label,
        (float(u[1]), float(u[2]), tau_mod),
        (float(ky[pj0]), float(kz[pi0]), float(t[pn0])),
        _peak_bins_master(period),
        period,
        phase0,
    )
    power = np.abs(cir) ** 2
    dom_bin, dom_delay, _ = delay.dominant_delay(power, t)
    found_delay = float(dom_delay[pi0, pj0])
    bin_t = float(period) / float(NUM_BINS)
    delay_msg = (
        f"{label} dominant delay: expected {tau_mod:.4e} s "
        f"({tau_mod * BANDWIDTH:.4f} bins), found {found_delay:.4e} s "
        f"({found_delay * BANDWIDTH:.4f} bins); {peak_msg}"
    )
    assert _circular_distance(found_delay, tau_mod, period) <= 0.5 * bin_t + 1e-15, delay_msg
    assert int(dom_bin[pi0, pj0]) == int(pn0), delay_msg


def _triplet(values: np.ndarray) -> tuple[float, float, float]:
    """Return three floats as a typed tuple for the geometry helpers."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    return (float(flat[0]), float(flat[1]), float(flat[2]))


def _native_pixel(axis_y: np.ndarray, axis_z: np.ndarray, u: np.ndarray) -> tuple[int, int]:
    """Return the circularly nearest native (iy, iz) pixel to (u_y, u_z)."""
    iy = _nearest_index(np.asarray(axis_y), float(u[1]), 2.0)
    iz = _nearest_index(np.asarray(axis_z), float(u[2]), 2.0)
    return iy, iz


@pytest.mark.parametrize("azimuth_deg,elevation_deg,delay_bins", CASES, ids=CASE_IDS)
def test_vs_atom_through_master_chain(
    azimuth_deg: float, elevation_deg: float, delay_bins: float
) -> None:
    """Check one VS atom through the master chain: (a), peaks and dominant delay."""
    case = _vs_case(azimuth_deg, elevation_deg, delay_bins, "master chain")
    _check_master_chain(case.label, case.y[0, 0], case.u, case.tau, case.gamma, case.geom)


@pytest.mark.parametrize("azimuth_deg,elevation_deg,range_m,wraps", BV_CASES, ids=BV_IDS)
def test_bv_atom_through_master_chain(
    azimuth_deg: float, elevation_deg: float, range_m: float, wraps: int
) -> None:
    """Check three BV atoms through the master chain with independent truth."""
    geom = _synthetic_geometry()
    rot = _synthetic_rotation()
    period = float(geom.delay_period)
    wavelength = float(geom.wavelength)
    wavenumber = float(geom.wavenumber)
    u = _local_unit(azimuth_deg, elevation_deg)
    point = UE_POS + float(range_m) * (rot @ u)
    bs = np.asarray(geom.bs_pos[0], dtype=np.float64)
    r1 = float(np.linalg.norm(point - bs))
    r2 = float(range_m)
    tau = float((r1 + r2) / SPEED_OF_LIGHT)
    label = f"bv master chain az{azimuth_deg:g}_el{elevation_deg:g}-r{range_m:g}"
    wraps_found = int(tau // period)
    assert wraps_found == int(wraps), (
        f"{label}: expected {int(wraps)} wraps but tau={tau:.4e} s, "
        f"period={period:.4e} s gives {wraps_found} (tau/T={tau / period:.3f})"
    )
    gamma = _bv_gamma(r1, r2, wavelength, wavenumber)
    y = atom_cfr(point, [1.0], geom, "bv", pattern="iso")
    _check_master_chain(label, y[0, 0], u, tau, gamma, geom)


@pytest.mark.parametrize("azimuth_deg,elevation_deg,delay_bins", CASES, ids=CASE_IDS)
def test_t04_volume_matches_analytic_and_master(
    azimuth_deg: float, elevation_deg: float, delay_bins: float
) -> None:
    """Check the T04 volume against (b), its peaks (d) and the master link (c)."""
    case = _vs_case(azimuth_deg, elevation_deg, delay_bins, "t04")
    geom = case.geom
    u = case.u
    tau = case.tau
    tau_mod = case.tau_mod
    gamma = case.gamma
    y = case.y
    label = case.label
    delta_f = float(geom.delta_f)
    period = float(geom.delay_period)
    u_y, u_z, t4 = observables.volume_axes(Q, Q, DELAY_OS * NUM_BINS, delta_f=delta_f)
    centre = observables.aperture_centre_phase(u_y, u_z, rows=ROWS, cols=COLS)
    vol4 = observables.angle_delay_volume(y, oversample=(ANGLE_OS, DELAY_OS))[0, 0, case.hemi]
    expected4 = _expected_t04(u, tau, gamma, u_y, u_z, t4, centre, delta_f)
    denom4 = float(np.max(np.abs(expected4)))
    rel4 = float(np.max(np.abs(vol4 - expected4)) / denom4)
    flat4 = int(np.argmax(np.abs(vol4)))
    iy4, iz4, it4 = np.unravel_index(flat4, vol4.shape)
    phase4 = _phase_offset(complex(vol4[iy4, iz4, it4]), complex(expected4[iy4, iz4, it4]))
    peak4_msg = _peak_message(
        label,
        (float(u[1]), float(u[2]), tau_mod),
        (float(u_y[iy4]), float(u_z[iz4]), float(t4[it4])),
        _peak_bins_t04(period, DELAY_OS * NUM_BINS),
        period,
        phase4,
    )
    assert rel4 <= 1e-10, f"{peak4_msg}; t04 volume max rel err {rel4:.2e} > 1e-10"
    _check_peak_within_half_bin(
        label,
        (float(u[1]), float(u[2]), tau_mod),
        (float(u_y[iy4]), float(u_z[iz4]), float(t4[it4])),
        _peak_bins_t04(period, DELAY_OS * NUM_BINS),
        period,
        phase4,
    )
    cir, ky, kz, _ = _master_chain(y[0, 0, case.hemi], np.asarray(geom.freq_offsets))
    vol1 = observables.angle_delay_volume(y, oversample=(ANGLE_OS, 1))[0, 0, case.hemi]
    _, _, t_native = observables.volume_axes(Q, Q, NUM_BINS, delta_f=delta_f)
    _check_relation_c(label, vol1, vol4, cir, ky, kz, u_y, u_z, centre, t_native)


@pytest.mark.parametrize("azimuth_deg,elevation_deg,delay_bins", CASES, ids=CASE_IDS)
def test_delay_returns_match_analytic_delay(
    azimuth_deg: float, elevation_deg: float, delay_bins: float
) -> None:
    """Check the D and D-o delay returns against the analytic wrapped delay."""
    case = _vs_case(azimuth_deg, elevation_deg, delay_bins, "delay")
    u = case.u
    tau_mod = case.tau_mod
    y = case.y
    label = case.label
    delta_f = float(case.geom.delta_f)
    period = float(case.geom.delay_period)
    axis_y, axis_z, _ = observables.volume_axes(COLS, ROWS, NUM_BINS, delta_f=delta_f)
    iy, iz = _native_pixel(axis_y, axis_z, u)
    d_data = observables.extract(y, "D", {"delta_f": delta_f}).data[0, 0]
    found = float(d_data[case.hemi, iy, iz, 0])
    tol = 0.01 / BANDWIDTH
    msg = (
        f"{label} D return: expected {tau_mod:.4e} s ({tau_mod * BANDWIDTH:.4f} bins), "
        f"found {found:.4e} s ({found * BANDWIDTH:.4f} bins), "
        f"analytic (u_y, u_z) = ({float(u[1]):.5f}, {float(u[2]):.5f}), "
        f"native pixel (iy, iz) = ({iy}, {iz}) "
        f"(u_y, u_z) = ({float(axis_y[iy]):.5f}, {float(axis_z[iz]):.5f})"
    )
    assert _circular_distance(found, tau_mod, period) <= tol + 1e-18, msg
    assert np.all(np.isnan(d_data[1 - case.hemi])), f"{label} empty hemisphere has returns; {msg}"
    omni = observables.extract(y, "D-o", {"delta_f": delta_f}).data[0, 0, 0]
    found_o = float(omni)
    msg_o = (
        f"{label} D-o return: expected {tau_mod:.4e} s ({tau_mod * BANDWIDTH:.4f} bins), "
        f"found {found_o:.4e} s ({found_o * BANDWIDTH:.4f} bins), "
        f"analytic (u_y, u_z) = ({float(u[1]):.5f}, {float(u[2]):.5f}), "
        f"native pixel (iy, iz) = ({iy}, {iz}) "
        f"(u_y, u_z) = ({float(axis_y[iy]):.5f}, {float(axis_z[iz]):.5f})"
    )
    assert _circular_distance(found_o, tau_mod, period) <= tol + 1e-18, msg_o


@pytest.mark.parametrize("view", [0, 1], ids=["view0", "view1"])
def test_sionna_fixture_through_master_chain(view: int) -> None:
    """Check real Sionna LoS captures through the master chain, T04 and returns."""
    fixture = np.load(FIXTURE)
    freqs = np.asarray(fixture["freq_offsets"])
    delta_f = float(imaging.uniform_frequency_spacing(freqs))
    period = float(1.0 / delta_f)
    ue_pos = np.asarray(fixture["ue_pos"], dtype=np.float64)
    ue_orientation = np.asarray(fixture["ue_orientation"], dtype=np.float64)
    bs_pos = np.asarray(fixture["bs_pos"], dtype=np.float64)
    bs_look_at = np.asarray(fixture["bs_look_at"], dtype=np.float64)
    f_c = float(fixture["f_c"])
    path_tau = np.asarray(fixture["path_tau"], dtype=np.float64)
    bs_in_front = np.asarray(fixture["bs_in_front"])
    label = f"sionna view{view}"
    u = np.asarray(
        calibration.geometric_los_source_direction_local(
            tx_position=_triplet(bs_pos[0]),
            ue_position=_triplet(ue_pos[view]),
            ue_orientation=_triplet(ue_orientation[view]),
        ),
        dtype=np.float64,
    )
    tau = float(
        delay.geometric_los_delay_s(
            _triplet(bs_pos[0]),
            _triplet(ue_pos[view]),
        )
    )
    hemi = 0 if float(u[0]) >= 0.0 else 1
    assert hemi == (0 if bool(bs_in_front[view]) else 1), f"{label}: hemisphere mismatch"
    y_s = np.asarray(fixture["aperture_cfr"][view], dtype=np.complex128)
    assert np.all(y_s[1 - hemi] == 0.0), f"{label}: empty Sionna hemisphere not exactly 0"
    assert tau > period, f"{label}: fixture path should wrap but tau={tau:.4e} s"
    assert abs(float(tau) - float(path_tau[view])) < 1e-12, (
        f"{label}: geometric tau {tau:.6e} s != path_tau {float(path_tau[view]):.6e} s"
    )
    tau_mod = float(tau % period)
    geom = CaptureGeometry(
        np.asarray(ue_pos),
        rotations_from_orientations(np.asarray(ue_orientation)),
        np.asarray(bs_pos),
        planar_element_offsets(SPEED_OF_LIGHT / float(f_c)),
        freqs.astype(np.float64),
        float(f_c),
        bs_rot=bs_orientation(np.asarray(bs_pos)[0], np.asarray(bs_look_at))[None],
    )
    y_m = atom_cfr(
        np.asarray(bs_pos),
        [1.0],
        geom,
        "vs",
        pattern="tr38901",
        polarization="vv",
    )[view, 0]
    cir_m, _, _, _ = _master_chain(y_m[hemi], freqs)
    vol4_m = observables.angle_delay_volume(y_m[None, None], oversample=(ANGLE_OS, DELAY_OS))[
        0, 0, hemi
    ]
    cir_s, ky, kz, t = _master_chain(y_s[hemi], freqs)
    flat_s = int(np.argmax(np.abs(cir_s)))
    pi_s, pj_s, pn_s = np.unravel_index(flat_s, cir_s.shape)
    phase_sm = _phase_offset(complex(cir_s[pi_s, pj_s, pn_s]), complex(cir_m[pi_s, pj_s, pn_s]))
    peak_s_msg = _peak_message(
        f"{label} master",
        (float(u[1]), float(u[2]), tau_mod),
        (float(ky[pj_s]), float(kz[pi_s]), float(t[pn_s])),
        _peak_bins_master(period),
        period,
        phase_sm,
    )
    _check_peak_within_half_bin(
        f"{label} master",
        (float(u[1]), float(u[2]), tau_mod),
        (float(ky[pj_s]), float(kz[pi_s]), float(t[pn_s])),
        _peak_bins_master(period),
        period,
        phase_sm,
    )
    y_6d = y_s[None, None, :, :, :, :]
    u_y, u_z, t4 = observables.volume_axes(Q, Q, DELAY_OS * NUM_BINS, delta_f=delta_f)
    centre = observables.aperture_centre_phase(u_y, u_z, rows=ROWS, cols=COLS)
    vol4 = observables.angle_delay_volume(y_6d, oversample=(ANGLE_OS, DELAY_OS))[0, 0, hemi]
    flat4 = int(np.argmax(np.abs(vol4)))
    iy4, iz4, it4 = np.unravel_index(flat4, vol4.shape)
    phase4 = _phase_offset(complex(vol4[iy4, iz4, it4]), complex(vol4_m[iy4, iz4, it4]))
    _check_peak_within_half_bin(
        f"{label} t04",
        (float(u[1]), float(u[2]), tau_mod),
        (float(u_y[iy4]), float(u_z[iz4]), float(t4[it4])),
        _peak_bins_t04(period, DELAY_OS * NUM_BINS),
        period,
        phase4,
    )
    vol1 = observables.angle_delay_volume(y_6d, oversample=(ANGLE_OS, 1))[0, 0, hemi]
    _, _, t_native = observables.volume_axes(Q, Q, NUM_BINS, delta_f=delta_f)
    _check_relation_c(label, vol1, vol4, cir_s, ky, kz, u_y, u_z, centre, t_native)
    axis_y, axis_z, _ = observables.volume_axes(COLS, ROWS, NUM_BINS, delta_f=delta_f)
    iy, iz = _native_pixel(axis_y, axis_z, u)
    d_data = observables.extract(y_6d, "D", {"delta_f": delta_f}).data[0, 0]
    found = float(d_data[hemi, iy, iz, 0])
    tol = 0.01 / BANDWIDTH
    msg = (
        f"{label} D return: expected {tau_mod:.4e} s ({tau_mod * BANDWIDTH:.4f} bins), "
        f"found {found:.4e} s ({found * BANDWIDTH:.4f} bins); {peak_s_msg}"
    )
    assert _circular_distance(found, tau_mod, period) <= tol + 1e-18, msg
    assert np.all(np.isnan(d_data[1 - hemi])), f"{label} empty hemisphere has returns"
    found_o = float(observables.extract(y_6d, "D-o", {"delta_f": delta_f}).data[0, 0, 0])
    msg_o = (
        f"{label} D-o return: expected {tau_mod:.4e} s ({tau_mod * BANDWIDTH:.4f} bins), "
        f"found {found_o:.4e} s; {peak_s_msg}"
    )
    assert _circular_distance(found_o, tau_mod, period) <= tol + 1e-18, msg_o
    denom = float(np.linalg.norm(cir_s))
    rel = float(np.linalg.norm(cir_m - cir_s) / denom)
    beta = complex(np.vdot(cir_m, cir_s) / np.vdot(cir_m, cir_m))
    rel_fit = float(np.linalg.norm(beta * cir_m - cir_s) / denom)
    phase_model = _phase_offset(complex(cir_s[pi_s, pj_s, pn_s]), complex(cir_m[pi_s, pj_s, pn_s]))
    model_msg = (
        f"{label} model vs sionna: expected (u_y, u_z, t) = "
        f"({float(u[1]):.5f}, {float(u[2]):.5f}, {tau_mod:.4e} s), found "
        f"({float(ky[pj_s]):.5f}, {float(kz[pi_s]):.5f}, {float(t[pn_s]):.4e} s), "
        f"phase offset {phase_model:.1e} rad, rel {rel:.2e}, rel-fit {rel_fit:.2e}"
    )
    assert rel <= 1e-3, model_msg
    assert rel_fit <= 1e-4, model_msg
    assert abs(phase_model) <= 1e-3, model_msg


def test_reference_discriminates_mirrored_conventions() -> None:
    """Show the analytic reference rejects mirrored angle/delay conventions."""
    case = _vs_case(35.0, 20.0, 5.4219, "mirror control")
    geom = case.geom
    u = case.u
    tau = case.tau
    gamma = case.gamma
    hemi = case.hemi
    y = case.y
    label = case.label
    delta_f = float(geom.delta_f)
    period = float(geom.delay_period)
    cir, ky, kz, t = _master_chain(y[0, 0, hemi], np.asarray(geom.freq_offsets))
    correct = _expected_master(u, tau, gamma, ky, kz, t, delta_f)
    denom = float(np.max(np.abs(correct)))
    rel_ok = float(np.max(np.abs(cir - correct)) / denom)
    flat = int(np.argmax(np.abs(cir)))
    pi0, pj0, pn0 = np.unravel_index(flat, cir.shape)
    phase_ok = _phase_offset(complex(cir[pi0, pj0, pn0]), complex(correct[pi0, pj0, pn0]))
    context = _peak_message(
        label,
        (float(u[1]), float(u[2]), float(tau % period)),
        (float(ky[pj0]), float(kz[pi0]), float(t[pn0])),
        _peak_bins_master(period),
        period,
        phase_ok,
    )
    assert rel_ok <= 1e-10, f"{context}; correct reference rel err {rel_ok:.2e}"
    u_flip_y = np.array([u[0], -u[1], u[2]])
    u_flip_z = np.array([u[0], u[1], -u[2]])
    u_swap = np.array([u[0], u[2], u[1]])
    wrongs = {
        "u_y -> -u_y": _expected_master(u_flip_y, tau, gamma, ky, kz, t, delta_f),
        "u_z -> -u_z": _expected_master(u_flip_z, tau, gamma, ky, kz, t, delta_f),
        "tau -> -tau": _expected_master(u, -tau, gamma, ky, kz, t, delta_f),
        "swap u_y/u_z": _expected_master(u_swap, tau, gamma, ky, kz, t, delta_f),
        "carrier conj": _expected_master(u, tau, np.conj(gamma), ky, kz, t, delta_f),
    }
    for name, wrong in wrongs.items():
        rel = float(np.max(np.abs(cir - wrong)) / float(np.max(np.abs(wrong))))
        msg = f"mirror control {name}: rel err {rel:.2e} <= 0.5; {context}"
        assert rel > 0.5, msg
