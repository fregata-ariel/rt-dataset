"""Observables for the tomography baselines (docs/tomography_baselines.md §2.3).

Raw data ``Y[v, b, h, r, col, n]`` is complex128: view, base station, hemisphere
(0 = front ``u_x >= 0``, 1 = back), aperture row (row 0 is the top, +z), column and
frequency bin. ``n0 = N // 2`` is the DC bin and ``df_n = (n - N // 2) * delta_f``.

Angle-delay volume
------------------
For oversampling ``(a, d_t)`` the volume ``vol[v, b, h, iy, iz, it]`` has shape
``[V, B, H, Qy, Qz, Nt]`` with ``Qy = a*C``, ``Qz = a*R``, ``Nt = d_t*N`` and::

    vol = 1/sqrt(R*C*N) * sum_{r,col,n} w_r w_c w_n Y
          * exp(-j 2 pi col fy[iy]) * exp(+j 2 pi r fz[iz])
          * exp(+j 2 pi (n - N//2) it / Nt)

with ``fy = fftshift(fftfreq(Qy))`` and ``fz = fftshift(fftfreq(Qz))``. The axis
order is ``(u_y, u_z, t)`` (columns first); both angle axes increase toward local
+y and +z. The phase reference is element ``(r=0, col=0)`` (the index origin), not
the aperture centre: the physically centred field of the design is
``c = vol * aperture_centre_phase(u_y, u_z, ...)``. The scale is always
1/sqrt(R*C*N); unwindowed with ``a = d_t = 1`` the map is unitary, so white
``CN(0, sigma^2)`` raw noise stays white. Windowed/oversampled cells carry power
``sigma^2 * volume_noise_gain(R, C, N, window)``.

Relation to the master pipeline (``rf_camera``): with
``raw = imaging.aperture_to_angular_fft(Y[v, b, h], fft_rows=Qz, fft_cols=Qy)``
(unwindowed, ``d_t = 1``), ``vol[..., iy, iz, :]`` is the delay IFFT of
``raw[(Qz - iz) % Qz, iy, :]`` for even ``Qz``, and
``calibration.direction_cosine_axes(fft_rows=Qz, fft_cols=Qy, ...)`` returns
``ky == u_y`` and ``kz[i] == u_z[i + 1]``. With ``cal =
calibrate_angular_cfr(raw, ...)`` and ``cir = delay.angular_cfr_to_delay(cal.cfr,
freqs).cir`` (shape ``[kz, ky, N]``), ``c[iy, iz, :] == sqrt(N / (R*C)) *
cir[iz - 1, iy, :]`` for ``iz = 1..Qz-1``.

Node table
----------
``U = angle_delay_volume(Y)`` (unitary, unwindowed, native grid ``[..., C, R, N]``)
and ``beam(X)`` the unitary angle-only transform of a ``[..., R, C]`` slice
(``angle_delay_volume(X[..., None])[..., 0]``, shape ``[..., C, R]``). A PHAT mask
keeps a sample iff ``|Y| > 0`` and ``|Y| >= mask_k * sqrt(noise_var)``; invalid
samples carry data 0 and mask False. Canonicalising over axes A multiplies each
slice by ``conj(ref) / |ref|`` with ``ref`` the largest-|Y| sample over A (first in
C order), removing one unit-modulus unknown per slice while leaving the mask alone.

===========  ==============================================================  ===========
name         data (dtype, shape)                                              noise_dof
===========  ==============================================================  ===========
I            ``sum_t |U|^2`` float [V,B,H,C,R]                                2N
I_n0         ``|beam(Y[..., n0])|^2`` float [V,B,H,C,R]                       2
D            CFAR-of-``|angle_delay_volume(Y,"taylor",(1,d_t))|^2``,
             position*T/Nt seconds, float [V,B,H,C,R,K], NaN absent           -
D_PHAT       as D, on the volume of the masked unit-modulus DP data            -
T            ``|U|^2 / sum_t|U|^2`` float [V,B,H,C,R,N], 0 where invalid       -
P            ``Y[..., n0] / |Y[..., n0]|`` complex [V,B,H,R,C]                 -
P_W          unit-modulus of the IP_W data, complex [V,B,H,R,C,N]              -
ID           ``|U|^2`` float [V,B,H,C,R,N]                                    2
IP           ``Y[..., n0]`` complex [V,B,H,R,C]                               -
IP_W         Y canonicalised over (h,r,col), complex [V,B,H,R,C,N]             -
DP           ``Y / |Y|`` complex [V,B,H,R,C,N]                                 -
IDP          copy of Y, complex [V,B,H,R,C,N]                                  -
IDP-o        Y canonicalised over n, complex [V,B,H,R,C,N]                     -
DP-o         unit-modulus of the IDP-o data, complex [V,B,H,R,C,N]             -
IP-o         ``|Y|`` float [V,B,H,R,C,N]                                       -
P-o          empty float [V,B,H,0]                                             -
ID-o         ``sum_{r,col} |E|^2`` float [V,B,H,N], E the unitary per-element
             delay transform                                                  2M
I-o          ``sum_{r,col,n} |Y|^2`` float [V,B,H] (RSS)                      2MN
D-o          CFAR-of-``sum_{h,r,col}|E_w|^2`` seconds, float [V,B,K]           -
IDP-1el      ``Y[..., r_e, c_e, :]`` complex [V,B,H,N]                        -
DP-1el       unit-modulus of the IDP-1el data, complex [V,B,H,N]               -
IPxK         ``Y[..., bins]`` complex [V,B,H,R,C,K]                            -
PxK          unit-modulus of the IPxK data, complex [V,B,H,R,C,K]              -
===========  ==============================================================  ===========

Semantics
---------
``Observable.noise_var`` is sigma^2 of the raw complex AWGN per sample of ``Y``
(the one absolute dataset value, ``params["noise_var"]``) or None; it is the same
for every node, and the noise-only mean of a power node is
``noise_dof / 2 * sigma^2``. ``mask`` is True for valid samples; invalid PHAT
samples carry data 0 and absent D returns are NaN. IP_W/P_W remove one phase per
``(v, b, n)`` shared by both hemispheres (the capture gauge applies to both), so
inter-hemisphere and inter-element phases within a bin are kept. The omni column
removes one phase per ``(v, b, h, r, col)``: each hemisphere channel of an element
is treated as its own antenna, which is what makes P-o empty and IP-o = ``|Y|`` as
in §2.3. D-type nodes use the order-statistic CFAR (per-profile median / ``ln 2``),
so they need no ``sigma^2`` and are exactly shift-covariant; returns may include
Taylor sidelobes of very strong paths (-35 dB in delay, about -32 dB in angle for
8 elements). The unitary unwindowed volume is the likelihood domain; the
Taylor-windowed, 8x oversampled volume is for detection, E1 and the noise estimate.

Meta always holds ``node`` and ``domain``; beam nodes add ``u_y``/``u_z``, D-type
nodes ``t_period``/``pfa``/``max_returns``/``delay_oversample``, power nodes the
``noise_dof`` above, partial-D nodes ``bins``, 1el nodes ``element`` and the DC
nodes ``n0``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT
from scipy.signal.windows import taylor

TAYLOR_NBAR: int = 4
TAYLOR_SLL_DB: float = 35.0
DEFAULT_PFA: float = 1e-4
DEFAULT_MAX_RETURNS: int = 3
DEFAULT_MASK_K: float = 1.0
DEFAULT_DELAY_OVERSAMPLE: int = 8
PARTIAL_D_NUM_BINS: int = 4
ONE_ELEMENT: tuple[int, int] = (3, 3)
NODE_NAMES: tuple[str, ...] = (
    "I",
    "I_n0",
    "D",
    "D_PHAT",
    "T",
    "P",
    "P_W",
    "ID",
    "IP",
    "IP_W",
    "DP",
    "IDP",
    "I-o",
    "D-o",
    "ID-o",
    "IP-o",
    "P-o",
    "DP-o",
    "IDP-o",
    "IDP-1el",
    "DP-1el",
    "PxK",
    "IPxK",
)
NODE_ALIASES: dict[str, str] = {"I@n0": "I_n0", "P×K": "PxK", "IP×K": "IPxK"}

_ALLOWED_PARAMS = frozenset(
    {
        "noise_var",
        "mask_k",
        "delta_f",
        "pfa",
        "max_returns",
        "delay_oversample",
        "bins",
        "element",
        "spacing_lambda",
    }
)
_PHAT_NODES = frozenset({"P", "P_W", "DP", "DP-o", "DP-1el", "PxK", "D_PHAT"})
_DELAY_NODES = frozenset({"D", "D_PHAT", "D-o"})
_BEAM_NODES = frozenset({"I", "I_n0", "D", "D_PHAT", "T", "ID"})
_DOMAINS: dict[str, str] = {
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
    "I-o": "scalar",
    "D-o": "returns",
    "ID-o": "delay",
    "IP-o": "element_freq",
    "P-o": "empty",
    "DP-o": "element_freq",
    "IDP-o": "element_freq",
    "IDP-1el": "freq",
    "DP-1el": "freq",
    "PxK": "element_freq",
    "IPxK": "element_freq",
}


@dataclass(frozen=True)
class Observable:
    """A derived observable, its validity mask, raw noise variance and metadata."""

    data: np.ndarray
    mask: np.ndarray
    noise_var: float | None
    meta: dict[str, Any]


@dataclass(frozen=True)
class CfarReturns:
    """Up to ``K`` sub-sample CFAR returns per profile along the last axis."""

    position: np.ndarray
    power: np.ndarray
    mask: np.ndarray
    noise_power: np.ndarray


def taylor_window(length: int) -> np.ndarray:
    """Return the fixed -35 dB, nbar=4 Taylor window of ``length`` samples."""
    length = int(length)
    if length < 1:
        raise ValueError("length must be >= 1")
    window = taylor(length, nbar=TAYLOR_NBAR, sll=TAYLOR_SLL_DB, norm=True, sym=True)
    return np.asarray(window, dtype=np.float64)


def volume_noise_gain(rows: int, cols: int, num_bins: int, window: str | None) -> float:
    """Return the per-cell noise power gain of the windowed volume transform."""
    if window is None:
        return 1.0
    if window != "taylor":
        raise ValueError(f"unknown window {window!r}; expected None or 'taylor'")
    w_r, w_c, w_n = _window_vectors(rows, cols, num_bins, window)
    return float(np.mean(w_r**2) * np.mean(w_c**2) * np.mean(w_n**2))


def volume_axes(
    qy: int,
    qz: int,
    nt: int,
    *,
    delta_f: float,
    spacing_lambda: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the volume axes ``(u_y[qy], u_z[qz], t[nt])``.

    ``u_y = fftshift(fftfreq(qy)) / s``, ``u_z`` likewise, and
    ``t = arange(nt) / (nt * delta_f)``.
    """
    qy, qz, nt = int(qy), int(qz), int(nt)
    if qy < 1 or qz < 1 or nt < 1:
        raise ValueError("qy, qz and nt must be >= 1")
    if not np.isfinite(delta_f) or delta_f <= 0.0:
        raise ValueError("delta_f must be finite and > 0")
    if not np.isfinite(spacing_lambda) or spacing_lambda <= 0.0:
        raise ValueError("spacing_lambda must be finite and > 0")
    u_y = np.fft.fftshift(np.fft.fftfreq(qy)) / spacing_lambda
    u_z = np.fft.fftshift(np.fft.fftfreq(qz)) / spacing_lambda
    t = np.arange(nt, dtype=np.float64) / (nt * delta_f)
    return u_y, u_z, t


def aperture_centre_phase(
    u_y: np.ndarray,
    u_z: np.ndarray,
    *,
    rows: int,
    cols: int,
    spacing_lambda: float = 0.5,
) -> np.ndarray:
    """Return the phase taking the index-origin volume to the centred field."""
    u_y = np.asarray(u_y, dtype=np.float64)
    u_z = np.asarray(u_z, dtype=np.float64)
    if u_y.ndim != 1 or u_z.ndim != 1:
        raise ValueError("u_y and u_z must be one-dimensional")
    rows, cols = int(rows), int(cols)
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    if not np.isfinite(spacing_lambda) or spacing_lambda <= 0.0:
        raise ValueError("spacing_lambda must be finite and > 0")
    exponent = spacing_lambda * (-(cols - 1) / 2.0 * u_y[:, None] + (rows - 1) / 2.0 * u_z[None, :])
    return np.exp(-2j * np.pi * exponent)


def angle_delay_volume(
    Y: np.ndarray,
    window: str | None = None,
    oversample: tuple[int, int] = (1, 1),
) -> np.ndarray:
    """Return the complex angle-delay volume ``[V, B, H, a*C, a*R, d_t*N]``.

    ``oversample = (a, d_t)`` are integer angle and delay factors. The phase
    reference is the index origin ``(r=0, col=0)``; multiply by
    :func:`aperture_centre_phase` to obtain the physically centred field. The
    exact relation to :func:`imaging.aperture_to_angular_fft` and
    :func:`calibration.direction_cosine_axes` is spelled out in the module
    docstring.
    """
    arr = _validate_y(Y)
    a, d_t = _validate_oversample(oversample)
    w_r, w_c, w_n = _window_vectors(arr.shape[3], arr.shape[4], arr.shape[5], window)
    num_rows, num_cols, num_bins = arr.shape[3], arr.shape[4], arr.shape[5]
    qy, qz, nt = a * num_cols, a * num_rows, d_t * num_bins

    spec = arr * w_r[:, None, None] * w_c[None, :, None] * w_n
    spec = np.fft.fftshift(np.fft.fft(spec, n=qy, axis=-2), axes=-2)
    spec = np.fft.fftshift(qz * np.fft.ifft(spec, n=qz, axis=-3), axes=-3)
    spec = nt * np.fft.ifft(spec, n=nt, axis=-1)
    spec = spec * np.exp(-2j * np.pi * (num_bins // 2) * np.arange(nt) / nt)
    volume = np.swapaxes(spec, -3, -2)
    return (volume / np.sqrt(num_rows * num_cols * num_bins)).astype(np.complex128, copy=False)


def cfar_returns(
    power_volume: np.ndarray,
    pfa: float = DEFAULT_PFA,
    max_returns: int = DEFAULT_MAX_RETURNS,
    *,
    noise_power: float | np.ndarray | None = None,
) -> CfarReturns:
    """Return sub-sample CFAR returns along the circular last axis.

    The noise level is the order-statistic estimate ``median / ln 2`` unless
    ``noise_power`` is given, the threshold is ``-ln(pfa) * noise`` and candidates
    are circular local maxima above it. Positions are fractional sample indices in
    ``[0, Nt)``; absent slots are NaN with ``mask`` False.
    """
    power = np.asarray(power_volume, dtype=np.float64)
    if power.ndim < 1 or power.shape[-1] < 3:
        raise ValueError("power_volume must have shape [..., Nt] with Nt >= 3")
    if np.any(~np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("power_volume must be finite and >= 0")
    if not 0.0 < float(pfa) < 1.0:
        raise ValueError("pfa must lie strictly between 0 and 1")
    max_returns = int(max_returns)
    if max_returns < 1:
        raise ValueError("max_returns must be >= 1")
    nt = power.shape[-1]

    if noise_power is None:
        noise = np.median(power, axis=-1) / np.log(2.0)
    else:
        noise = np.asarray(noise_power, dtype=np.float64)
        if np.any(~np.isfinite(noise)) or np.any(noise < 0.0):
            raise ValueError("noise_power must be finite and >= 0")
        noise = np.broadcast_to(noise, power.shape[:-1]).astype(np.float64)
    threshold = -np.log(float(pfa)) * noise

    previous = np.roll(power, 1, axis=-1)
    following = np.roll(power, -1, axis=-1)
    is_peak = (power > previous) & (power >= following) & (power > threshold[..., None])

    denominator = previous - 2.0 * power + following
    delta = np.zeros_like(power)
    nonzero = denominator != 0.0
    delta[nonzero] = 0.5 * (previous[nonzero] - following[nonzero]) / denominator[nonzero]
    np.clip(delta, -0.5, 0.5, out=delta)

    sample = np.arange(nt, dtype=np.float64)
    position = np.mod(sample + delta, float(nt))
    vertex = power - 0.25 * (previous - following) * delta
    score = np.where(is_peak, vertex, -np.inf)

    keep = min(max_returns, nt)
    order = np.argsort(-score, axis=-1, kind="stable")[..., :keep]
    positions = np.take_along_axis(position, order, axis=-1)
    powers = np.take_along_axis(vertex, order, axis=-1)
    found = np.take_along_axis(is_peak, order, axis=-1)
    positions = np.where(found, positions, np.nan)
    powers = np.where(found, powers, np.nan)

    if keep < max_returns:
        pad = max_returns - keep
        positions = np.concatenate(
            [positions, np.full(positions.shape[:-1] + (pad,), np.nan)], axis=-1
        )
        powers = np.concatenate([powers, np.full(powers.shape[:-1] + (pad,), np.nan)], axis=-1)
        found = np.concatenate([found, np.zeros(found.shape[:-1] + (pad,), dtype=bool)], axis=-1)

    return CfarReturns(
        position=positions.astype(np.float64),
        power=powers.astype(np.float64),
        mask=found.astype(bool),
        noise_power=noise.astype(np.float64),
    )


def noise_var_estimate(
    Y: np.ndarray,
    max_path_m: float,
    *,
    delta_f: float,
    spacing_lambda: float = 0.5,
    oversample: tuple[int, int] = (8, 8),
) -> float:
    """Estimate the raw per-sample AWGN variance ``sigma^2`` from a capture set.

    Uses Taylor-windowed, oversampled volumes and only cells that are both
    evanescent in angle (``u_y**2 + u_z**2 > 1``) and beyond ``max_path_m`` in
    delay (``t > max_path_m / c``). The pooled ``|vol|**2`` is
    ``median / ln 2 / volume_noise_gain``. Raises ValueError when no cell qualifies.
    """
    arr = _validate_y(Y)
    a, d_t = _validate_oversample(oversample)
    max_path_m = float(max_path_m)
    if not np.isfinite(max_path_m) or max_path_m < 0.0:
        raise ValueError("max_path_m must be finite and >= 0")
    num_views, num_bs, _, num_rows, num_cols, num_bins = arr.shape
    qy, qz, nt = a * num_cols, a * num_rows, d_t * num_bins
    u_y, u_z, t = volume_axes(qy, qz, nt, delta_f=delta_f, spacing_lambda=spacing_lambda)
    angle_mask = u_y[:, None] ** 2 + u_z[None, :] ** 2 > 1.0
    delay_mask = t > max_path_m / SPEED_OF_LIGHT
    cell_mask = angle_mask[:, :, None] & delay_mask[None, None, :]
    if not np.any(cell_mask):
        raise ValueError("no evanescent and beyond-range delay cell qualifies")

    gain = volume_noise_gain(num_rows, num_cols, num_bins, "taylor")
    chunks = []
    for view in range(num_views):
        for bs in range(num_bs):
            volume = angle_delay_volume(
                arr[view, bs][None, None], window="taylor", oversample=(a, d_t)
            )[0, 0]
            chunks.append((np.abs(volume) ** 2)[:, cell_mask])
    pooled = np.concatenate(chunks, axis=0)
    if pooled.size == 0:
        raise ValueError("no cell selected for the noise estimate")
    return float(np.median(pooled) / np.log(2.0) / gain)


def extract(Y: np.ndarray, name: str, params: Mapping[str, Any] | None = None) -> Observable:
    """Extract the observable ``name`` from raw data ``Y``.

    Unknown names (after :data:`NODE_ALIASES`) and unknown ``params`` keys raise
    ValueError. ``params`` may hold ``noise_var``, ``mask_k``, ``delta_f``,
    ``pfa``, ``max_returns``, ``delay_oversample``, ``bins``, ``element`` and
    ``spacing_lambda``; see the module docstring for the node definitions.
    """
    arr = _validate_y(Y)
    options = {} if params is None else dict(params)
    unknown = set(options) - _ALLOWED_PARAMS
    if unknown:
        raise ValueError(f"unknown params key(s): {sorted(unknown)}")
    canonical = NODE_ALIASES.get(name, name)
    if canonical not in NODE_NAMES:
        raise ValueError(f"unknown observable node {name!r}")

    num_views, num_bs, num_hemispheres, num_rows, num_cols, num_bins = arr.shape
    num_elements = num_rows * num_cols
    n0 = num_bins // 2

    mask_k = float(options.get("mask_k", DEFAULT_MASK_K))
    if not np.isfinite(mask_k) or mask_k < 0.0:
        raise ValueError("mask_k must be finite and >= 0")
    spacing = float(options.get("spacing_lambda", 0.5))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing_lambda must be finite and > 0")
    noise_var_raw = options.get("noise_var")
    noise_var = None if noise_var_raw is None else float(noise_var_raw)
    if noise_var is not None and (not np.isfinite(noise_var) or noise_var < 0.0):
        raise ValueError("noise_var must be finite and >= 0")
    delta_f = _optional_positive(options.get("delta_f"), "delta_f")

    if canonical in _PHAT_NODES and mask_k > 0.0 and noise_var is None:
        raise ValueError(f"node {canonical!r} requires params['noise_var'] when mask_k > 0")
    if canonical in _DELAY_NODES and delta_f is None:
        raise ValueError(f"node {canonical!r} requires params['delta_f']")

    meta: dict[str, Any] = {"node": canonical, "domain": _DOMAINS[canonical]}
    axes_delta_f = 1.0 if delta_f is None else delta_f

    def phat(values: np.ndarray, magnitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = _phat_mask(magnitude, mask_k, noise_var)
        return _unit_modulus(values, valid), valid

    if canonical == "I":
        power = np.abs(angle_delay_volume(arr)) ** 2
        data = power.sum(axis=-1).astype(np.float64)
        mask = np.ones(data.shape, dtype=bool)
        meta.update(_beam_axes(num_cols, num_rows, num_bins, axes_delta_f, spacing))
        meta["noise_dof"] = 2 * num_bins
    elif canonical == "I_n0":
        beam = _beam(arr[..., n0])
        data = (np.abs(beam) ** 2).astype(np.float64)
        mask = np.ones(data.shape, dtype=bool)
        meta.update(_beam_axes(num_cols, num_rows, num_bins, axes_delta_f, spacing))
        meta["n0"] = n0
        meta["noise_dof"] = 2
    elif canonical == "ID":
        data = (np.abs(angle_delay_volume(arr)) ** 2).astype(np.float64)
        mask = np.ones(data.shape, dtype=bool)
        meta.update(_beam_axes(num_cols, num_rows, num_bins, axes_delta_f, spacing))
        meta["noise_dof"] = 2
    elif canonical == "T":
        power = np.abs(angle_delay_volume(arr)) ** 2
        total = power.sum(axis=-1, keepdims=True)
        valid = total > 0.0
        data = np.zeros_like(power)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(power, total, out=data, where=valid)
        mask = np.broadcast_to(valid, power.shape).copy()
        meta.update(_beam_axes(num_cols, num_rows, num_bins, axes_delta_f, spacing))
    elif canonical in ("D", "D_PHAT"):
        assert delta_f is not None
        if canonical == "D_PHAT":
            source = _unit_modulus(arr, _phat_mask(np.abs(arr), mask_k, noise_var))
        else:
            source = arr
        data, mask, return_meta = _delay_returns(
            source, delta_f, options, num_bins, num_hemispheres
        )
        meta.update(_beam_axes(num_cols, num_rows, num_bins, delta_f, spacing))
        meta.update(return_meta)
    elif canonical == "IP":
        data = arr[..., n0].copy()
        mask = np.ones(data.shape, dtype=bool)
        meta["n0"] = n0
    elif canonical == "IP_W":
        data = _canonicalise(arr, (2, 3, 4))
        mask = np.ones(data.shape, dtype=bool)
    elif canonical == "IDP":
        data = arr.copy()
        mask = np.ones(data.shape, dtype=bool)
    elif canonical == "IDP-o":
        data = _canonicalise(arr, (5,))
        mask = np.ones(data.shape, dtype=bool)
    elif canonical == "IP-o":
        data = np.abs(arr).astype(np.float64)
        mask = np.ones(data.shape, dtype=bool)
    elif canonical == "P":
        data, mask = phat(arr[..., n0], np.abs(arr[..., n0]))
        meta["n0"] = n0
    elif canonical == "P_W":
        data, mask = phat(_canonicalise(arr, (2, 3, 4)), np.abs(arr))
    elif canonical == "DP":
        data, mask = phat(arr, np.abs(arr))
    elif canonical == "DP-o":
        data, mask = phat(_canonicalise(arr, (5,)), np.abs(arr))
    elif canonical == "P-o":
        data = np.empty((num_views, num_bs, num_hemispheres, 0), dtype=np.float64)
        mask = np.empty((num_views, num_bs, num_hemispheres, 0), dtype=bool)
    elif canonical == "ID-o":
        transform = _delay_transform(arr, num_out=num_bins, window=None)
        data = (np.abs(transform) ** 2).sum(axis=(3, 4)).astype(np.float64)
        mask = np.ones(data.shape, dtype=bool)
        meta["noise_dof"] = 2 * num_elements
    elif canonical == "I-o":
        data = (np.abs(arr) ** 2).sum(axis=(3, 4, 5)).astype(np.float64)
        mask = np.ones(data.shape, dtype=bool)
        meta["noise_dof"] = 2 * num_elements * num_bins
    elif canonical == "D-o":
        assert delta_f is not None
        data, mask, return_meta = _omni_delay_returns(arr, delta_f, options, num_bins)
        meta.update(return_meta)
    elif canonical in ("IDP-1el", "DP-1el"):
        element = _element(options, num_rows, num_cols)
        slice_nd = arr[..., element[0], element[1], :]
        if canonical == "IDP-1el":
            data = slice_nd.copy()
            mask = np.ones(data.shape, dtype=bool)
        else:
            data, mask = phat(slice_nd, np.abs(slice_nd))
        meta["element"] = element
    elif canonical in ("IPxK", "PxK"):
        bins = _partial_bins(options, num_bins)
        slice_k = arr[..., bins]
        if canonical == "IPxK":
            data = slice_k.copy()
            mask = np.ones(data.shape, dtype=bool)
        else:
            data, mask = phat(slice_k, np.abs(slice_k))
        meta["bins"] = list(bins)
    else:  # pragma: no cover - exhaustive over NODE_NAMES
        raise ValueError(f"unhandled observable node {canonical!r}")

    return Observable(
        data=np.asarray(data),
        mask=np.asarray(mask, dtype=bool),
        noise_var=noise_var,
        meta=meta,
    )


def _validate_y(Y: np.ndarray) -> np.ndarray:
    """Return ``Y`` as complex128, requiring the rank-6 raw-data layout."""
    arr = np.asarray(Y, dtype=np.complex128)
    if arr.ndim != 6:
        raise ValueError("Y must have shape [V, B, H, R, C, N]")
    return arr


def _validate_oversample(oversample: tuple[int, int]) -> tuple[int, int]:
    """Return the ``(angle, delay)`` oversampling pair, both integers >= 1."""
    try:
        values = tuple(oversample)
    except TypeError as error:
        raise ValueError("oversample must be a pair of integers >= 1") from error
    if len(values) != 2:
        raise ValueError("oversample must be a pair of integers >= 1")
    factors = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("oversample factors must be integers")
        if int(value) < 1:
            raise ValueError("oversample factors must be >= 1")
        factors.append(int(value))
    return factors[0], factors[1]


def _window_vectors(
    rows: int, cols: int, num_bins: int, window: str | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the separable ``(row, col, frequency)`` window vectors."""
    if window is None:
        return (
            np.ones(rows, dtype=np.float64),
            np.ones(cols, dtype=np.float64),
            np.ones(num_bins, dtype=np.float64),
        )
    if window != "taylor":
        raise ValueError(f"unknown window {window!r}; expected None or 'taylor'")
    return taylor_window(rows), taylor_window(cols), taylor_window(num_bins)


def _optional_positive(value: Any, label: str) -> float | None:
    """Return ``value`` as a finite positive float, or None when it is None."""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and > 0")
    return number


def _beam_axes(qy: int, qz: int, nt: int, delta_f: float, spacing: float) -> dict[str, np.ndarray]:
    """Return the ``u_y``/``u_z`` metadata of the native beam grid."""
    u_y, u_z, _ = volume_axes(qy, qz, nt, delta_f=delta_f, spacing_lambda=spacing)
    return {"u_y": u_y, "u_z": u_z}


def _beam(slice_rc: np.ndarray) -> np.ndarray:
    """Return the unitary angle-only transform ``[..., C, R]`` of ``[..., R, C]``."""
    return angle_delay_volume(slice_rc[..., None])[..., 0]


def _phat_mask(magnitude: np.ndarray, mask_k: float, noise_var: float | None) -> np.ndarray:
    """Return the PHAT validity mask for a magnitude array."""
    valid = np.asarray(magnitude) > 0.0
    if mask_k > 0.0:
        assert noise_var is not None
        valid = valid & (np.asarray(magnitude) >= mask_k * np.sqrt(noise_var))
    return valid


def _unit_modulus(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return unit-modulus ``values`` where ``mask`` holds, else zero."""
    magnitude = np.abs(values)
    out = np.zeros_like(values, dtype=np.complex128)
    nonzero = magnitude > 0.0
    out[nonzero] = values[nonzero] / magnitude[nonzero]
    return np.where(mask, out, 0.0).astype(np.complex128)


def _canonicalise(values: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Remove one unit-modulus unknown per slice outside ``axes``."""
    arr = np.asarray(values, dtype=np.complex128)
    axes = tuple(int(axis) for axis in axes)
    if any(axis < 0 or axis >= arr.ndim for axis in axes) or len(set(axes)) != len(axes):
        raise ValueError("axes must be distinct valid dimensions")
    other = tuple(index for index in range(arr.ndim) if index not in axes)
    permutation = other + axes
    transposed = np.transpose(arr, permutation)
    other_shape = transposed.shape[: len(other)]
    axes_shape = transposed.shape[len(other) :]
    flat = transposed.reshape(other_shape + (-1,))
    reference = np.take_along_axis(flat, np.argmax(np.abs(flat), axis=-1)[..., None], axis=-1)[
        ..., 0
    ]
    reference_magnitude = np.abs(reference)
    phase = np.ones_like(reference)
    nonzero = reference_magnitude > 0.0
    phase[nonzero] = np.conj(reference[nonzero]) / reference_magnitude[nonzero]
    out = (flat * phase[..., None]).reshape(other_shape + axes_shape)
    return np.transpose(out, np.argsort(permutation)).astype(np.complex128, copy=False)


def _delay_transform(values: np.ndarray, *, num_out: int, window: np.ndarray | None) -> np.ndarray:
    """Return ``1/sqrt(N) sum_n w_n Y exp(+j 2 pi (n - N//2) k / num_out)``."""
    num_bins = values.shape[-1]
    weighted = values if window is None else values * window
    out = num_out * np.fft.ifft(weighted, n=num_out, axis=-1)
    out = out * np.exp(-2j * np.pi * (num_bins // 2) * np.arange(num_out) / num_out)
    return out / np.sqrt(num_bins)


def _delay_returns(
    source: np.ndarray,
    delta_f: float,
    options: Mapping[str, Any],
    num_bins: int,
    num_hemispheres: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the D/D_PHAT delay-return data, mask and metadata."""
    delay_oversample = int(options.get("delay_oversample", DEFAULT_DELAY_OVERSAMPLE))
    pfa = float(options.get("pfa", DEFAULT_PFA))
    max_returns = int(options.get("max_returns", DEFAULT_MAX_RETURNS))
    volume = angle_delay_volume(source, window="taylor", oversample=(1, delay_oversample))
    returns = cfar_returns(np.abs(volume) ** 2, pfa, max_returns)
    period = 1.0 / delta_f
    num_out = delay_oversample * num_bins
    data = returns.position * (period / num_out)
    meta = {
        "t_period": period,
        "pfa": pfa,
        "max_returns": max_returns,
        "delay_oversample": delay_oversample,
    }
    return data, returns.mask, meta


def _omni_delay_returns(
    arr: np.ndarray,
    delta_f: float,
    options: Mapping[str, Any],
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the D-o delay-return data, mask and metadata."""
    delay_oversample = int(options.get("delay_oversample", DEFAULT_DELAY_OVERSAMPLE))
    pfa = float(options.get("pfa", DEFAULT_PFA))
    max_returns = int(options.get("max_returns", DEFAULT_MAX_RETURNS))
    num_out = delay_oversample * num_bins
    transform = _delay_transform(arr, num_out=num_out, window=taylor_window(num_bins))
    profile = (np.abs(transform) ** 2).sum(axis=(2, 3, 4))
    returns = cfar_returns(profile, pfa, max_returns)
    period = 1.0 / delta_f
    data = returns.position * (period / num_out)
    meta = {
        "t_period": period,
        "pfa": pfa,
        "max_returns": max_returns,
        "delay_oversample": delay_oversample,
    }
    return data, returns.mask, meta


def _element(options: Mapping[str, Any], num_rows: int, num_cols: int) -> tuple[int, int]:
    """Return the validated ``(row, col)`` element index for the 1el nodes."""
    value = options.get("element", ONE_ELEMENT)
    try:
        row, col = (int(entry) for entry in value)
    except (TypeError, ValueError) as error:
        raise ValueError("element must be a pair of integers") from error
    if not 0 <= row < num_rows or not 0 <= col < num_cols:
        raise ValueError("element lies outside the aperture")
    return row, col


def _partial_bins(options: Mapping[str, Any], num_bins: int) -> tuple[int, ...]:
    """Return the validated partial-D bin indices."""
    if "bins" in options:
        bins = np.asarray(options["bins"], dtype=np.int64)
        if bins.ndim != 1 or bins.size == 0:
            raise ValueError("bins must be a non-empty one-dimensional sequence")
        if np.any(bins < 0) or np.any(bins >= num_bins):
            raise ValueError("bins lie outside the frequency axis")
        return tuple(int(value) for value in bins)
    k = PARTIAL_D_NUM_BINS
    return tuple((2 * index + 1) * num_bins // (2 * k) for index in range(k))
