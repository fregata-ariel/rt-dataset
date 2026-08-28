"""Develop calibrated angular CFR into an angle-delay RF volume.

This is a CPU-only post-processing stage. It takes the physically calibrated
complex angular CFR produced by :mod:`rf_camera_calibration` and performs an
IFFT along the uniformly sampled baseband-frequency axis.

The resulting tensor keeps complex phase and has axes

    [UE-local kz/k, UE-local ky/k, delay]

so delay bins can later be treated as image channels or as a small RF volume.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plateau_rt.adapters.sionna.rf_camera_calibration import (
    geometric_los_source_direction_local,
)

SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class AngularDelayVolume:
    """Complex angle-delay response and its physical axes."""

    cir: np.ndarray
    delay_s: np.ndarray
    frequency_spacing_hz: float
    unambiguous_delay_s: float


def angular_cfr_to_delay(
    angular_cfr: np.ndarray,
    frequency_offsets_hz: np.ndarray,
) -> AngularDelayVolume:
    """IFFT a centered, uniformly sampled CFR into positive modulo-delay bins.

    ``frequency_offsets_hz`` is expected to contain the baseband offsets used
    for ``Paths.cfr()``, ordered from negative to positive frequency. The
    resulting delay axis starts at zero and spans one unambiguous delay period
    ``1 / delta_f``. Absolute Sionna delays therefore appear modulo this period.
    """
    cfr = np.asarray(angular_cfr)
    frequencies = np.asarray(frequency_offsets_hz, dtype=np.float64)

    if cfr.ndim != 3:
        raise ValueError("angular_cfr must have shape [kz, ky, frequency]")
    if frequencies.ndim != 1 or frequencies.size != cfr.shape[-1]:
        raise ValueError("frequency_offsets_hz must match the CFR frequency axis")
    if frequencies.size < 2:
        raise ValueError("at least two frequency bins are required for delay imaging")

    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    cfr = cfr[..., order]

    differences = np.diff(frequencies)
    delta_f = float(np.median(differences))
    if delta_f <= 0.0:
        raise ValueError("frequency offsets must contain distinct increasing bins")
    if not np.allclose(differences, delta_f, rtol=1e-6, atol=max(1e-3, abs(delta_f) * 1e-9)):
        raise ValueError("frequency offsets must be uniformly spaced")

    # The stored frequency axis is centered as [-B/2, ..., 0, ..., +B/2).
    # Move DC to index zero before using NumPy's inverse DFT convention.
    cir = np.fft.ifft(np.fft.ifftshift(cfr, axes=-1), axis=-1)

    num_bins = frequencies.size
    delay_resolution_s = 1.0 / (num_bins * delta_f)
    delay_s = np.arange(num_bins, dtype=np.float64) * delay_resolution_s

    return AngularDelayVolume(
        cir=cir,
        delay_s=delay_s,
        frequency_spacing_hz=delta_f,
        unambiguous_delay_s=1.0 / delta_f,
    )


def direction_axes_from_metadata(
    *,
    fft_rows: int,
    fft_cols: int,
    horizontal_spacing_lambda: float,
    vertical_spacing_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct calibrated UE-local ``ky/k`` and ``kz/k`` axes."""
    q_col = np.fft.fftshift(np.fft.fftfreq(fft_cols))
    q_row = np.fft.fftshift(np.fft.fftfreq(fft_rows))
    ky_over_k = q_col / horizontal_spacing_lambda
    kz_over_k = -q_row[::-1] / vertical_spacing_lambda
    return ky_over_k, kz_over_k


def propagating_direction_mask(
    ky_over_k: np.ndarray,
    kz_over_k: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Return the far-field propagating disk for a planar y-z aperture.

    A real plane-wave direction must satisfy ``kx^2 + ky^2 + kz^2 = 1``.
    The planar aperture does not determine the sign of ``kx`` but it does
    determine whether a sampled y-z projection can correspond to a propagating
    wave at all.
    """
    ky = np.asarray(ky_over_k, dtype=np.float64)[None, :]
    kz = np.asarray(kz_over_k, dtype=np.float64)[:, None]
    return ky**2 + kz**2 <= 1.0 + tolerance


def geometric_los_delay_s(
    tx_position: tuple[float, float, float],
    ue_position: tuple[float, float, float],
) -> float:
    """Free-space geometric delay from BS to UE aperture center."""
    tx = np.asarray(tx_position, dtype=np.float64)
    ue = np.asarray(ue_position, dtype=np.float64)
    return float(np.linalg.norm(tx - ue) / SPEED_OF_LIGHT_M_S)


def circular_delay_error_s(value: float, reference: float, period: float) -> float:
    """Shortest delay error on a modulo-delay circle."""
    if period <= 0.0:
        raise ValueError("period must be > 0")
    difference = abs((value - reference) % period)
    return float(min(difference, period - difference))


def _earliest_path_delay(path_gt_path: Path) -> float | None:
    if not path_gt_path.exists():
        return None
    with np.load(path_gt_path) as payload:
        if "tau" not in payload:
            return None
        tau = np.asarray(payload["tau"], dtype=np.float64)
        valid = np.isfinite(tau) & (tau >= 0.0)
        if "valid" in payload:
            valid_values = np.asarray(payload["valid"], dtype=bool)
            if valid_values.shape == tau.shape:
                valid &= valid_values
        values = tau[valid]
        if values.size == 0:
            return None
        return float(np.min(values))


def develop_angle_delay(
    output_dir: Path,
    *,
    power_floor_db: float = -35.0,
) -> dict[str, Path]:
    """Build and validate an angle-delay RF volume from one calibrated view."""
    output_dir = Path(output_dir)
    cfr_path = output_dir / "angular_cfr_calibrated.npy"
    metadata_path = output_dir / "rf_camera_metadata.json"
    if not cfr_path.exists():
        raise FileNotFoundError(cfr_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    cfr = np.load(cfr_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cfg = metadata["config"]
    frequency_offsets_hz = np.asarray(metadata["frequency_offsets_hz"], dtype=np.float64)

    volume = angular_cfr_to_delay(cfr, frequency_offsets_hz)
    ky_over_k, kz_over_k = direction_axes_from_metadata(
        fft_rows=cfr.shape[0],
        fft_cols=cfr.shape[1],
        horizontal_spacing_lambda=float(cfg["horizontal_spacing_lambda"]),
        vertical_spacing_lambda=float(cfg["vertical_spacing_lambda"]),
    )
    physical_mask = propagating_direction_mask(ky_over_k, kz_over_k)

    power = np.abs(volume.cir) ** 2
    masked_power = np.where(physical_mask[:, :, None], power, -np.inf)
    peak_row, peak_col, peak_delay_bin = np.unravel_index(
        int(np.argmax(masked_power)), masked_power.shape
    )

    los_local = geometric_los_source_direction_local(
        tx_position=tuple(cfg["tx_position"]),
        ue_position=tuple(cfg["ue_position"]),
        ue_orientation=tuple(cfg["ue_orientation"]),
    )
    los_ky = float(los_local[1])
    los_kz = float(los_local[2])
    los_row = int(np.argmin(np.abs(kz_over_k - los_kz)))
    los_col = int(np.argmin(np.abs(ky_over_k - los_ky)))

    geometric_delay = geometric_los_delay_s(
        tuple(cfg["tx_position"]), tuple(cfg["ue_position"])
    )
    geometric_delay_mod = geometric_delay % volume.unambiguous_delay_s

    los_profile = power[los_row, los_col, :]
    los_delay_bin = int(np.argmax(los_profile))
    los_profile_delay = float(volume.delay_s[los_delay_bin])
    los_delay_error = circular_delay_error_s(
        los_profile_delay,
        geometric_delay_mod,
        volume.unambiguous_delay_s,
    )

    strongest_delay = float(volume.delay_s[peak_delay_bin])
    strongest_delay_error = circular_delay_error_s(
        strongest_delay,
        geometric_delay_mod,
        volume.unambiguous_delay_s,
    )

    earliest_path = _earliest_path_delay(output_dir / "path_gt.npz")

    cir_path = output_dir / "angular_delay_cir.npy"
    delay_axis_path = output_dir / "delay_axis_s.npy"
    mask_path = output_dir / "propagating_direction_mask.npy"
    np.save(cir_path, volume.cir.astype(np.complex64, copy=False))
    np.save(delay_axis_path, volume.delay_s)
    np.save(mask_path, physical_mask)

    print("=== RF Camera angle-delay development ===")
    print(
        f"frequency spacing={volume.frequency_spacing_hz / 1e3:.3f} kHz, "
        f"delay resolution={1e9 / (len(volume.delay_s) * volume.frequency_spacing_hz):.3f} ns, "
        f"unambiguous delay={volume.unambiguous_delay_s * 1e9:.3f} ns"
    )
    print(
        "geometric LoS delay: "
        f"absolute={geometric_delay * 1e9:.3f} ns, "
        f"modulo={geometric_delay_mod * 1e9:.3f} ns"
    )
    print(
        "nearest LoS angular bin: "
        f"ky/k={ky_over_k[los_col]:+.6f}, kz/k={kz_over_k[los_row]:+.6f}, "
        f"delay peak={los_profile_delay * 1e9:.3f} ns, "
        f"error={los_delay_error * 1e9:.3f} ns"
    )
    print(
        "strongest physical voxel: "
        f"ky/k={ky_over_k[peak_col]:+.6f}, kz/k={kz_over_k[peak_row]:+.6f}, "
        f"delay={strongest_delay * 1e9:.3f} ns, "
        f"delay error to LoS={strongest_delay_error * 1e9:.3f} ns"
    )
    if earliest_path is not None:
        print(f"earliest path_gt delay={earliest_path * 1e9:.3f} ns")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [
        float(ky_over_k[0]),
        float(ky_over_k[-1]),
        float(kz_over_k[0]),
        float(kz_over_k[-1]),
    ]

    global_peak_power = max(float(np.max(power[physical_mask, :])), 1e-30)

    delay_slice = power[:, :, peak_delay_bin]
    delay_slice_db = 10.0 * np.log10(
        np.maximum(delay_slice / global_peak_power, 1e-12)
    )
    delay_slice_db = np.ma.masked_where(~physical_mask, delay_slice_db)
    delay_slice_png = output_dir / "angular_power_strongest_delay.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        delay_slice_db,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=-60.0,
        vmax=0.0,
    )
    ax.scatter([los_ky], [los_kz], marker="x", label="geometric LoS")
    ax.scatter(
        [ky_over_k[peak_col]],
        [kz_over_k[peak_row]],
        marker="+",
        label="strongest voxel",
    )
    ax.set_xlabel("UE-local horizontal direction cosine ky/k")
    ax.set_ylabel("UE-local vertical direction cosine kz/k")
    ax.set_title(
        "RF camera angle-delay: normalized power at "
        f"{strongest_delay * 1e9:.1f} ns"
    )
    ax.legend()
    fig.colorbar(image, ax=ax, label="dB relative to volume peak")
    fig.tight_layout()
    fig.savefig(delay_slice_png, dpi=150)
    plt.close(fig)

    profile_peak = max(float(np.max(los_profile)), 1e-30)
    profile_db = 10.0 * np.log10(np.maximum(los_profile / profile_peak, 1e-12))
    delay_profile_png = output_dir / "delay_profile_los_direction.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(volume.delay_s * 1e9, profile_db)
    ax.axvline(geometric_delay_mod * 1e9, linestyle="--", label="geometric LoS")
    if earliest_path is not None:
        ax.axvline(
            (earliest_path % volume.unambiguous_delay_s) * 1e9,
            linestyle=":",
            label="earliest path GT",
        )
    ax.set_xlabel("delay [ns]")
    ax.set_ylabel("normalized power [dB]")
    ax.set_ylim(-60.0, 3.0)
    ax.set_title("Delay profile at nearest geometric-LoS angular bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(delay_profile_png, dpi=150)
    plt.close(fig)

    per_direction_peak = np.max(power, axis=2)
    per_direction_db = 10.0 * np.log10(
        np.maximum(per_direction_peak / global_peak_power, 1e-12)
    )
    dominant_delay_bin = np.argmax(power, axis=2)
    dominant_delay_ns = volume.delay_s[dominant_delay_bin] * 1e9
    dominant_mask = physical_mask & (per_direction_db >= power_floor_db)
    dominant_delay_plot = np.ma.masked_where(~dominant_mask, dominant_delay_ns)

    dominant_delay_png = output_dir / "dominant_delay_map.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        dominant_delay_plot,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=0.0,
        vmax=volume.unambiguous_delay_s * 1e9,
    )
    ax.scatter([los_ky], [los_kz], marker="x", label="geometric LoS")
    ax.set_xlabel("UE-local horizontal direction cosine ky/k")
    ax.set_ylabel("UE-local vertical direction cosine kz/k")
    ax.set_title(
        "RF camera dominant delay [ns] "
        f"(peak power >= {power_floor_db:g} dB)"
    )
    ax.legend()
    fig.colorbar(image, ax=ax, label="dominant delay [ns]")
    fig.tight_layout()
    fig.savefig(dominant_delay_png, dpi=150)
    plt.close(fig)

    report_path = output_dir / "angle_delay_report.json"
    report = {
        "schema_version": 1,
        "source_cfr": cfr_path.name,
        "angular_delay_cir": cir_path.name,
        "axis_order": ["kz_over_k", "ky_over_k", "delay"],
        "shape": list(volume.cir.shape),
        "frequency_spacing_hz": volume.frequency_spacing_hz,
        "delay_resolution_s": float(1.0 / (len(volume.delay_s) * volume.frequency_spacing_hz)),
        "unambiguous_delay_s": volume.unambiguous_delay_s,
        "delay_convention": "positive absolute propagation delay modulo 1/delta_f",
        "geometric_los": {
            "source_direction_local": {
                "kx_over_k": float(los_local[0]),
                "ky_over_k": los_ky,
                "kz_over_k": los_kz,
            },
            "absolute_delay_s": geometric_delay,
            "modulo_delay_s": geometric_delay_mod,
            "nearest_angular_bin": {
                "row": los_row,
                "col": los_col,
                "ky_over_k": float(ky_over_k[los_col]),
                "kz_over_k": float(kz_over_k[los_row]),
                "peak_delay_bin": los_delay_bin,
                "peak_delay_s": los_profile_delay,
                "circular_delay_error_s": los_delay_error,
            },
        },
        "strongest_physical_voxel": {
            "row": int(peak_row),
            "col": int(peak_col),
            "delay_bin": int(peak_delay_bin),
            "ky_over_k": float(ky_over_k[peak_col]),
            "kz_over_k": float(kz_over_k[peak_row]),
            "delay_s": strongest_delay,
            "circular_delay_error_to_geometric_los_s": strongest_delay_error,
        },
        "earliest_path_gt_delay_s": earliest_path,
        "propagating_direction_mask": "ky^2 + kz^2 <= 1; sign(kx) remains ambiguous",
        "dominant_delay_plot_power_floor_db": float(power_floor_db),
        "notes": [
            "delay resolution is set by total sampled bandwidth, not by zero-padding",
            "delays repeat modulo the unambiguous period 1/delta_f",
            "the rectangular frequency window produces delay sidelobes; no window is applied to the saved complex target",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"angle-delay CIR: {cir_path}")
    print(f"delay axis: {delay_axis_path}")
    print(f"propagating mask: {mask_path}")
    print(f"strongest-delay image: {delay_slice_png}")
    print(f"LoS delay profile: {delay_profile_png}")
    print(f"dominant-delay map: {dominant_delay_png}")
    print(f"angle-delay report: {report_path}")

    return {
        "cir": cir_path,
        "delay_axis": delay_axis_path,
        "direction_mask": mask_path,
        "delay_slice_png": delay_slice_png,
        "delay_profile_png": delay_profile_png,
        "dominant_delay_png": dominant_delay_png,
        "report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Develop a calibrated RF-camera CFR into an angle-delay volume"
    )
    parser.add_argument("output_dir", type=Path, help="Existing rf_camera output directory")
    parser.add_argument(
        "--power-floor-db",
        type=float,
        default=-35.0,
        help="Mask dominant-delay visualization below this relative peak power",
    )
    args = parser.parse_args()
    develop_angle_delay(args.output_dir, power_floor_db=args.power_floor_db)


if __name__ == "__main__":
    main()
