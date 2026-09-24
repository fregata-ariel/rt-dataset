"""Develop a calibrated RF-camera output directory into an angle-delay volume.

Reads ``angular_cfr_calibrated.npy`` (from ``rf-camera-calibrate``) and the
metadata, applies :func:`plateau_rt.domain.rf_camera.delay.angular_cfr_to_delay`
and writes the angle-delay CIR, diagnostic images and
``angle_delay_report.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plateau_rt.adapters.plotting.rf_camera_plots import (
    Marker,
    image_extent,
    normalized_power_db,
    pyplot,
    save_direction_image,
)
from plateau_rt.domain.rf_camera.calibration import (
    direction_cosine_axes,
    geometric_los_source_direction_local,
)
from plateau_rt.domain.rf_camera.delay import (
    angular_cfr_to_delay,
    circular_delay_error_s,
    dominant_delay,
    geometric_los_delay_s,
    propagating_direction_mask,
)


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
    ky_over_k, kz_over_k = direction_cosine_axes(
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

    geometric_delay = geometric_los_delay_s(tuple(cfg["tx_position"]), tuple(cfg["ue_position"]))
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
        f"delay resolution={volume.delay_resolution_s * 1e9:.3f} ns, "
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

    extent = image_extent(ky_over_k, kz_over_k)
    los_marker = Marker(los_ky, los_kz, "x", "geometric LoS")
    global_peak_power = max(float(np.max(power[physical_mask, :])), 1e-30)

    delay_slice_db = normalized_power_db(power[:, :, peak_delay_bin], global_peak_power)
    delay_slice_png = save_direction_image(
        np.ma.masked_where(~physical_mask, delay_slice_db),
        output_dir / "angular_power_strongest_delay.png",
        extent=extent,
        title=f"RF camera angle-delay: normalized power at {strongest_delay * 1e9:.1f} ns",
        colorbar_label="dB relative to volume peak",
        vmin=-60.0,
        vmax=0.0,
        markers=[
            los_marker,
            Marker(float(ky_over_k[peak_col]), float(kz_over_k[peak_row]), "+", "strongest voxel"),
        ],
    )

    profile_db = normalized_power_db(los_profile, max(float(np.max(los_profile)), 1e-30))
    delay_profile_png = output_dir / "delay_profile_los_direction.png"
    plt = pyplot()
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

    _, dominant_delay_s, per_direction_peak = dominant_delay(power, volume.delay_s)
    per_direction_db = normalized_power_db(per_direction_peak, global_peak_power)
    dominant_mask = physical_mask & (per_direction_db >= power_floor_db)
    dominant_delay_png = save_direction_image(
        np.ma.masked_where(~dominant_mask, dominant_delay_s * 1e9),
        output_dir / "dominant_delay_map.png",
        extent=extent,
        title=f"RF camera dominant delay [ns] (peak power >= {power_floor_db:g} dB)",
        colorbar_label="dominant delay [ns]",
        vmin=0.0,
        vmax=volume.unambiguous_delay_s * 1e9,
        markers=[los_marker],
    )

    report_path = output_dir / "angle_delay_report.json"
    report = {
        "schema_version": 1,
        "source_cfr": cfr_path.name,
        "angular_delay_cir": cir_path.name,
        "axis_order": ["kz_over_k", "ky_over_k", "delay"],
        "shape": list(volume.cir.shape),
        "frequency_spacing_hz": volume.frequency_spacing_hz,
        "delay_resolution_s": float(volume.delay_resolution_s),
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
            "the rectangular frequency window produces delay sidelobes; "
            "no window is applied to the saved complex target",
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
