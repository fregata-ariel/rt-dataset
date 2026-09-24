"""Calibrate an existing RF-camera output directory without re-running tracing.

Reads ``angular_cfr.npy`` and ``rf_camera_metadata.json`` written by
``rf-camera``, applies the physical calibration of
:mod:`plateau_rt.domain.rf_camera.calibration`, and writes the calibrated CFR,
diagnostic images and ``angular_calibration.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plateau_rt.domain.rf_camera.calibration import (
    angular_peak_projection,
    calibrate_angular_cfr,
    geometric_los_source_direction_local,
)


def calibrate_directory(output_dir: Path, phase_floor_db: float = -35.0) -> dict[str, Path]:
    """Calibrate one existing RF-camera output directory and render diagnostics."""
    output_dir = Path(output_dir)
    raw_path = output_dir / "angular_cfr.npy"
    metadata_path = output_dir / "rf_camera_metadata.json"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    raw = np.load(raw_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cfg = metadata["config"]

    calibration = calibrate_angular_cfr(
        raw,
        aperture_rows=int(cfg["rx_rows"]),
        aperture_cols=int(cfg["rx_cols"]),
        horizontal_spacing_lambda=float(cfg["horizontal_spacing_lambda"]),
        vertical_spacing_lambda=float(cfg["vertical_spacing_lambda"]),
    )

    calibrated_path = output_dir / "angular_cfr_calibrated.npy"
    np.save(calibrated_path, calibration.cfr.astype(np.complex64, copy=False))

    center_bin = calibration.cfr.shape[2] // 2
    angular_slice = calibration.cfr[:, :, center_bin]
    peak_ky, peak_kz, peak_index = angular_peak_projection(
        angular_slice,
        ky_over_k=calibration.ky_over_k,
        kz_over_k=calibration.kz_over_k,
    )

    los_local = geometric_los_source_direction_local(
        tx_position=tuple(cfg["tx_position"]),
        ue_position=tuple(cfg["ue_position"]),
        ue_orientation=tuple(cfg["ue_orientation"]),
    )
    los_ky = float(los_local[1])
    los_kz = float(los_local[2])
    los_projection_error = float(np.hypot(peak_ky - los_ky, peak_kz - los_kz))

    print("=== RF Camera angular calibration ===")
    print(
        "geometric LoS source direction, UE-local: "
        f"kx/k={los_local[0]:+.6f}, ky/k={los_ky:+.6f}, kz/k={los_kz:+.6f}"
    )
    print(f"center-bin angular peak: ky/k={peak_ky:+.6f}, kz/k={peak_kz:+.6f}, index={peak_index}")
    print(f"peak-to-LoS yz projection error={los_projection_error:.6f}")
    print(
        "note: a planar y-z aperture has front/back ambiguity in local x; "
        "the angular image alone does not determine the sign of kx/k"
    )

    power = np.abs(angular_slice) ** 2
    peak_power = max(float(np.max(power)), 1e-30)
    power_db = 10.0 * np.log10(np.maximum(power / peak_power, 1e-12))
    phase = np.angle(angular_slice)
    phase_masked = np.ma.masked_where(power_db < phase_floor_db, phase)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [
        float(calibration.ky_over_k[0]),
        float(calibration.ky_over_k[-1]),
        float(calibration.kz_over_k[0]),
        float(calibration.kz_over_k[-1]),
    ]

    power_png = output_dir / "angular_power_center_calibrated.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        power_db,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=-60.0,
        vmax=0.0,
    )
    ax.scatter([los_ky], [los_kz], marker="x", label="geometric LoS")
    ax.scatter([peak_ky], [peak_kz], marker="+", label="strongest bin")
    ax.set_xlabel("UE-local horizontal direction cosine ky/k")
    ax.set_ylabel("UE-local vertical direction cosine kz/k")
    ax.set_title("RF camera angular spectrum: calibrated normalized power [dB]")
    ax.legend()
    fig.colorbar(image, ax=ax, label="dB relative to peak")
    fig.tight_layout()
    fig.savefig(power_png, dpi=150)
    plt.close(fig)

    phase_png = output_dir / "angular_phase_center_calibrated.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        phase_masked,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=-np.pi,
        vmax=np.pi,
    )
    ax.scatter([los_ky], [los_kz], marker="x", label="geometric LoS")
    ax.set_xlabel("UE-local horizontal direction cosine ky/k")
    ax.set_ylabel("UE-local vertical direction cosine kz/k")
    ax.set_title(
        f"RF camera angular spectrum: calibrated phase [rad] (power >= {phase_floor_db:g} dB)"
    )
    ax.legend()
    fig.colorbar(image, ax=ax, label="phase [rad]")
    fig.tight_layout()
    fig.savefig(phase_png, dpi=150)
    plt.close(fig)

    report_path = output_dir / "angular_calibration.json"
    report = {
        "schema_version": 1,
        "source_angular_cfr": raw_path.name,
        "calibrated_angular_cfr": calibrated_path.name,
        "coordinate_convention": {
            "array_plane": "UE-local y-z",
            "horizontal_axis": "local ky/k, increasing toward +y",
            "vertical_axis": "local kz/k, increasing toward +z",
            "array_normal": "local x",
            "front_back_ambiguity": "planar phase sampling does not determine sign of local kx/k",
        },
        "phase_origin": "UE/aperture center",
        "geometric_los_source_direction_local": {
            "kx_over_k": float(los_local[0]),
            "ky_over_k": los_ky,
            "kz_over_k": los_kz,
        },
        "center_frequency_peak": {
            "ky_over_k": peak_ky,
            "kz_over_k": peak_kz,
            "index": list(peak_index),
            "yz_projection_error_to_geometric_los": los_projection_error,
        },
        "phase_plot_power_floor_db": float(phase_floor_db),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"calibrated CFR: {calibrated_path}")
    print(f"calibrated power: {power_png}")
    print(f"calibrated phase: {phase_png}")
    print(f"calibration report: {report_path}")

    return {
        "calibrated_cfr": calibrated_path,
        "power_png": power_png,
        "phase_png": phase_png,
        "report": report_path,
    }
