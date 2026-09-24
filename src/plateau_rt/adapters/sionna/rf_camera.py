"""1 BS / 1 UE RF-camera MVP built on Sionna RT.

The raw observation is a complex channel frequency response (CFR) sampled
across a planar receive aperture. A simple 2-D spatial FFT is used as the
first "development" step to produce a human-inspectable angular-spectrum
image. This module intentionally keeps the RF-camera image formation separate
from Sionna's optical scene renderer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sionna.rt import Receiver, Transmitter, load_scene

from plateau_rt.adapters.plotting.rf_camera_plots import (
    image_extent,
    normalized_power_db,
    save_direction_image,
)
from plateau_rt.adapters.sionna.rf_tracing import (
    PATH_ANGLE_FIELDS,
    aperture_cfrs,
    configure_rf_camera_arrays,
    path_attributes,
    trace_paths,
)
from plateau_rt.application.scene_checks import check_scene_carrier_frequency
from plateau_rt.domain.rf_camera.imaging import (
    aperture_to_angular_fft,
    frequency_offsets,
    raw_spatial_frequency_axes,
)


@dataclass(frozen=True)
class RFCameraConfig:
    """Configuration for the first 1-BS / 1-UE RF-camera milestone."""

    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 100e6
    num_frequency_bins: int = 64

    # Keep Tx spatially simple for the first milestone. The RF-camera aperture
    # is the Rx array; Tx beamforming will be introduced independently later.
    tx_position: tuple[float, float, float] = (-50.0, -50.0, 30.0)
    tx_orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    ue_position: tuple[float, float, float] = (0.0, 0.0, 1.5)
    ue_orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    rx_rows: int = 8
    rx_cols: int = 8
    vertical_spacing_lambda: float = 0.5
    horizontal_spacing_lambda: float = 0.5

    max_depth: int = 5
    synthetic_array: bool = True
    seed: int = 42

    # Display only. Zero-padding improves readability of the FFT image but does
    # not increase the physical angular resolution of the aperture.
    fft_rows: int = 128
    fft_cols: int = 128

    def validate(self) -> None:
        if self.num_frequency_bins < 1:
            raise ValueError("num_frequency_bins must be >= 1")
        if self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be > 0")
        if self.rx_rows < 1 or self.rx_cols < 1:
            raise ValueError("rx_rows and rx_cols must be >= 1")
        if self.fft_rows < self.rx_rows or self.fft_cols < self.rx_cols:
            raise ValueError("FFT grid must not be smaller than the receive aperture")


@dataclass(frozen=True)
class RFCameraArtifacts:
    aperture_cfr: Path
    angular_cfr: Path
    metadata: Path
    power_png: Path
    phase_png: Path
    path_gt: Path


class RFCameraMVP:
    """Generate one coherent RF-camera view from one BS and one UE aperture."""

    def __init__(self, xml_path: Path, config: RFCameraConfig | None = None):
        self.xml_path = Path(xml_path)
        self.config = config or RFCameraConfig()
        self.config.validate()

    def run(self, output_dir: Path) -> RFCameraArtifacts:
        """Trace paths, export aperture CFR, and develop an FFT RF image."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config
        check_scene_carrier_frequency(self.xml_path, cfg.carrier_frequency_hz)
        scene = load_scene(str(self.xml_path))
        scene.frequency = cfg.carrier_frequency_hz
        configure_rf_camera_arrays(
            scene,
            rx_rows=cfg.rx_rows,
            rx_cols=cfg.rx_cols,
            vertical_spacing_lambda=cfg.vertical_spacing_lambda,
            horizontal_spacing_lambda=cfg.horizontal_spacing_lambda,
            tx_pattern="tr38901",
            rx_pattern="dipole",
        )

        tx = Transmitter(
            name="rf_camera_bs_0",
            position=list(cfg.tx_position),
            orientation=list(cfg.tx_orientation),
        )
        rx = Receiver(
            name="rf_camera_ue_0",
            position=list(cfg.ue_position),
            orientation=list(cfg.ue_orientation),
        )
        scene.add(tx)
        scene.add(rx)

        print("=== RF Camera MVP: path tracing ===")
        print(f"scene={self.xml_path}")
        print(f"carrier={cfg.carrier_frequency_hz / 1e9:.6f} GHz")
        print(f"BS position={cfg.tx_position}")
        print(f"UE position={cfg.ue_position}, orientation={cfg.ue_orientation} rad")
        print(
            "Rx aperture="
            f"{cfg.rx_rows}x{cfg.rx_cols}, spacing="
            f"({cfg.vertical_spacing_lambda}, {cfg.horizontal_spacing_lambda}) lambda"
        )

        paths = trace_paths(
            scene,
            max_depth=cfg.max_depth,
            synthetic_array=cfg.synthetic_array,
            seed=cfg.seed,
        )
        frequency_offsets_hz = frequency_offsets(cfg.bandwidth_hz, cfg.num_frequency_bins)
        # Single receiver, single (dipole) pattern -> [row, col, frequency]
        aperture_cfr = aperture_cfrs(
            paths,
            frequency_offsets_hz,
            num_rx=1,
            rx_rows=cfg.rx_rows,
            rx_cols=cfg.rx_cols,
        )[0, 0]
        print(f"aperture_cfr shape={aperture_cfr.shape}, dtype={aperture_cfr.dtype}")

        angular_cfr = aperture_to_angular_fft(
            aperture_cfr,
            fft_rows=cfg.fft_rows,
            fft_cols=cfg.fft_cols,
        )
        print(f"angular_cfr shape={angular_cfr.shape}, dtype={angular_cfr.dtype}")

        aperture_path = output_dir / "aperture_cfr.npy"
        angular_path = output_dir / "angular_cfr.npy"
        np.save(aperture_path, aperture_cfr.astype(np.complex64, copy=False))
        np.save(angular_path, angular_cfr.astype(np.complex64, copy=False))

        path_gt_path = output_dir / "path_gt.npz"
        # cir() returns complex coefficients and delays. Keep absolute delays.
        a, tau = paths.cir(normalize_delays=False, out_type="numpy")
        np.savez_compressed(
            path_gt_path,
            a=np.asarray(a),
            tau=np.asarray(tau),
            **path_attributes(paths, ("valid",) + PATH_ANGLE_FIELDS),
        )

        center_bin = cfg.num_frequency_bins // 2
        power_png = output_dir / "angular_power_center.png"
        phase_png = output_dir / "angular_phase_center.png"
        _render_raw_fft_images(
            angular_cfr[:, :, center_bin],
            power_png=power_png,
            phase_png=phase_png,
            horizontal_spacing_lambda=cfg.horizontal_spacing_lambda,
            vertical_spacing_lambda=cfg.vertical_spacing_lambda,
        )

        metadata_path = output_dir / "rf_camera_metadata.json"
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "mode": "1bs_1ue_rf_camera_mvp",
            "config": asdict(cfg),
            "frequency_offsets_hz": frequency_offsets_hz.tolist(),
            "absolute_frequencies_hz": (cfg.carrier_frequency_hz + frequency_offsets_hz).tolist(),
            "sionna_cfr_axis_order": [
                "rx",
                "rx_ant",
                "tx",
                "tx_ant",
                "time",
                "frequency_offset",
            ],
            "sionna_cfr_shape": [1, cfg.rx_rows * cfg.rx_cols, 1, 1, 1, cfg.num_frequency_bins],
            "aperture_axis_order": ["row", "col", "frequency_offset"],
            "aperture_shape": list(aperture_cfr.shape),
            "planar_array_numbering": "column-first, top-left to bottom-right",
            "array_plane": "local y-z",
            "angular_fft_axis_order": [
                "vertical_spatial_frequency",
                "horizontal_spatial_frequency",
                "frequency_offset",
            ],
            "angular_fft_shape": list(angular_cfr.shape),
            "notes": [
                "angular_cfr is a zero-padded 2-D spatial FFT diagnostic, "
                "not yet a calibrated AoA image",
                "Tx uses one active antenna/port for the MVP",
                "ideal coherent Sionna phase is preserved",
                "absolute path delays are requested with normalize_delays=False",
            ],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print("=== RF Camera MVP complete ===")
        for path in (
            aperture_path,
            angular_path,
            path_gt_path,
            power_png,
            phase_png,
            metadata_path,
        ):
            print(f"  {path}")

        return RFCameraArtifacts(
            aperture_cfr=aperture_path,
            angular_cfr=angular_path,
            metadata=metadata_path,
            power_png=power_png,
            phase_png=phase_png,
            path_gt=path_gt_path,
        )


def _render_raw_fft_images(
    angular_slice: np.ndarray,
    *,
    power_png: Path,
    phase_png: Path,
    horizontal_spacing_lambda: float,
    vertical_spacing_lambda: float,
) -> None:
    """Render log-power and phase diagnostics of the uncalibrated FFT image."""
    power = np.abs(angular_slice) ** 2
    peak = float(np.max(power)) if power.size else 0.0
    horizontal, vertical = raw_spatial_frequency_axes(
        fft_rows=angular_slice.shape[0],
        fft_cols=angular_slice.shape[1],
        horizontal_spacing_lambda=horizontal_spacing_lambda,
        vertical_spacing_lambda=vertical_spacing_lambda,
    )
    raw_axes = {
        "extent": image_extent(horizontal, vertical),
        "xlabel": "horizontal spatial coordinate ky/k",
        "ylabel": "vertical spatial coordinate kz/k",
    }
    save_direction_image(
        normalized_power_db(power, max(peak, 1e-30)),
        power_png,
        title="RF camera angular spectrum: normalized power [dB]",
        colorbar_label="dB relative to peak",
        vmin=-60.0,
        vmax=0.0,
        **raw_axes,
    )
    save_direction_image(
        np.angle(angular_slice),
        phase_png,
        title="RF camera angular spectrum: phase [rad]",
        colorbar_label="phase [rad]",
        vmin=-np.pi,
        vmax=np.pi,
        **raw_axes,
    )
