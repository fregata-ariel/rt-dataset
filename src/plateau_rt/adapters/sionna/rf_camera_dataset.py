"""1 BS / multi-UE RF-camera dataset generation.

This stage turns the validated 1-BS / 1-UE RF-camera pipeline into a
multi-view dataset suitable for later Gaussian-Splatting experiments.

The canonical stored observation is the compact complex receive-aperture CFR,
recorded separately for the front and back hemispheres of the UE (one
PathSolver call with the ``rf_camera_split`` element pattern). Per-view
angular and delay summaries are developed from the front hemisphere as the
complex amplitude per unit solid angle and can be regenerated without
re-running Sionna RT.
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
from plateau_rt.adapters.sionna.rf_patterns import HEMISPHERE_SPLIT_PATTERN
from plateau_rt.adapters.sionna.rf_tracing import (
    PATH_ANGLE_FIELDS,
    aperture_cfrs,
    configure_rf_camera_arrays,
    path_attributes,
    trace_paths,
)
from plateau_rt.domain.rf_camera.calibration import (
    calibrate_angular_cfr,
    geometric_los_source_direction_local,
)
from plateau_rt.domain.rf_camera.camera import (
    HEMISPHERES,
    IMAGE_QUANTITY,
    PROJECTION,
    RFViewSpec,
    build_direction_cosine_camera_model,
    to_solid_angle_amplitude,
    view_pose_payload,
)
from plateau_rt.domain.rf_camera.delay import angular_cfr_to_delay, dominant_delay
from plateau_rt.domain.rf_camera.imaging import aperture_to_angular_fft, frequency_offsets


@dataclass(frozen=True)
class RFMultiViewConfig:
    """Configuration for the first 1-BS / multi-UE dataset milestone."""

    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 100e6
    num_frequency_bins: int = 64

    tx_position: tuple[float, float, float] = (-50.0, -50.0, 30.0)
    tx_look_at: tuple[float, float, float] = (5.0, 5.0, 5.0)

    rx_rows: int = 8
    rx_cols: int = 8
    vertical_spacing_lambda: float = 0.5
    horizontal_spacing_lambda: float = 0.5
    tx_pattern: str = "tr38901"
    polarization: str = "V"

    fft_rows: int = 128
    fft_cols: int = 128
    phase_floor_db: float = -35.0

    max_depth: int = 5
    synthetic_array: bool = True
    seed: int = 42

    def validate(self) -> None:
        if self.carrier_frequency_hz <= 0.0:
            raise ValueError("carrier_frequency_hz must be > 0")
        if self.bandwidth_hz <= 0.0:
            raise ValueError("bandwidth_hz must be > 0")
        if self.num_frequency_bins < 2:
            raise ValueError("num_frequency_bins must be >= 2")
        if self.rx_rows < 1 or self.rx_cols < 1:
            raise ValueError("rx_rows and rx_cols must be >= 1")
        if self.vertical_spacing_lambda <= 0.0 or self.horizontal_spacing_lambda <= 0.0:
            raise ValueError("antenna spacing must be > 0")
        if self.fft_rows < self.rx_rows or self.fft_cols < self.rx_cols:
            raise ValueError("FFT grid must not be smaller than the receive aperture")


class RFMultiViewDataset:
    """Generate a compact 1-BS / multi-UE RF-camera dataset."""

    def __init__(
        self,
        xml_path: Path,
        *,
        views: list[RFViewSpec],
        config: RFMultiViewConfig | None = None,
    ):
        if not views:
            raise ValueError("at least one RF view is required")
        self.xml_path = Path(xml_path)
        self.views = list(views)
        self.config = config or RFMultiViewConfig()
        self.config.validate()

    def run(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config
        scene = load_scene(str(self.xml_path))
        scene.frequency = cfg.carrier_frequency_hz
        configure_rf_camera_arrays(
            scene,
            rx_rows=cfg.rx_rows,
            rx_cols=cfg.rx_cols,
            vertical_spacing_lambda=cfg.vertical_spacing_lambda,
            horizontal_spacing_lambda=cfg.horizontal_spacing_lambda,
            tx_pattern=cfg.tx_pattern,
            rx_pattern=HEMISPHERE_SPLIT_PATTERN,
            polarization=cfg.polarization,
        )

        tx = Transmitter(
            name="rf_camera_bs_000",
            position=list(cfg.tx_position),
            look_at=list(cfg.tx_look_at),
        )
        scene.add(tx)

        for view in self.views:
            scene.add(
                Receiver(
                    name=view.view_id,
                    position=list(view.position),
                    orientation=list(view.orientation),
                )
            )

        print("=== RF Camera multi-view dataset: path tracing ===")
        print(f"scene={self.xml_path}")
        print(f"BS={cfg.tx_position}, look_at={cfg.tx_look_at}")
        print(f"views={len(self.views)}")
        print(
            f"Rx={cfg.rx_rows}x{cfg.rx_cols} {HEMISPHERE_SPLIT_PATTERN}, "
            f"spacing=({cfg.vertical_spacing_lambda}, {cfg.horizontal_spacing_lambda}) lambda"
        )

        # All receivers are solved in one PathSolver call.
        paths = trace_paths(
            scene,
            max_depth=cfg.max_depth,
            synthetic_array=cfg.synthetic_array,
            seed=cfg.seed,
        )
        frequency_offsets_hz = frequency_offsets(cfg.bandwidth_hz, cfg.num_frequency_bins)
        apertures = aperture_cfrs(
            paths,
            frequency_offsets_hz,
            num_rx=len(self.views),
            rx_rows=cfg.rx_rows,
            rx_cols=cfg.rx_cols,
        )

        camera_model = build_direction_cosine_camera_model(
            fft_rows=cfg.fft_rows,
            fft_cols=cfg.fft_cols,
            horizontal_spacing_lambda=cfg.horizontal_spacing_lambda,
            vertical_spacing_lambda=cfg.vertical_spacing_lambda,
        )
        camera_model_path = output_dir / "camera_model.npz"
        np.savez_compressed(camera_model_path, **camera_model)

        path_gt_path = output_dir / "path_geometry_gt.npz"
        path_gt = path_attributes(paths, ("valid", "tau") + PATH_ANGLE_FIELDS)
        if path_gt:
            np.savez_compressed(path_gt_path, **path_gt)

        unambiguous_delay_s = cfg.num_frequency_bins / cfg.bandwidth_hz
        _warn_if_delay_aliased(paths, unambiguous_delay_s)

        manifest_views: list[dict[str, Any]] = []
        for view_index, (view, aperture_cfr) in enumerate(zip(self.views, apertures)):
            manifest_views.append(
                self._write_view(
                    output_dir,
                    view,
                    aperture_cfr,
                    frequency_offsets_hz=frequency_offsets_hz,
                    valid_ray_mask=camera_model["valid_mask"],
                )
            )
            print(
                f"  [{view_index + 1:02d}/{len(self.views):02d}] {view.view_id}: "
                f"aperture={aperture_cfr.shape}, center={(cfg.fft_rows, cfg.fft_cols)}"
            )

        manifest = {
            "schema_version": 2,
            "mode": "1bs_multiue_rf_camera_dataset",
            "source_scene": str(self.xml_path),
            "config": asdict(cfg),
            "frequency_offsets_hz": frequency_offsets_hz.tolist(),
            "absolute_frequencies_hz": (cfg.carrier_frequency_hz + frequency_offsets_hz).tolist(),
            "delay_resolution_s": 1.0 / cfg.bandwidth_hz,
            "unambiguous_delay_s": unambiguous_delay_s,
            "raw_observation": {
                "artifact": "aperture_cfr",
                "axis_order": ["hemisphere", "row", "col", "frequency_offset"],
                "hemispheres": list(HEMISPHERES),
                "rx_element_pattern": HEMISPHERE_SPLIT_PATTERN,
                "note": (
                    "Front (local kx >= 0) and back (kx < 0) arrivals of a vertically "
                    "polarized isotropic element; front + back equals the isotropic "
                    "element. A finite front-to-back ratio g can be synthesized as "
                    "front + g * back."
                ),
            },
            "camera_model": {
                "projection": PROJECTION,
                "forward_axis_local": "+x",
                "array_plane_local": "y-z",
                "ray_directions": camera_model_path.name,
                "developed_hemisphere": HEMISPHERES[0],
                "image_quantity": IMAGE_QUANTITY,
                "image_definition": (
                    "A(ky, kz) = kx * U(ky, kz) with kx = sqrt(1 - ky^2 - kz^2): U is the "
                    "calibrated angular spectrum of the front-hemisphere aperture CFR, A the "
                    "complex amplitude per unit solid angle (camera_model.npz "
                    "solid_angle_weight). Back-hemisphere arrivals are excluded, like light "
                    "behind an optical camera."
                ),
            },
            "path_geometry_gt": path_gt_path.name,
            "views": manifest_views,
        }
        manifest_path = output_dir / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print("=== RF Camera multi-view dataset complete ===")
        print(f"manifest: {manifest_path}")
        print(f"camera model: {camera_model_path}")
        print(f"path geometry GT: {path_gt_path}")
        return manifest_path

    def _write_view(
        self,
        output_dir: Path,
        view: RFViewSpec,
        aperture_cfr: np.ndarray,
        *,
        frequency_offsets_hz: np.ndarray,
        valid_ray_mask: np.ndarray,
    ) -> dict[str, Any]:
        """Save one view's pose, canonical aperture CFR and derived summaries.

        ``aperture_cfr`` is ``[hemisphere, row, col, freq]``; the developed
        summaries use the front hemisphere. Returns the view's manifest entry.
        """
        cfg = self.config
        view_dir = output_dir / "views" / view.view_id
        rf_dir = view_dir / "rf"
        rf_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "pose": view_dir / "pose.json",
            "aperture_cfr": rf_dir / "aperture_cfr.npy",
            "angular_cfr_center": rf_dir / "angular_cfr_center.npy",
            "angular_power_center": rf_dir / "angular_power_center.npy",
            "phase_valid_mask": rf_dir / "phase_valid_mask.npy",
            "dominant_delay_s": rf_dir / "dominant_delay_s.npy",
            "dominant_delay_power": rf_dir / "dominant_delay_power.npy",
            "debug_power_png": rf_dir / "angular_power_center.png",
        }

        artifacts["pose"].write_text(
            json.dumps(view_pose_payload(view), indent=2),
            encoding="utf-8",
        )
        np.save(artifacts["aperture_cfr"], aperture_cfr.astype(np.complex64, copy=False))

        front = aperture_cfr[HEMISPHERES.index("front")]
        calibration = calibrate_angular_cfr(
            aperture_to_angular_fft(front, fft_rows=cfg.fft_rows, fft_cols=cfg.fft_cols),
            aperture_rows=cfg.rx_rows,
            aperture_cols=cfg.rx_cols,
            horizontal_spacing_lambda=cfg.horizontal_spacing_lambda,
            vertical_spacing_lambda=cfg.vertical_spacing_lambda,
        )
        image = to_solid_angle_amplitude(
            calibration.cfr, calibration.ky_over_k, calibration.kz_over_k
        )

        center_cfr = image[:, :, cfg.num_frequency_bins // 2]
        center_power = np.abs(center_cfr) ** 2
        np.save(artifacts["angular_cfr_center"], center_cfr.astype(np.complex64, copy=False))
        np.save(artifacts["angular_power_center"], center_power.astype(np.float32, copy=False))

        view_peak = max(float(np.max(center_power[valid_ray_mask])), 1e-30)
        phase_valid = valid_ray_mask & (
            center_power >= view_peak * 10.0 ** (cfg.phase_floor_db / 10.0)
        )
        np.save(artifacts["phase_valid_mask"], phase_valid)

        delay_volume = angular_cfr_to_delay(image, frequency_offsets_hz)
        _, dominant_delay_s, dominant_power = dominant_delay(
            np.abs(delay_volume.cir) ** 2,
            delay_volume.delay_s,
        )
        dominant_delay_s = dominant_delay_s.astype(np.float32)
        dominant_delay_s[~valid_ray_mask] = np.nan
        dominant_power = dominant_power.astype(np.float32)
        dominant_power[~valid_ray_mask] = 0.0
        np.save(artifacts["dominant_delay_s"], dominant_delay_s)
        np.save(artifacts["dominant_delay_power"], dominant_power)

        save_direction_image(
            np.ma.masked_where(~valid_ray_mask, normalized_power_db(center_power, view_peak)),
            artifacts["debug_power_png"],
            extent=image_extent(calibration.ky_over_k, calibration.kz_over_k),
            title="RF camera front hemisphere |A|^2, center frequency [dB rel. view peak]",
            colorbar_label="dB",
            vmin=-60.0,
            vmax=0.0,
            xlabel="UE-local ky/k",
            ylabel="UE-local kz/k",
            figsize=(7, 6),
            dpi=120,
        )

        bs_local = geometric_los_source_direction_local(
            tx_position=cfg.tx_position,
            ue_position=view.position,
            ue_orientation=view.orientation,
        )
        energy = {
            name: float(np.sum(np.abs(aperture_cfr[index]) ** 2))
            for index, name in enumerate(HEMISPHERES)
        }
        return {
            "view_id": view.view_id,
            "position_m": list(view.position),
            "look_at_m": list(view.look_at),
            "orientation_rad": list(view.orientation),
            "bs_direction_local": bs_local.tolist(),
            "bs_in_front_hemisphere": bool(bs_local[0] >= 0.0),
            # Sum of |aperture CFR|^2 over elements and frequencies per hemisphere
            "hemisphere_energy": energy,
            "artifacts": {
                name: str(path.relative_to(output_dir)) for name, path in artifacts.items()
            },
        }


def _warn_if_delay_aliased(paths: Any, unambiguous_delay_s: float) -> None:
    """Warn when a traced path is longer than the CFR's unambiguous delay."""
    try:
        tau = np.asarray(paths.tau)
        valid_tau = tau[tau >= 0.0]
        max_tau = float(np.max(valid_tau)) if valid_tau.size else 0.0
        if max_tau >= unambiguous_delay_s:
            print(
                "WARNING: path delay exceeds the CFR unambiguous delay; "
                f"max path={max_tau * 1e9:.3f} ns, "
                f"unambiguous={unambiguous_delay_s * 1e9:.3f} ns. "
                "Increase num_frequency_bins or reduce bandwidth."
            )
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not check delay aliasing: {exc}")
