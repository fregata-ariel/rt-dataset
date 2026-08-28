"""1 BS / multi-UE RF-camera dataset generation.

This stage turns the validated 1-BS / 1-UE RF-camera pipeline into a
multi-view dataset suitable for later Gaussian-Splatting experiments.

The canonical stored observation remains the compact complex receive-aperture
CFR. Per-view angular and delay summaries are derived from it and can be
regenerated without re-running Sionna RT.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

from plateau_rt.adapters.sionna.rf_camera import (
    _frequency_offsets,
    _reshape_planar_column_first,
    aperture_to_angular_fft,
)
from plateau_rt.adapters.sionna.rf_camera_calibration import (
    calibrate_angular_cfr,
    rotation_matrix_numpy,
)
from plateau_rt.adapters.sionna.rf_camera_delay import (
    angular_cfr_to_delay,
    direction_axes_from_metadata,
    propagating_direction_mask,
)


@dataclass(frozen=True)
class RFViewSpec:
    """One RF-camera pose."""

    view_id: str
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    orientation: tuple[float, float, float]


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
    rx_pattern: str = "tr38901"
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


def look_at_orientation(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return Sionna Euler angles whose local +x points at ``target``."""

    p = np.asarray(position, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    direction = t - p
    distance = float(np.linalg.norm(direction))
    if distance == 0.0:
        raise ValueError("position and look-at target must differ")
    direction /= distance

    theta = float(np.arccos(np.clip(direction[2], -1.0, 1.0)))
    phi = float(np.arctan2(direction[1], direction[0]))
    return (phi, theta - np.pi / 2.0, 0.0)


def generate_ring_views(
    *,
    target: tuple[float, float, float],
    radius_m: float,
    ue_height_m: float,
    num_views: int,
    start_azimuth_deg: float = 0.0,
) -> list[RFViewSpec]:
    """Generate deterministic RF-camera poses around a target."""

    if radius_m <= 0.0:
        raise ValueError("radius_m must be > 0")
    if num_views < 1:
        raise ValueError("num_views must be >= 1")

    target_arr = np.asarray(target, dtype=np.float64)
    start = np.deg2rad(start_azimuth_deg)
    views: list[RFViewSpec] = []
    for index in range(num_views):
        azimuth = start + 2.0 * np.pi * index / num_views
        position = (
            float(target_arr[0] + radius_m * np.cos(azimuth)),
            float(target_arr[1] + radius_m * np.sin(azimuth)),
            float(ue_height_m),
        )
        orientation = look_at_orientation(position, target)
        views.append(
            RFViewSpec(
                view_id=f"ue_{index:06d}",
                position=position,
                look_at=tuple(float(v) for v in target),
                orientation=orientation,
            )
        )
    return views


def build_direction_cosine_camera_model(
    *,
    fft_rows: int,
    fft_cols: int,
    horizontal_spacing_lambda: float,
    vertical_spacing_lambda: float,
) -> dict[str, np.ndarray]:
    """Build front-hemisphere local rays for the direction-cosine image."""

    ky, kz = direction_axes_from_metadata(
        fft_rows=fft_rows,
        fft_cols=fft_cols,
        horizontal_spacing_lambda=horizontal_spacing_lambda,
        vertical_spacing_lambda=vertical_spacing_lambda,
    )
    valid = propagating_direction_mask(ky, kz)

    ky_grid = np.broadcast_to(ky[None, :], (fft_rows, fft_cols))
    kz_grid = np.broadcast_to(kz[:, None], (fft_rows, fft_cols))
    kx_sq = 1.0 - ky_grid**2 - kz_grid**2
    kx = np.sqrt(np.maximum(kx_sq, 0.0))

    rays = np.stack([kx, ky_grid, kz_grid], axis=-1).astype(np.float32)
    rays[~valid] = 0.0

    return {
        "ray_directions_local": rays,
        "valid_mask": valid.astype(bool),
        "ky_over_k": ky.astype(np.float32),
        "kz_over_k": kz.astype(np.float32),
    }


def _view_pose_payload(view: RFViewSpec) -> dict[str, Any]:
    rotation = rotation_matrix_numpy(view.orientation)
    return {
        "view_id": view.view_id,
        "position_m": list(view.position),
        "look_at_m": list(view.look_at),
        "orientation_rad": list(view.orientation),
        "world_from_local_rotation": rotation.tolist(),
        "camera_forward_axis_local": [1.0, 0.0, 0.0],
        "camera_forward_world": rotation[:, 0].tolist(),
        "projection": "front_hemisphere_direction_cosine",
    }


def _save_path_geometry_gt(paths: Any, output_path: Path) -> None:
    payload: dict[str, np.ndarray] = {}
    for name in ("valid", "tau", "theta_t", "phi_t", "theta_r", "phi_r"):
        try:
            payload[name] = np.asarray(getattr(paths, name))
        except Exception as exc:  # pragma: no cover - backend dependent
            print(f"Warning: could not export paths.{name}: {exc}")
    if payload:
        np.savez_compressed(output_path, **payload)


def _render_power_debug(
    power: np.ndarray,
    *,
    ky_over_k: np.ndarray,
    kz_over_k: np.ndarray,
    output_path: Path,
    valid_mask: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    power = np.asarray(power)
    peak = max(float(np.max(power[valid_mask])), 1e-30) if np.any(valid_mask) else 1e-30
    power_db = 10.0 * np.log10(np.maximum(power / peak, 1e-12))
    power_db = np.ma.masked_where(~valid_mask, power_db)

    extent = [
        float(ky_over_k[0]),
        float(ky_over_k[-1]),
        float(kz_over_k[0]),
        float(kz_over_k[-1]),
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(
        power_db,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=-60.0,
        vmax=0.0,
    )
    ax.set_xlabel("UE-local ky/k")
    ax.set_ylabel("UE-local kz/k")
    ax.set_title("RF camera center-frequency power [dB rel. view peak]")
    fig.colorbar(image, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


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
        views_dir = output_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config
        scene = load_scene(str(self.xml_path))
        scene.frequency = cfg.carrier_frequency_hz

        scene.tx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern=cfg.tx_pattern,
            polarization=cfg.polarization,
        )
        scene.rx_array = PlanarArray(
            num_rows=cfg.rx_rows,
            num_cols=cfg.rx_cols,
            vertical_spacing=cfg.vertical_spacing_lambda,
            horizontal_spacing=cfg.horizontal_spacing_lambda,
            pattern=cfg.rx_pattern,
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
            f"Rx={cfg.rx_rows}x{cfg.rx_cols} {cfg.rx_pattern}, "
            f"spacing=({cfg.vertical_spacing_lambda}, {cfg.horizontal_spacing_lambda}) lambda"
        )

        paths = PathSolver()(
            scene=scene,
            max_depth=cfg.max_depth,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            synthetic_array=cfg.synthetic_array,
            seed=cfg.seed,
        )

        frequency_offsets_hz = _frequency_offsets(
            cfg.bandwidth_hz,
            cfg.num_frequency_bins,
        )
        cfr = np.asarray(
            paths.cfr(
                frequencies=frequency_offsets_hz,
                normalize_delays=False,
                normalize=False,
                out_type="numpy",
            )
        )
        expected = (
            len(self.views),
            cfg.rx_rows * cfg.rx_cols,
            1,
            1,
            1,
            cfg.num_frequency_bins,
        )
        print(f"Paths.cfr shape={cfr.shape}, dtype={cfr.dtype}")
        if cfr.shape != expected:
            raise RuntimeError(
                "Unexpected multi-view Paths.cfr shape: "
                f"expected={expected}, actual={cfr.shape}"
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
        _save_path_geometry_gt(paths, path_gt_path)

        unambiguous_delay_s = cfg.num_frequency_bins / cfg.bandwidth_hz
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

        manifest_views: list[dict[str, Any]] = []
        center_bin = cfg.num_frequency_bins // 2
        valid_ray_mask = camera_model["valid_mask"]

        for view_index, view in enumerate(self.views):
            view_dir = views_dir / view.view_id
            rf_dir = view_dir / "rf"
            rf_dir.mkdir(parents=True, exist_ok=True)

            pose_payload = _view_pose_payload(view)
            (view_dir / "pose.json").write_text(
                json.dumps(pose_payload, indent=2),
                encoding="utf-8",
            )

            aperture_flat = cfr[view_index, :, 0, 0, 0, :]
            aperture_cfr = _reshape_planar_column_first(
                aperture_flat,
                rows=cfg.rx_rows,
                cols=cfg.rx_cols,
            )
            np.save(
                rf_dir / "aperture_cfr.npy",
                aperture_cfr.astype(np.complex64, copy=False),
            )

            raw_angular = aperture_to_angular_fft(
                aperture_cfr,
                fft_rows=cfg.fft_rows,
                fft_cols=cfg.fft_cols,
            )
            calibration = calibrate_angular_cfr(
                raw_angular,
                aperture_rows=cfg.rx_rows,
                aperture_cols=cfg.rx_cols,
                horizontal_spacing_lambda=cfg.horizontal_spacing_lambda,
                vertical_spacing_lambda=cfg.vertical_spacing_lambda,
            )

            center_cfr = calibration.cfr[:, :, center_bin]
            center_power = np.abs(center_cfr) ** 2
            np.save(
                rf_dir / "angular_cfr_center.npy",
                center_cfr.astype(np.complex64, copy=False),
            )
            np.save(
                rf_dir / "angular_power_center.npy",
                center_power.astype(np.float32, copy=False),
            )

            view_peak = max(float(np.max(center_power[valid_ray_mask])), 1e-30)
            phase_valid = valid_ray_mask & (
                center_power >= view_peak * 10.0 ** (cfg.phase_floor_db / 10.0)
            )
            np.save(rf_dir / "phase_valid_mask.npy", phase_valid)

            delay_volume = angular_cfr_to_delay(
                calibration.cfr,
                frequency_offsets_hz,
            )
            delay_power = np.abs(delay_volume.cir) ** 2
            dominant_index = np.argmax(delay_power, axis=-1)
            dominant_delay_s = delay_volume.delay_s[dominant_index]
            dominant_power = np.take_along_axis(
                delay_power,
                dominant_index[..., None],
                axis=-1,
            )[..., 0]

            dominant_delay_s = dominant_delay_s.astype(np.float32)
            dominant_delay_s[~valid_ray_mask] = np.nan
            dominant_power = dominant_power.astype(np.float32)
            dominant_power[~valid_ray_mask] = 0.0
            np.save(rf_dir / "dominant_delay_s.npy", dominant_delay_s)
            np.save(rf_dir / "dominant_delay_power.npy", dominant_power)

            debug_png = rf_dir / "angular_power_center.png"
            _render_power_debug(
                center_power,
                ky_over_k=calibration.ky_over_k,
                kz_over_k=calibration.kz_over_k,
                output_path=debug_png,
                valid_mask=valid_ray_mask,
            )

            rotation = rotation_matrix_numpy(view.orientation)
            bs_world = np.asarray(cfg.tx_position, dtype=np.float64) - np.asarray(
                view.position, dtype=np.float64
            )
            bs_world /= np.linalg.norm(bs_world)
            bs_local = rotation.T @ bs_world

            manifest_views.append(
                {
                    "view_id": view.view_id,
                    "position_m": list(view.position),
                    "look_at_m": list(view.look_at),
                    "orientation_rad": list(view.orientation),
                    "bs_direction_local": bs_local.tolist(),
                    "bs_in_front_hemisphere": bool(bs_local[0] >= 0.0),
                    "artifacts": {
                        "pose": str((view_dir / "pose.json").relative_to(output_dir)),
                        "aperture_cfr": str(
                            (rf_dir / "aperture_cfr.npy").relative_to(output_dir)
                        ),
                        "angular_cfr_center": str(
                            (rf_dir / "angular_cfr_center.npy").relative_to(output_dir)
                        ),
                        "angular_power_center": str(
                            (rf_dir / "angular_power_center.npy").relative_to(output_dir)
                        ),
                        "phase_valid_mask": str(
                            (rf_dir / "phase_valid_mask.npy").relative_to(output_dir)
                        ),
                        "dominant_delay_s": str(
                            (rf_dir / "dominant_delay_s.npy").relative_to(output_dir)
                        ),
                        "dominant_delay_power": str(
                            (rf_dir / "dominant_delay_power.npy").relative_to(output_dir)
                        ),
                        "debug_power_png": str(debug_png.relative_to(output_dir)),
                    },
                }
            )
            print(
                f"  [{view_index + 1:02d}/{len(self.views):02d}] {view.view_id}: "
                f"aperture={aperture_cfr.shape}, center={center_cfr.shape}"
            )

        manifest = {
            "schema_version": 1,
            "mode": "1bs_multiue_rf_camera_dataset",
            "source_scene": str(self.xml_path),
            "config": asdict(cfg),
            "frequency_offsets_hz": frequency_offsets_hz.tolist(),
            "absolute_frequencies_hz": (
                cfg.carrier_frequency_hz + frequency_offsets_hz
            ).tolist(),
            "delay_resolution_s": 1.0 / cfg.bandwidth_hz,
            "unambiguous_delay_s": unambiguous_delay_s,
            "camera_model": {
                "projection": "front_hemisphere_direction_cosine",
                "forward_axis_local": "+x",
                "array_plane_local": "y-z",
                "ray_directions": camera_model_path.name,
                "front_back_note": (
                    "The 2-D planar aperture measures only ky/k and kz/k. "
                    "The dataset chooses +kx for camera rays and uses a directional "
                    "tr38901 Rx element pattern to suppress back-hemisphere energy; "
                    "back energy is attenuated, not mathematically eliminated."
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a 1-BS / multi-UE RF-camera dataset"
    )
    parser.add_argument("xml_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--radius-m", type=float, default=30.0)
    parser.add_argument("--ue-height-m", type=float, default=1.5)
    parser.add_argument("--target", type=float, nargs=3, default=(5.0, 5.0, 5.0))
    parser.add_argument(
        "--bs-position",
        type=float,
        nargs=3,
        default=(-50.0, -50.0, 30.0),
    )
    parser.add_argument("--carrier-ghz", type=float, default=3.5)
    parser.add_argument("--bandwidth-mhz", type=float, default=100.0)
    parser.add_argument("--frequency-bins", type=int, default=64)
    parser.add_argument("--rx-rows", type=int, default=8)
    parser.add_argument("--rx-cols", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument(
        "--explicit-array",
        action="store_true",
        help="Trace each antenna element explicitly instead of synthetic-array mode",
    )
    args = parser.parse_args()

    target = tuple(float(v) for v in args.target)
    views = generate_ring_views(
        target=target,
        radius_m=args.radius_m,
        ue_height_m=args.ue_height_m,
        num_views=args.num_views,
    )
    config = RFMultiViewConfig(
        carrier_frequency_hz=args.carrier_ghz * 1e9,
        bandwidth_hz=args.bandwidth_mhz * 1e6,
        num_frequency_bins=args.frequency_bins,
        tx_position=tuple(float(v) for v in args.bs_position),
        tx_look_at=target,
        rx_rows=args.rx_rows,
        rx_cols=args.rx_cols,
        max_depth=args.max_depth,
        synthetic_array=not args.explicit_array,
    )
    RFMultiViewDataset(args.xml_file, views=views, config=config).run(args.output_dir)


if __name__ == "__main__":
    main()
