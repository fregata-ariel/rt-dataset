"""N-BS / multi-UE RF-camera dataset generation.

This stage turns the validated 1-BS / 1-UE RF-camera pipeline into a
multi-view dataset suitable for later Gaussian-Splatting experiments.

Several base stations illuminate the scene; all of them (and all UE views)
are traced in a single ``PathSolver`` call. The canonical stored observation
is the compact complex receive-aperture CFR with a leading BS axis,
recorded separately for the front and back hemispheres of the UE (one
PathSolver call with the ``rf_camera_split`` element pattern). Per-(view, BS)
angular and delay summaries are developed from the front hemisphere as the
complex amplitude per unit solid angle and can be regenerated without
re-running Sionna RT.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
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
    configure_rf_camera_arrays,
    multi_tx_aperture_cfrs,
    path_ground_truth,
    trace_paths,
)
from plateau_rt.application.provenance import collect_provenance
from plateau_rt.application.rf_dataset_manifest import (
    BUILD_MANIFEST_FILE_NAME,
    ManifestError,
    load_scene_transform,
)
from plateau_rt.application.scene_checks import check_scene_carrier_frequency
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
    channel_gain_reference_payload,
    image_axes_payload,
    to_solid_angle_amplitude,
    view_pose_payload,
)
from plateau_rt.domain.rf_camera.delay import angular_cfr_to_delay, dominant_delay
from plateau_rt.domain.rf_camera.imaging import aperture_to_angular_fft, frequency_offsets
from plateau_rt.domain.rf_camera.paths import (
    PATH_GEOMETRY_GT_FILE_NAME,
    PATH_SCHEMA_FILE_NAME,
    build_path_schema,
)


def _validate_3vector(value: Any, *, name: str) -> None:
    """Require a length-3 sequence of finite real numbers (ValueError otherwise)."""
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a length-3 sequence of finite numbers") from exc
    if len(items) != 3:
        raise ValueError(f"{name} must be a length-3 sequence of finite numbers")
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a length-3 sequence of finite numbers")
        if not math.isfinite(float(item)):
            raise ValueError(f"{name} must be a length-3 sequence of finite numbers")


def _validate_position_list(values: Any, *, name: str) -> None:
    """Require a non-empty sequence of length-3 finite-real vectors."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain at least one 3-element position")
    try:
        entries = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain at least one 3-element position") from exc
    if len(entries) < 1:
        if name == "tx_positions":
            raise ValueError("at least one base station is required")
        raise ValueError(f"{name} must contain at least one 3-element position")
    for entry in entries:
        _validate_3vector(entry, name=name)


@dataclass(frozen=True)
class RFMultiViewConfig:
    """Configuration for the multi-BS / multi-UE dataset milestone."""

    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 100e6
    num_frequency_bins: int = 64

    tx_positions: tuple[tuple[float, float, float], ...] = ((-50.0, -50.0, 30.0),)
    tx_look_at: tuple[float, float, float] = (5.0, 5.0, 5.0)
    tx_look_ats: tuple[tuple[float, float, float], ...] | None = None

    rx_rows: int = 8
    rx_cols: int = 8
    vertical_spacing_lambda: float = 0.5
    horizontal_spacing_lambda: float = 0.5
    tx_pattern: str = "tr38901"
    polarization: str = "V"
    tx_power_dbm: float = 44.0

    fft_rows: int = 128
    fft_cols: int = 128
    phase_floor_db: float = -35.0

    max_depth: int = 5
    synthetic_array: bool = True
    seed: int = 42
    specular_reflection: bool = True
    refraction: bool = True
    diffraction: bool = False
    los_free_trace: bool = False

    def validate(self) -> None:
        if self.carrier_frequency_hz <= 0.0:
            raise ValueError("carrier_frequency_hz must be > 0")
        if self.bandwidth_hz <= 0.0:
            raise ValueError("bandwidth_hz must be > 0")
        if self.num_frequency_bins < 2:
            raise ValueError("num_frequency_bins must be >= 2")
        _validate_position_list(self.tx_positions, name="tx_positions")
        _validate_3vector(self.tx_look_at, name="tx_look_at")
        if self.tx_look_ats is not None:
            _validate_position_list(self.tx_look_ats, name="tx_look_ats")
            if len(self.tx_look_ats) != len(self.tx_positions):
                raise ValueError("tx_look_ats must have one entry per base station")
        if self.rx_rows < 1 or self.rx_cols < 1:
            raise ValueError("rx_rows and rx_cols must be >= 1")
        if self.vertical_spacing_lambda <= 0.0 or self.horizontal_spacing_lambda <= 0.0:
            raise ValueError("antenna spacing must be > 0")
        if self.fft_rows < self.rx_rows or self.fft_cols < self.rx_cols:
            raise ValueError("FFT grid must not be smaller than the receive aperture")
        if (
            isinstance(self.tx_power_dbm, bool)
            or not isinstance(self.tx_power_dbm, (int, float, np.integer, np.floating))
            or not math.isfinite(float(self.tx_power_dbm))
        ):
            raise ValueError("tx_power_dbm must be a finite number")
        for name in ("specular_reflection", "refraction", "diffraction", "los_free_trace"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")

    def resolve_base_stations(
        self,
    ) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
        """Return the resolved ``(bs_id, position, look_at)`` list.

        ``bs_id`` is ``f"bs_{i:03d}"``. When ``tx_look_ats`` is None, every BS
        looks at the shared ``tx_look_at``.
        """
        self.validate()
        stations = []
        for index, position in enumerate(self.tx_positions):
            bs_id = f"bs_{index:03d}"
            look_at = self.tx_look_ats[index] if self.tx_look_ats is not None else self.tx_look_at
            stations.append(
                (
                    bs_id,
                    tuple(float(v) for v in position),
                    tuple(float(v) for v in look_at),
                )
            )
        return stations


def prepare_rf_camera_scene(
    xml_path: Path, config: RFMultiViewConfig
) -> tuple[Any, list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]]:
    """Load the scene with the RF-camera arrays and one transmitter per BS.

    Checks the carrier against the scene materials, sets ``scene.frequency``,
    configures the arrays (``config.tx_pattern``, the hemisphere-split receive
    pattern, ``config.polarization``) and adds ``Transmitter("rf_camera_<bs_id>")``
    in :meth:`RFMultiViewConfig.resolve_base_stations` order. Returns the scene
    and the resolved base stations.
    """
    check_scene_carrier_frequency(xml_path, config.carrier_frequency_hz)
    scene = load_scene(str(xml_path))
    scene.frequency = config.carrier_frequency_hz
    configure_rf_camera_arrays(
        scene,
        rx_rows=config.rx_rows,
        rx_cols=config.rx_cols,
        vertical_spacing_lambda=config.vertical_spacing_lambda,
        horizontal_spacing_lambda=config.horizontal_spacing_lambda,
        tx_pattern=config.tx_pattern,
        rx_pattern=HEMISPHERE_SPLIT_PATTERN,
        polarization=config.polarization,
    )
    base_stations = config.resolve_base_stations()
    for bs_id, position, look_at in base_stations:
        scene.add(
            Transmitter(
                name=f"rf_camera_{bs_id}",
                position=list(position),
                look_at=list(look_at),
                power_dbm=config.tx_power_dbm,
            )
        )
    return scene, base_stations


class RFMultiViewDataset:
    """Generate a compact multi-BS / multi-UE RF-camera dataset."""

    def __init__(
        self,
        xml_path: Path,
        *,
        views: list[RFViewSpec],
        config: RFMultiViewConfig | None = None,
        placement: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        view_placements: Sequence[Mapping[str, Any]] | None = None,
    ):
        if not views:
            raise ValueError("at least one RF view is required")
        self.xml_path = Path(xml_path)
        self.views = list(views)
        self.config = config or RFMultiViewConfig()
        self.config.validate()
        self.placement = dict(placement) if placement is not None else None
        self.provenance: dict[str, Any] | None = (
            dict(provenance) if provenance is not None else None
        )
        if view_placements is None:
            self.view_placements: list[dict[str, Any]] | None = None
        else:
            entries = list(view_placements)
            if len(entries) != len(self.views):
                raise ValueError(
                    f"view_placements has {len(entries)} entries for {len(self.views)} views"
                )
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError(f"view_placements entries must be mappings, got {entry!r}")
            self.view_placements = [dict(entry) for entry in entries]

    def run(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.config
        scene, base_stations = prepare_rf_camera_scene(self.xml_path, cfg)

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
        for bs_id, position, look_at in base_stations:
            print(f"BS {bs_id}={position}, look_at={look_at}")
        print(f"views={len(self.views)}")
        print(
            f"Rx={cfg.rx_rows}x{cfg.rx_cols} {HEMISPHERE_SPLIT_PATTERN}, "
            f"spacing=({cfg.vertical_spacing_lambda}, {cfg.horizontal_spacing_lambda}) lambda"
        )

        # All transmitters and receivers are solved in one PathSolver call.
        paths = trace_paths(
            scene,
            max_depth=cfg.max_depth,
            synthetic_array=cfg.synthetic_array,
            seed=cfg.seed,
            specular_reflection=cfg.specular_reflection,
            refraction=cfg.refraction,
            diffraction=cfg.diffraction,
        )
        frequency_offsets_hz = frequency_offsets(cfg.bandwidth_hz, cfg.num_frequency_bins)
        apertures = multi_tx_aperture_cfrs(
            paths,
            frequency_offsets_hz,
            num_rx=len(self.views),
            num_tx=len(base_stations),
            rx_rows=cfg.rx_rows,
            rx_cols=cfg.rx_cols,
        )
        apertures_los_free: np.ndarray | None = None
        if cfg.los_free_trace:
            paths_los_free = trace_paths(
                scene,
                max_depth=cfg.max_depth,
                synthetic_array=cfg.synthetic_array,
                seed=cfg.seed,
                specular_reflection=cfg.specular_reflection,
                refraction=cfg.refraction,
                diffraction=cfg.diffraction,
                los=False,
            )
            apertures_los_free = multi_tx_aperture_cfrs(
                paths_los_free,
                frequency_offsets_hz,
                num_rx=len(self.views),
                num_tx=len(base_stations),
                rx_rows=cfg.rx_rows,
                rx_cols=cfg.rx_cols,
            )
            print("oracle los=False trace complete (aperture_cfr_los_free)")

        camera_model = build_direction_cosine_camera_model(
            fft_rows=cfg.fft_rows,
            fft_cols=cfg.fft_cols,
            horizontal_spacing_lambda=cfg.horizontal_spacing_lambda,
            vertical_spacing_lambda=cfg.vertical_spacing_lambda,
        )
        camera_model_path = output_dir / "camera_model.npz"
        np.savez_compressed(camera_model_path, **camera_model)

        path_gt_path = output_dir / PATH_GEOMETRY_GT_FILE_NAME
        path_gt = path_ground_truth(paths, scene, rx_rows=cfg.rx_rows, rx_cols=cfg.rx_cols)
        np.savez_compressed(path_gt_path, **path_gt.arrays)

        path_schema_path = output_dir / PATH_SCHEMA_FILE_NAME
        path_schema_path.write_text(
            json.dumps(
                build_path_schema(
                    path_gt.arrays,
                    mode=path_gt.mode,
                    object_names=path_gt.object_names,
                    carrier_frequency_hz=cfg.carrier_frequency_hz,
                    bs_ids=[bs_id for bs_id, _, _ in base_stations],
                    view_ids=[view.view_id for view in self.views],
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        unambiguous_delay_s = cfg.num_frequency_bins / cfg.bandwidth_hz
        _warn_if_delay_aliased(paths, unambiguous_delay_s)

        manifest_views: list[dict[str, Any]] = []
        for view_index, (view, aperture_cfr) in enumerate(zip(self.views, apertures)):
            los_free = None if apertures_los_free is None else apertures_los_free[view_index]
            manifest_views.append(
                self._write_view(
                    output_dir,
                    view,
                    aperture_cfr,
                    base_stations=base_stations,
                    frequency_offsets_hz=frequency_offsets_hz,
                    valid_ray_mask=camera_model["valid_mask"],
                    placement=(
                        None if self.view_placements is None else self.view_placements[view_index]
                    ),
                    aperture_cfr_los_free=los_free,
                )
            )
            print(
                f"  [{view_index + 1:02d}/{len(self.views):02d}] {view.view_id}: "
                f"aperture={aperture_cfr.shape}, center={(cfg.fft_rows, cfg.fft_cols)}"
            )

        manifest = {
            "schema_version": 3,
            "mode": "multibs_multiue_rf_camera_dataset",
            "provenance": (
                self.provenance if self.provenance is not None else collect_provenance()
            ),
            "source_scene": str(self.xml_path),
            "config": {
                **asdict(cfg),
                "channel_gain_reference": channel_gain_reference_payload(),
            },
            "frequency_offsets_hz": frequency_offsets_hz.tolist(),
            "absolute_frequencies_hz": (cfg.carrier_frequency_hz + frequency_offsets_hz).tolist(),
            "delay_resolution_s": 1.0 / cfg.bandwidth_hz,
            "unambiguous_delay_s": unambiguous_delay_s,
            "base_stations": [
                {
                    "bs_id": bs_id,
                    "index": index,
                    "position_m": list(position),
                    "look_at_m": list(look_at),
                }
                for index, (bs_id, position, look_at) in enumerate(base_stations)
            ],
            "raw_observation": {
                "artifact": "aperture_cfr",
                "axis_order": ["bs", "hemisphere", "row", "col", "frequency_offset"],
                "bs_ids": [bs_id for bs_id, _, _ in base_stations],
                "hemispheres": list(HEMISPHERES),
                "rx_element_pattern": HEMISPHERE_SPLIT_PATTERN,
                "note": (
                    "Front (local kx >= 0) and back (kx < 0) arrivals of a vertically "
                    "polarized isotropic element; front + back equals the isotropic "
                    "element. A finite front-to-back ratio g can be synthesized as "
                    "front + g * back."
                ),
                **(
                    {"oracle_los_free_artifact": "aperture_cfr_los_free"}
                    if cfg.los_free_trace
                    else {}
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
                "image_axes": image_axes_payload(),
            },
            "path_geometry_gt": PATH_GEOMETRY_GT_FILE_NAME,
            "path_schema": PATH_SCHEMA_FILE_NAME,
            "views": manifest_views,
        }
        # Optional sections keep their documented positions in the manifest.
        scene_transform = self._build_manifest_scene_transform()
        ordered: dict[str, Any] = {}
        for key, value in manifest.items():
            ordered[key] = value
            if key == "source_scene" and scene_transform is not None:
                ordered["scene_transform"] = scene_transform
            if key == "base_stations" and self.placement is not None:
                ordered["placement"] = self.placement
        manifest = ordered
        manifest_path = output_dir / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print("=== RF Camera multi-view dataset complete ===")
        print(f"manifest: {manifest_path}")
        print(f"camera model: {camera_model_path}")
        print(f"path geometry GT: {path_gt_path}")
        return manifest_path

    def _build_manifest_scene_transform(self) -> dict[str, Any] | None:
        """Return the build manifest's scene-transform payload, if usable.

        Reads ``manifest.json`` next to the source scene and returns its
        non-legacy transform payload. Prints one line and returns None when
        the build manifest is missing, has no transform, is legacy-only or
        is malformed (the run continues without ``scene_transform``).
        """
        build_manifest = self.xml_path.parent / BUILD_MANIFEST_FILE_NAME
        if not build_manifest.is_file():
            print(f"scene_transform: no build manifest at {build_manifest}, omitting")
            return None
        try:
            transform = load_scene_transform(build_manifest)
        except ManifestError as exc:
            print(f"scene_transform: unreadable build manifest {build_manifest}: {exc}")
            return None
        if transform is None:
            print(f"scene_transform: no transform in build manifest {build_manifest}, omitting")
            return None
        if transform.legacy:
            print(f"scene_transform: build manifest {build_manifest} is legacy-only, omitting")
            return None
        return transform.to_payload()

    def _write_view(
        self,
        output_dir: Path,
        view: RFViewSpec,
        aperture_cfr: np.ndarray,
        *,
        base_stations: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]],
        frequency_offsets_hz: np.ndarray,
        valid_ray_mask: np.ndarray,
        placement: Mapping[str, Any] | None = None,
        aperture_cfr_los_free: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Save one view's pose, canonical aperture CFR and per-BS summaries.

        ``aperture_cfr`` is ``[bs, hemisphere, row, col, freq]``; per-BS
        summaries are developed from each BS's front hemisphere. Returns the
        view's manifest entry.
        """
        view_dir = output_dir / "views" / view.view_id
        rf_dir = view_dir / "rf"
        rf_dir.mkdir(parents=True, exist_ok=True)
        pose_path = view_dir / "pose.json"
        aperture_path = rf_dir / "aperture_cfr.npy"

        pose_path.write_text(
            json.dumps(view_pose_payload(view), indent=2),
            encoding="utf-8",
        )
        np.save(aperture_path, aperture_cfr.astype(np.complex64, copy=False))
        artifacts: dict[str, str] = {
            "pose": str(pose_path.relative_to(output_dir)),
            "aperture_cfr": str(aperture_path.relative_to(output_dir)),
        }
        if aperture_cfr_los_free is not None:
            los_free_path = rf_dir / "aperture_cfr_los_free.npy"
            np.save(los_free_path, np.asarray(aperture_cfr_los_free).astype(np.complex64))
            artifacts["aperture_cfr_los_free"] = str(los_free_path.relative_to(output_dir))

        bs_entries = []
        for bs_index, (bs_id, bs_position, _look_at) in enumerate(base_stations):
            bs_entries.append(
                self._write_view_bs(
                    output_dir,
                    view,
                    aperture_cfr[bs_index],
                    bs_id=bs_id,
                    bs_position=bs_position,
                    frequency_offsets_hz=frequency_offsets_hz,
                    valid_ray_mask=valid_ray_mask,
                )
            )
        return {
            "view_id": view.view_id,
            "position_m": list(view.position),
            "look_at_m": list(view.look_at),
            "orientation_rad": list(view.orientation),
            **({"placement": dict(placement)} if placement is not None else {}),
            "artifacts": artifacts,
            "bs": bs_entries,
        }

    def _write_view_bs(
        self,
        output_dir: Path,
        view: RFViewSpec,
        aperture_cfr_bs: np.ndarray,
        *,
        bs_id: str,
        bs_position: tuple[float, float, float],
        frequency_offsets_hz: np.ndarray,
        valid_ray_mask: np.ndarray,
    ) -> dict[str, Any]:
        """Save one (view, BS) slice's derived summaries.

        ``aperture_cfr_bs`` is ``[hemisphere, row, col, freq]``; the developed
        summaries use the front hemisphere. Returns the per-BS manifest entry.
        """
        cfg = self.config
        bs_dir = output_dir / "views" / view.view_id / "rf" / bs_id
        bs_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "angular_cfr_center": bs_dir / "angular_cfr_center.npy",
            "angular_power_center": bs_dir / "angular_power_center.npy",
            "phase_valid_mask": bs_dir / "phase_valid_mask.npy",
            "dominant_delay_s": bs_dir / "dominant_delay_s.npy",
            "dominant_delay_power": bs_dir / "dominant_delay_power.npy",
            "debug_power_png": bs_dir / "angular_power_center.png",
        }

        front = aperture_cfr_bs[HEMISPHERES.index("front")]
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
        observed_mask = np.asarray(valid_ray_mask, dtype=bool) & (np.asarray(dominant_power) > 0.0)
        dominant_delay_s = dominant_delay_s.astype(np.float32)
        dominant_delay_s[~observed_mask] = np.nan
        dominant_power = dominant_power.astype(np.float32)
        dominant_power[~observed_mask] = 0.0
        np.save(artifacts["dominant_delay_s"], dominant_delay_s)
        np.save(artifacts["dominant_delay_power"], dominant_power)

        save_direction_image(
            np.ma.masked_where(~valid_ray_mask, normalized_power_db(center_power, view_peak)),
            artifacts["debug_power_png"],
            extent=image_extent(calibration.ky_over_k, calibration.kz_over_k),
            title=f"RF camera {bs_id} front hemisphere |A|^2, center freq [dB rel. view peak]",
            colorbar_label="dB",
            vmin=-60.0,
            vmax=0.0,
            xlabel="UE-local ky/k",
            ylabel="UE-local kz/k",
            figsize=(7, 6),
            dpi=120,
        )

        bs_local = geometric_los_source_direction_local(
            tx_position=bs_position,
            ue_position=view.position,
            ue_orientation=view.orientation,
        )
        energy = {
            name: float(np.sum(np.abs(aperture_cfr_bs[index]) ** 2))
            for index, name in enumerate(HEMISPHERES)
        }
        return {
            "bs_id": bs_id,
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
