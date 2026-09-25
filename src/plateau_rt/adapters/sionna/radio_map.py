"""Sionna radio-map adapter for coverage-map UE placement (#16).

Computes a 2D path-gain map at UE height with Sionna's ``RadioMapSolver`` and
derives a building-interior mask with an upward ray test. The pure-NumPy
placement logic lives in :mod:`plateau_rt.domain.rf_camera.placement`; this
module only drives Sionna and Mitsuba.

GPU tracing is not bit-reproducible run to run (measured ~6e-7 relative jitter
in ``path_gain``), so callers save the returned arrays as a content-addressed
artifact and reuse them through
:mod:`plateau_rt.application.ue_placement` when poses must be reproduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mitsuba as mi
import numpy as np
from sionna.rt import RadioMapSolver, Transmitter, load_scene

from plateau_rt.adapters.sionna.rf_camera_dataset import RFMultiViewConfig
from plateau_rt.adapters.sionna.rf_patterns import HEMISPHERE_SPLIT_PATTERN
from plateau_rt.adapters.sionna.rf_tracing import configure_rf_camera_arrays
from plateau_rt.application.scene_checks import check_scene_carrier_frequency
from plateau_rt.domain.rf_camera.placement import RadioMapGrid


@dataclass(frozen=True)
class RadioMapSolverSettings:
    """Settings of one ``sionna.rt.RadioMapSolver`` call."""

    max_depth: int = 5
    samples_per_tx: int = 1_000_000
    seed: int = 42
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = True
    diffraction: bool = False

    def validate(self) -> None:
        """Check the solver settings ranges (ValueError otherwise)."""
        if isinstance(self.max_depth, bool) or int(self.max_depth) < 0:
            raise ValueError(f"max_depth must be >= 0, got {self.max_depth!r}")
        if isinstance(self.samples_per_tx, bool) or int(self.samples_per_tx) < 1:
            raise ValueError(f"samples_per_tx must be >= 1, got {self.samples_per_tx!r}")
        if isinstance(self.seed, bool) or int(self.seed) < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable description of these settings."""
        record = asdict(self)
        record["solver"] = "sionna.rt.RadioMapSolver"
        return record


@dataclass(frozen=True)
class RadioMapResult:
    """One computed radio map: path gain, indoor mask and its grid."""

    path_gain: np.ndarray
    indoor_mask: np.ndarray
    grid: RadioMapGrid


def _indoor_mask_from_upward_rays(scene: Any, grid: RadioMapGrid) -> np.ndarray:
    """Return True for cell centres whose upward ray hits scene geometry.

    Cells inside a building still receive a nonzero path gain (refraction
    through walls), so the path-gain map alone cannot exclude building
    interiors. Casting a ray straight up from every cell centre and testing for
    an intersection works for any footprint shape (PLATEAU too); the ground
    plane sits below the UE plane and is never hit.
    """
    points = np.asarray(grid.cell_centers(), dtype=np.float64).reshape(-1, 3)
    ray = mi.Ray3f(
        mi.Point3f(points[:, 0], points[:, 1], points[:, 2]),
        mi.Vector3f(0.0, 0.0, 1.0),
    )
    surface = scene.mi_scene.ray_intersect(ray)
    hit = np.asarray(surface.is_valid().numpy(), dtype=bool)
    return hit.reshape(grid.shape)


def compute_radio_map(
    xml_path: Path,
    *,
    dataset_config: RFMultiViewConfig,
    grid: RadioMapGrid,
    solver: RadioMapSolverSettings,
) -> RadioMapResult:
    """Compute a path-gain map at UE height and the building-interior mask.

    The scene, arrays and transmitters are configured exactly like
    :meth:`RFMultiViewDataset.run`, so the transmitter order matches
    :meth:`RFMultiViewConfig.resolve_base_stations`.
    """
    dataset_config.validate()
    grid.validate()
    solver.validate()
    check_scene_carrier_frequency(xml_path, dataset_config.carrier_frequency_hz)

    cfg = dataset_config
    scene = load_scene(str(xml_path))
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

    base_stations = cfg.resolve_base_stations()
    for bs_id, position, look_at in base_stations:
        scene.add(
            Transmitter(
                name=f"rf_camera_{bs_id}",
                position=list(position),
                look_at=list(look_at),
            )
        )

    radio_map = RadioMapSolver()(
        scene,
        center=list(grid.center_m),
        orientation=[0.0, 0.0, 0.0],
        size=list(grid.size_m),
        cell_size=list(grid.cell_size_m),
        max_depth=int(solver.max_depth),
        samples_per_tx=int(solver.samples_per_tx),
        los=bool(solver.los),
        specular_reflection=bool(solver.specular_reflection),
        diffuse_reflection=bool(solver.diffuse_reflection),
        refraction=bool(solver.refraction),
        diffraction=bool(solver.diffraction),
        seed=int(solver.seed),
    )

    path_gain = np.asarray(radio_map.path_gain.numpy(), dtype=np.float32)
    expected_shape = (len(base_stations), *grid.shape)
    if path_gain.shape != expected_shape:
        raise RuntimeError(
            f"RadioMapSolver returned path_gain shape {path_gain.shape}, expected {expected_shape}"
        )
    cell_centers = np.asarray(radio_map.cell_centers.numpy(), dtype=np.float64)
    if not np.allclose(cell_centers, grid.cell_centers(), atol=1e-3):
        raise RuntimeError(
            "RadioMapSolver cell centres do not match RadioMapGrid.cell_centers() "
            f"(max deviation {float(np.max(np.abs(cell_centers - grid.cell_centers()))):.3e})"
        )

    indoor_mask = _indoor_mask_from_upward_rays(scene, grid)

    print("=== Radio map ===")
    print(f"scene={xml_path}")
    print(f"grid shape={grid.shape} cell_size_m={grid.cell_size_m}")
    for index, (bs_id, position, _look_at) in enumerate(base_stations):
        gain = path_gain[index]
        peak = float(np.max(gain))
        peak_db = 10.0 * np.log10(peak) if peak > 0.0 else float("-inf")
        print(f"BS {bs_id} at {position}: max path gain {peak_db:.2f} dB")
    print(f"indoor cells={int(np.sum(indoor_mask))}/{indoor_mask.size}")

    return RadioMapResult(path_gain=path_gain, indoor_mask=indoor_mask, grid=grid)
