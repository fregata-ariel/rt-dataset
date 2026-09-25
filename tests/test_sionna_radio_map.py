"""CPU subprocess test for the Sionna radio-map adapter (#16)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_BUILDING_JSON = REPO_ROOT / "data/raw/mock_building.city.json"


def test_sionna_radio_map_on_mock_building(tmp_path: Path) -> None:
    """Compute a radio map on the mock box and check its shape and indoor mask."""
    code = "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r})",
            "from pathlib import Path",
            "import numpy as np",
            "import sionna.rt",
            "from plateau_rt.adapters.sionna.radio_map import (",
            "    RadioMapSolverSettings, compute_radio_map)",
            "from plateau_rt.adapters.sionna.rf_camera_dataset import RFMultiViewConfig",
            "from plateau_rt.application.build_scene import SceneBuilder",
            "from plateau_rt.domain.rf_camera.placement import RadioMapGrid",
            "out = Path(sys.argv[1])",
            f"xml = SceneBuilder(Path({str(MOCK_BUILDING_JSON)!r}), out).run()",
            "grid = RadioMapGrid(",
            "    center_m=(0.0, 0.0, 1.5), size_m=(42.0, 42.0), cell_size_m=(2.0, 2.0))",
            "config = RFMultiViewConfig(tx_positions=((-30.0, 0.0, 20.0),))",
            "solver = RadioMapSolverSettings(max_depth=1, samples_per_tx=10000, seed=1)",
            "result = compute_radio_map(",
            "    xml, dataset_config=config, grid=grid, solver=solver)",
            "assert result.path_gain.shape == (1, 21, 21), result.path_gain.shape",
            "assert result.path_gain.dtype == np.float32, result.path_gain.dtype",
            "assert bool(np.any(result.path_gain > 0.0))",
            "centers = grid.cell_centers()",
            "# Cell centres sit on even coordinates, so none lies on the box walls",
            "# (|x| = 5 or |y| = 5), where an upward ray would only graze a face.",
            "expected = (np.abs(centers[:, :, 0]) < 5.0) & (np.abs(centers[:, :, 1]) < 5.0)",
            "assert int(expected.sum()) == 25, int(expected.sum())",
            "assert np.array_equal(result.indoor_mask, expected), (",
            "    int(result.indoor_mask.sum()), int(expected.sum()))",
            "# LoS mask: bool [B, ny, nx], compared with an analytic segment-vs-box",
            "# slab test (grazing cells evaluated with the box grown/shrunk 0.05 m).",
            "assert result.los_mask is not None",
            "assert result.los_mask.shape == (1, 21, 21), result.los_mask.shape",
            "assert result.los_mask.dtype == bool, result.los_mask.dtype",
            "bs = np.array([-30.0, 0.0, 20.0])",
            "p0 = centers.reshape(-1, 3)",
            "dirv = bs[None, :] - p0",
            "",
            "def analytic(box_min, box_max):",
            "    lo = box_min[None, :]",
            "    hi = box_max[None, :]",
            "    with np.errstate(divide='ignore', invalid='ignore'):",
            "        inv = 1.0 / dirv",
            "    t1 = (lo - p0) * inv",
            "    t2 = (hi - p0) * inv",
            "    tmin = np.minimum(t1, t2)",
            "    tmax = np.maximum(t1, t2)",
            "    # Axis-parallel rays (delta == 0) never enter a slab the point is outside of.",
            "    outside = (dirv == 0.0) & ((p0 < lo) | (p0 > hi))",
            "    tmin = np.where(np.isnan(tmin), -np.inf, tmin)",
            "    tmax = np.where(np.isnan(tmax), np.inf, tmax)",
            "    enter = np.maximum(tmin.max(axis=1), 0.0)",
            "    exit_ = np.minimum(tmax.min(axis=1), 1.0)",
            "    blocked = (exit_ >= enter) & ~outside.any(axis=1)",
            "    return ~blocked",
            "",
            "box_min = np.array([-5.0, -5.0, 0.0])",
            "box_max = np.array([5.0, 5.0, 10.0])",
            "grown = analytic(box_min - 0.05, box_max + 0.05)",
            "shrunk = analytic(box_min + 0.05, box_max - 0.05)",
            "unambiguous = grown == shrunk",
            "mismatch = np.count_nonzero((result.los_mask[0].reshape(-1) != grown) & unambiguous)",
            "assert mismatch == 0, mismatch",
            "lo_matches = int(np.sum(unambiguous & grown))",
            "nl_matches = int(np.sum(unambiguous & ~grown))",
            "assert lo_matches >= 50, lo_matches",
            "assert nl_matches >= 20, nl_matches",
        ]
    )
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import sionna.rt"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("sionna import probe timed out")
    if probe.returncode != 0:
        pytest.skip("sionna.rt not available")
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True, timeout=300)
