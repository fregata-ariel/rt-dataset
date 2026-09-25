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
