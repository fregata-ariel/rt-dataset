import pkgutil
import subprocess
import sys
from pathlib import Path

import plateau_rt.domain.rf_tomography as rf_tomography

# RF-camera math and directory post-processing must stay usable without a
# Sionna/Mitsuba scene (CPU-only analysis, lightweight unit tests).
SIONNA_FREE_MODULES = [
    "plateau_rt.domain.scene_transform",
    "plateau_rt.application.provenance",
    "plateau_rt.domain.rf_camera.imaging",
    "plateau_rt.domain.rf_camera.calibration",
    "plateau_rt.domain.rf_camera.delay",
    "plateau_rt.domain.rf_camera.gauge",
    "plateau_rt.domain.rf_camera.camera",
    "plateau_rt.domain.rf_camera.optical",
    "plateau_rt.domain.rf_camera.paths",
    "plateau_rt.domain.rf_camera.solver_metrics",
    "plateau_rt.domain.rf_camera.impairments",
    "plateau_rt.domain.rf_camera.image_sources",
    "plateau_rt.domain.ground",
    "plateau_rt.domain.rf_camera.partial",
    "plateau_rt.domain.rf_camera.placement",
    "plateau_rt.application.rf_camera_partial",
    "plateau_rt.application.rf_camera_calibration",
    "plateau_rt.application.rf_camera_delay",
    "plateau_rt.application.rf_camera_observe",
    "plateau_rt.application.rf_dataset_manifest",
    "plateau_rt.application.ue_placement",
    "plateau_rt.application.rf_tomography_io",
    "plateau_rt.application.rf_tomography_profile",
    "plateau_rt.application.rf_tomography_benchmark",
    "plateau_rt.application.rf_tomography_gt",
    "plateau_rt.application.rf_tomography_resynthesis",
    "plateau_rt.application.optical_reference",
    "plateau_rt.application.scene_checks",
    "plateau_rt.application.solver_profile_report",
    "plateau_rt.experimental.rf_scatterer_fit",
    "plateau_rt.experimental.rf_scatterer_study",
    "plateau_rt.experimental.compare_direct_path",
    # Every tomography module, subpackages included (docs/tomography_baselines.md §7.3).
    *(
        info.name
        for info in pkgutil.walk_packages(
            [str(Path(rf_tomography.__file__).parent)], prefix=f"{rf_tomography.__name__}."
        )
    ),
]


def test_rf_camera_domain_and_postprocessing_do_not_import_sionna():
    code = (
        "import importlib, sys\n"
        f"for name in {SIONNA_FREE_MODULES!r}:\n"
        "    importlib.import_module(name)\n"
        "loaded = sorted(m for m in ('sionna', 'mitsuba', 'drjit') if m in sys.modules)\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
