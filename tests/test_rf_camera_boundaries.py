import subprocess
import sys

# RF-camera math and directory post-processing must stay usable without a
# Sionna/Mitsuba scene (CPU-only analysis, lightweight unit tests).
SIONNA_FREE_MODULES = [
    "plateau_rt.domain.rf_camera.imaging",
    "plateau_rt.domain.rf_camera.calibration",
    "plateau_rt.domain.rf_camera.delay",
    "plateau_rt.domain.rf_camera.camera",
    "plateau_rt.application.rf_camera_calibration",
    "plateau_rt.application.rf_camera_delay",
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
