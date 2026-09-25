import subprocess
import sys

# RF-camera math and directory post-processing must stay usable without a
# Sionna/Mitsuba scene (CPU-only analysis, lightweight unit tests).
SIONNA_FREE_MODULES = [
    "plateau_rt.domain.rf_camera.imaging",
    "plateau_rt.domain.rf_camera.calibration",
    "plateau_rt.domain.rf_camera.delay",
    "plateau_rt.domain.rf_camera.develop",
    "plateau_rt.domain.rf_camera.gauge",
    "plateau_rt.domain.rf_camera.camera",
    "plateau_rt.domain.rf_camera.optical",
    "plateau_rt.domain.rf_camera.paths",
    "plateau_rt.domain.rf_camera.solver_metrics",
    "plateau_rt.domain.rf_camera.impairments",
    "plateau_rt.domain.rf_camera.image_sources",
    "plateau_rt.domain.ground",
    "plateau_rt.domain.rf_camera.partial",
    "plateau_rt.application.rf_camera_partial",
    "plateau_rt.application.rf_camera_calibration",
    "plateau_rt.application.rf_camera_delay",
    "plateau_rt.application.rf_camera_develop",
    "plateau_rt.application.rf_camera_observe",
    "plateau_rt.application.rf_dataset_manifest",
    "plateau_rt.application.scene_files",
    "plateau_rt.application.optical_reference",
    "plateau_rt.application.scene_checks",
    "plateau_rt.application.solver_profile_report",
    "plateau_rt.experimental.rf_scatterer_fit",
    "plateau_rt.experimental.rf_scatterer_study",
    "plateau_rt.experimental.compare_direct_path",
    "plateau_rt.viewer",
    "plateau_rt.viewer.settings",
    "plateau_rt.viewer.extract",
    "plateau_rt.viewer.store",
    "plateau_rt.viewer.derive",
    "plateau_rt.viewer.safeio",
    "plateau_rt.viewer.kinds",
    "plateau_rt.viewer.derive.overview",
    "plateau_rt.viewer.testing",
    "plateau_rt.viewer.ingest",
    "plateau_rt.viewer.api",
    "plateau_rt.viewer.api.app",
    "plateau_rt.viewer.api.errors",
    "plateau_rt.viewer.api.routes_bundles",
    "plateau_rt.viewer.api.routes_derived",
    "plateau_rt.viewer.jobs",
    "plateau_rt.viewer.__main__",
    "plateau_rt.viewer.api.routes_jobs",
    "plateau_rt.viewer.api.static_assets",
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
