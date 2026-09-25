"""Web viewer for RF-camera datasets (Sionna-free; see issue #26).

Viewer modules must not import ``sionna``, ``mitsuba``, ``drjit`` or ``matplotlib``, and every
module in this package must be listed in
``tests/test_rf_camera_boundaries.py::SIONNA_FREE_MODULES``.
"""

# Version of the viewer (store metadata, API); bump on releases.
VIEWER_VERSION = "0.1.0"
