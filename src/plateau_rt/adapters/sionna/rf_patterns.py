"""RF-camera receive element patterns registered with Sionna RT.

``rf_camera_split`` fills the two antenna-pattern slots of a Sionna array
(normally used for dual polarization) with the front (antenna-local
``kx >= 0``) and back (``kx < 0``) hemispheres of a vertically polarized
isotropic element. Sionna fuses the pattern and array axes, so a single
PathSolver call yields both hemispheres, and their sum equals the isotropic
element exactly. Because both slots are used, dual polarization is not
available with this pattern.

Importing this module registers the pattern.
"""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
from sionna.rt.antenna_pattern import (
    AntennaPattern,
    polarization_model_registry,
    polarization_registry,
    register_antenna_pattern,
    v_iso_pattern,
)

HEMISPHERE_SPLIT_PATTERN = "rf_camera_split"


class HemisphereSplitPattern(AntennaPattern):
    """Front / back hemisphere halves of a single-polarized isotropic element."""

    def __init__(self, *, polarization: str = "V", polarization_model: str = "tr38901_2"):
        super().__init__()
        slant_angles = polarization_registry.get(polarization)
        if len(slant_angles) != 1:
            raise ValueError(
                f"{HEMISPHERE_SPLIT_PATTERN} uses both pattern slots; "
                "a single polarization is required"
            )
        apply_polarization = polarization_model_registry.get(polarization_model)
        slant_angle = slant_angles[0]

        def hemisphere(front: bool):
            def pattern(theta, phi):
                # Direction cosine along the antenna boresight (local +x)
                kx = dr.sin(theta) * dr.cos(phi)
                keep = kx >= 0 if front else kx < 0
                c = v_iso_pattern(theta, phi)
                c = mi.Complex2f(dr.select(keep, c.real, 0.0), dr.select(keep, c.imag, 0.0))
                return apply_polarization(c, theta, phi, slant_angle)

            return pattern

        self.patterns = [hemisphere(front=True), hemisphere(front=False)]


register_antenna_pattern(
    HEMISPHERE_SPLIT_PATTERN,
    lambda **kwargs: HemisphereSplitPattern(**kwargs),
)
