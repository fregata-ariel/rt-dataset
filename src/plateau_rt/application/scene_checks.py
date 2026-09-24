"""Fail-fast carrier check for scenes containing the ITU ground material.

NumPy/Sionna-free: only :mod:`xml.etree` and the NumPy-only
:mod:`plateau_rt.domain.ground` helpers are used.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from plateau_rt.domain.ground import GROUND_ITU_MATERIAL, check_ground_carrier_frequency


def check_scene_carrier_frequency(xml_path: Path, carrier_frequency_hz: float) -> None:
    """Raise if the scene uses the ground material at an unsupported carrier.

    Args:
        xml_path: Path to the Mitsuba scene XML.
        carrier_frequency_hz: Carrier frequency in Hz.

    Raises:
        ValueError: If any ``<bsdf>`` has id ``mat-{GROUND_ITU_MATERIAL}``
            and the carrier lies outside its valid range.
    """
    root = ET.parse(xml_path).getroot()
    ground_bsdf_id = f"mat-{GROUND_ITU_MATERIAL}"
    for bsdf in root.iter("bsdf"):
        if bsdf.get("id") == ground_bsdf_id:
            check_ground_carrier_frequency(carrier_frequency_hz)
            return
