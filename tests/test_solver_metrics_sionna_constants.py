"""Pin solver_metrics interaction flags to sionna.rt.constants.InteractionType."""

import pytest

from plateau_rt.domain.rf_camera import solver_metrics


def test_interaction_flags_match_sionna_constants():
    constants = pytest.importorskip("sionna.rt.constants")
    interaction_type = constants.InteractionType
    assert getattr(interaction_type, "NONE") == 0
    assert solver_metrics.INTERACTION_SPECULAR == getattr(interaction_type, "SPECULAR")
    assert solver_metrics.INTERACTION_DIFFUSE == getattr(interaction_type, "DIFFUSE")
    assert solver_metrics.INTERACTION_REFRACTION == getattr(interaction_type, "REFRACTION")
    assert solver_metrics.INTERACTION_DIFFRACTION == getattr(interaction_type, "DIFFRACTION")
