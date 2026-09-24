"""Regenerate ``sionna_los_aperture.npz``: a real Sionna LoS aperture CFR.

An empty scene with one isotropic V-pol BS and two rolled UE apertures (one sees
the BS in its front hemisphere, one in its back hemisphere), traced with the
RF-camera array configuration (8x8 at lambda/2, ``rf_camera_split``,
``synthetic_array=True``). The tomography geometry tests check the element
offset map, local directions and delays against it without importing Sionna.

Run in the ci container (CPU is fine)::

    python tests/fixtures/rf_tomography/make_sionna_los_aperture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT
from sionna.rt import PlanarArray, Receiver, Transmitter, load_scene

import plateau_rt.adapters.sionna.rf_patterns  # noqa: F401  (registers rf_camera_split)
from plateau_rt.adapters.sionna.rf_tracing import (
    aperture_cfrs,
    configure_rf_camera_arrays,
    trace_paths,
)
from plateau_rt.domain.rf_camera.imaging import frequency_offsets

OUT = Path(__file__).with_name("sionna_los_aperture.npz")
F_C = 3.5e9
BANDWIDTH = 100e6
NUM_BINS = 8
BS_POS = np.array([12.0, -7.0, 9.0])
UE_POS = np.array([[1.0, 2.0, 1.5], [20.0, 5.0, 1.5]])
# Sionna (z, y, x) Euler angles with non-zero roll; UE 0 faces the BS, UE 1 faces away.
UE_ORI = np.array([[0.7, -0.3, 0.45], [0.9, 0.2, -0.6]])


def main() -> None:
    wavelength = SPEED_OF_LIGHT / F_C
    positions = np.asarray(
        PlanarArray(
            num_rows=8,
            num_cols=8,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        ).positions(wavelength),
        dtype=np.float64,
    )
    if positions.shape == (3, 64):
        positions = positions.T

    scene = load_scene()
    scene.frequency = F_C
    configure_rf_camera_arrays(
        scene,
        rx_rows=8,
        rx_cols=8,
        vertical_spacing_lambda=0.5,
        horizontal_spacing_lambda=0.5,
        tx_pattern="iso",
        rx_pattern="rf_camera_split",
    )
    scene.add(Transmitter("tx", position=BS_POS.tolist()))
    for index, (pos, ori) in enumerate(zip(UE_POS, UE_ORI, strict=True)):
        scene.add(Receiver(f"rx{index}", position=pos.tolist(), orientation=ori.tolist()))
    paths = trace_paths(scene, max_depth=0, synthetic_array=True, seed=1)
    offsets = frequency_offsets(BANDWIDTH, NUM_BINS)
    cfr = aperture_cfrs(paths, offsets, num_rx=len(UE_POS), rx_rows=8, rx_cols=8)

    np.savez_compressed(
        OUT,
        aperture_cfr=cfr.astype(np.complex128),  # [ue, hemisphere, row, col, freq]
        sionna_positions=positions,  # [64, 3], Sionna's flat (column-first) antenna order
        bs_pos=BS_POS,
        ue_pos=UE_POS,
        ue_orientation=UE_ORI,
        freq_offsets=offsets.astype(np.float64),
        f_c=np.float64(F_C),
    )
    print(f"wrote {OUT}: aperture_cfr {cfr.shape}")


if __name__ == "__main__":
    main()
