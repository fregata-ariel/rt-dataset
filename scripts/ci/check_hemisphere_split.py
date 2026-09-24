"""節目CI: 前面/背面分割パターンが等方性素子を正確に分割していることを確認する。

同じ mock シーン・視点を、等方性素子 (iso) と前面/背面分割パターン
(rf_camera_split) でそれぞれトレースし、front + back == iso を確かめる。
Sionna の「パターン × 素子」の軸の並び (pattern-major) が変わった場合もここで検出する。
トレースは2回に分かれるため、GPU の実行ごとの揺らぎ分だけ許容誤差を持たせる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sionna.rt import Receiver, Transmitter, load_scene

from plateau_rt.adapters.sionna.rf_patterns import HEMISPHERE_SPLIT_PATTERN
from plateau_rt.adapters.sionna.rf_tracing import (
    aperture_cfrs,
    configure_rf_camera_arrays,
    trace_paths,
)
from plateau_rt.domain.rf_camera.camera import generate_ring_views
from plateau_rt.domain.rf_camera.imaging import frequency_offsets

# 構造的な誤り (半球の取り違え・欠落) は O(1) の差になる
MAX_RELATIVE_ERROR = 1e-3
ROWS = COLS = 8
TARGET = (5.0, 5.0, 5.0)
BS_POSITION = (-50.0, -50.0, 30.0)


def trace_apertures(xml: Path, rx_pattern: str) -> np.ndarray:
    views = generate_ring_views(target=TARGET, radius_m=30.0, ue_height_m=1.5, num_views=8)
    scene = load_scene(str(xml))
    scene.frequency = 3.5e9
    configure_rf_camera_arrays(
        scene,
        rx_rows=ROWS,
        rx_cols=COLS,
        vertical_spacing_lambda=0.5,
        horizontal_spacing_lambda=0.5,
        tx_pattern="tr38901",
        rx_pattern=rx_pattern,
    )
    scene.add(Transmitter(name="bs", position=list(BS_POSITION), look_at=list(TARGET)))
    for view in views:
        scene.add(
            Receiver(
                name=view.view_id,
                position=list(view.position),
                orientation=list(view.orientation),
            )
        )
    paths = trace_paths(scene, max_depth=5, synthetic_array=True, seed=42)
    return aperture_cfrs(
        paths, frequency_offsets(100e6, 64), num_rx=len(views), rx_rows=ROWS, rx_cols=COLS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_xml", type=Path, help="mock scene XML built by build-mock")
    args = parser.parse_args()

    iso = trace_apertures(args.scene_xml, "iso")
    split = trace_apertures(args.scene_xml, HEMISPHERE_SPLIT_PATTERN)
    ok = iso.shape[1] == 1 and split.shape[1] == 2
    print(f"[{'OK' if ok else 'NG'}] pattern axis: iso {iso.shape}, split {split.shape}")

    error = float(np.max(np.abs(split.sum(axis=1) - iso[:, 0])) / np.max(np.abs(iso)))
    exact = error <= MAX_RELATIVE_ERROR
    print(f"[{'OK' if exact else 'NG'}] |front + back - iso| / max|iso| = {error:.2e}")
    if not (ok and exact):
        sys.exit(1)
    print("✅ Hemisphere split reproduces the isotropic element")


if __name__ == "__main__":
    main()
