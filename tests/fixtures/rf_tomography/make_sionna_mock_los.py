"""Regenerate ``sionna_mock_los.npz``: real Sionna LoS captures of the mock ring.

The source is the split-pattern multi-view mock dataset (tr38901 V-pol BS aimed
at the target, 8x8 ``rf_camera_split`` UE apertures, synthetic array), traced on
the GPU with::

    make MOCK_OUT=data/generated/tp0_mock/ rf-camera-multiview-mock

Every ring view of that mock receives exactly one path. The fixture keeps two
pure-LoS views, one with the BS in the front hemisphere (``ue_000000``) and one
with it in the back hemisphere (``ue_000004``), and every fourth frequency bin
(16 of 64, DC stays at index N // 2), in Sionna's complex64 precision. The
tomography operator tests reproduce it without importing Sionna.

Run from the repository root (no Sionna needed)::

    python tests/fixtures/rf_tomography/make_sionna_mock_los.py \
        data/generated/tp0_mock/rf_camera_multiview
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("sionna_mock_los.npz")
VIEW_IDS = ("ue_000000", "ue_000004")
BIN_STRIDE = 4


def main(dataset_dir: Path) -> None:
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    views = {view["view_id"]: view for view in manifest["views"]}
    view_index = [list(views).index(view_id) for view_id in VIEW_IDS]
    freq = np.asarray(manifest["frequency_offsets_hz"], dtype=np.float32)
    keep = np.arange(0, freq.size, BIN_STRIDE)

    gt = np.load(dataset_dir / manifest["path_geometry_gt"])
    valid = gt["valid"].reshape(len(views), -1)
    tau = gt["tau"].reshape(len(views), -1)
    if not np.all(valid[view_index].sum(axis=1) == 1):
        raise RuntimeError("each fixture view must receive exactly one path")

    aperture = np.stack(
        [
            np.load(dataset_dir / views[view_id]["artifacts"]["aperture_cfr"])[..., keep]
            for view_id in VIEW_IDS
        ]
    ).astype(np.complex64)

    np.savez_compressed(
        OUT,
        aperture_cfr=aperture,  # [view, hemisphere, row, col, freq], Sionna complex64
        freq_offsets=freq[keep],  # float32 grid Sionna evaluated Paths.cfr on
        view_ids=np.asarray(VIEW_IDS),
        ue_pos=np.asarray([views[v]["position_m"] for v in VIEW_IDS], dtype=np.float64),
        ue_orientation=np.asarray(
            [views[v]["orientation_rad"] for v in VIEW_IDS], dtype=np.float64
        ),
        bs_pos=np.asarray([config["tx_position"]], dtype=np.float64),
        bs_look_at=np.asarray(config["tx_look_at"], dtype=np.float64),
        f_c=np.float64(config["carrier_frequency_hz"]),
        path_tau=tau[view_index][valid[view_index]].astype(np.float64),
        bs_in_front=np.asarray([views[v]["bs_in_front_hemisphere"] for v in VIEW_IDS]),
    )
    print(f"wrote {OUT}: aperture_cfr {aperture.shape}, {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
