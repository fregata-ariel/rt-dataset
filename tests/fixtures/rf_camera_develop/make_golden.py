"""Generate (or check) the golden outputs of the per-BS RF-camera development.

The golden pins the five derived per-(view, BS) arrays written by
``RFMultiViewDataset._write_view_bs`` (``angular_cfr_center``,
``angular_power_center``, ``phase_valid_mask``, ``dominant_delay_s``,
``dominant_delay_power``) for a small, fixed-seed aperture: two plane waves
per hemisphere plus small complex noise, ``[hemisphere=2, row=8, col=8,
freq=16]``, developed on a 32x32 direction-cosine grid. The PNG is not part of
the golden.

The script imports the Sionna adapter, so it runs in the CI image (it does
not need a GPU; only NumPy code runs)::

    source scripts/ci/env.sh
    docker compose --profile ci run --rm ci \
        python tests/fixtures/rf_camera_develop/make_golden.py          # (re)write
    docker compose --profile ci run --rm ci \
        python tests/fixtures/rf_camera_develop/make_golden.py --check  # verify

The golden was written once with the first command (``ci`` service) before the
development steps were moved into ``plateau_rt.domain.rf_camera.develop``.
``--check`` re-runs the current ``_write_view_bs`` on the stored inputs and
requires byte-identical arrays (dtype, shape and NaN positions included).
The Sionna-free test ``tests/test_rf_camera_develop.py`` checks the domain
functions against the same files.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from plateau_rt.adapters.sionna.rf_camera_dataset import RFMultiViewConfig, RFMultiViewDataset
from plateau_rt.domain.rf_camera.camera import (
    HEMISPHERES,
    RFViewSpec,
    build_direction_cosine_camera_model,
    look_at_orientation,
)
from plateau_rt.domain.rf_camera.imaging import frequency_offsets

HERE = Path(__file__).resolve().parent
SEED = 20260925
NOISE_STD = 0.01
OUTPUT_NAMES = (
    "angular_cfr_center",
    "angular_power_center",
    "phase_valid_mask",
    "dominant_delay_s",
    "dominant_delay_power",
)
# (hemisphere, ky/k, kz/k, delay [s], complex amplitude) of the plane waves
PLANE_WAVES = (
    ("front", 0.31, -0.12, 23e-9, 1.0 + 0.0j),
    ("front", -0.45, 0.27, 71e-9, 0.35 - 0.2j),
    ("back", 0.12, 0.41, 47e-9, 0.6 + 0.3j),
    ("back", -0.62, -0.18, 131e-9, 0.25 + 0.1j),
)


def golden_config() -> RFMultiViewConfig:
    """Return the writer config used for the golden (defaults except the grid sizes)."""
    return RFMultiViewConfig(
        num_frequency_bins=16,
        rx_rows=8,
        rx_cols=8,
        fft_rows=32,
        fft_cols=32,
    )


def golden_view() -> RFViewSpec:
    """Return the (arbitrary) view the golden BS entry is written for."""
    position = (35.0, 5.0, 1.5)
    look_at = (5.0, 5.0, 5.0)
    return RFViewSpec("ue_000000", position, look_at, look_at_orientation(position, look_at))


def make_aperture(cfg: RFMultiViewConfig, offsets_hz: np.ndarray) -> np.ndarray:
    """Build the fixed-seed ``[hemisphere, row, col, freq]`` complex64 aperture."""
    rows, cols = cfg.rx_rows, cfg.rx_cols
    # Centered PlanarArray element positions in wavelengths; rows run toward -z.
    y = cfg.horizontal_spacing_lambda * (np.arange(cols) - (cols - 1) / 2.0)
    z = cfg.vertical_spacing_lambda * ((rows - 1) / 2.0 - np.arange(rows))
    freqs = np.asarray(offsets_hz, dtype=np.float64)
    aperture = np.zeros((len(HEMISPHERES), rows, cols, freqs.size), dtype=np.complex128)
    for hemisphere, ky, kz, tau, amplitude in PLANE_WAVES:
        spatial = np.exp(2j * np.pi * (ky * y[None, :] + kz * z[:, None]))
        spectral = np.exp(-2j * np.pi * freqs * tau)
        aperture[HEMISPHERES.index(hemisphere)] += (
            amplitude * spatial[:, :, None] * spectral[None, None, :]
        )
    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal(aperture.shape) + 1j * rng.standard_normal(aperture.shape)
    return (aperture + NOISE_STD * noise).astype(np.complex64)


def run_writer(
    cfg: RFMultiViewConfig,
    view: RFViewSpec,
    aperture: np.ndarray,
    offsets_hz: np.ndarray,
    valid_mask: np.ndarray,
    *,
    bs_id: str,
    bs_position: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    """Run the current ``_write_view_bs`` in a temp dir and load its five arrays."""
    fake_self = SimpleNamespace(config=cfg)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        entry = RFMultiViewDataset._write_view_bs(
            fake_self,  # type: ignore[arg-type]
            out,
            view,
            aperture,
            bs_id=bs_id,
            bs_position=bs_position,
            frequency_offsets_hz=offsets_hz,
            valid_ray_mask=valid_mask,
        )
        return {name: np.load(out / entry["artifacts"][name]) for name in OUTPUT_NAMES}


def load_inputs(
    directory: Path,
) -> tuple[RFMultiViewConfig, RFViewSpec, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load the stored golden inputs."""
    meta = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
    config = dict(meta["config"])
    for key in ("tx_positions", "tx_look_ats"):
        if config.get(key) is not None:
            config[key] = tuple(tuple(v) for v in config[key])
    config["tx_look_at"] = tuple(config["tx_look_at"])
    cfg = RFMultiViewConfig(**config)
    v = meta["view"]
    view = RFViewSpec(
        v["view_id"], tuple(v["position"]), tuple(v["look_at"]), tuple(v["orientation"])
    )
    with np.load(directory / "inputs.npz") as data:
        aperture = data["aperture_cfr_bs"]
        offsets_hz = data["frequency_offsets_hz"]
        valid_mask = data["valid_mask"]
    return cfg, view, aperture, offsets_hz, valid_mask, meta


def write_golden(directory: Path) -> None:
    """Write inputs and golden outputs into ``directory``."""
    cfg = golden_config()
    cfg.validate()
    view = golden_view()
    offsets_hz = frequency_offsets(cfg.bandwidth_hz, cfg.num_frequency_bins)
    valid_mask = build_direction_cosine_camera_model(
        fft_rows=cfg.fft_rows,
        fft_cols=cfg.fft_cols,
        horizontal_spacing_lambda=cfg.horizontal_spacing_lambda,
        vertical_spacing_lambda=cfg.vertical_spacing_lambda,
    )["valid_mask"]
    aperture = make_aperture(cfg, offsets_hz)
    bs_id, bs_position = "bs_000", (-50.0, -50.0, 30.0)

    np.savez(
        directory / "inputs.npz",
        aperture_cfr_bs=aperture,
        frequency_offsets_hz=offsets_hz,
        valid_mask=valid_mask,
    )
    meta = {
        "description": "Inputs of the RF-camera development golden (see make_golden.py)",
        "seed": SEED,
        "noise_std": NOISE_STD,
        "plane_waves": [
            {
                "hemisphere": h,
                "ky_over_k": ky,
                "kz_over_k": kz,
                "delay_s": tau,
                "amplitude": [a.real, a.imag],
            }
            for h, ky, kz, tau, a in PLANE_WAVES
        ],
        "hemispheres": list(HEMISPHERES),
        "aperture_axis_order": ["hemisphere", "row", "col", "freq"],
        "config": asdict(cfg),
        "view": {
            "view_id": view.view_id,
            "position": list(view.position),
            "look_at": list(view.look_at),
            "orientation": list(view.orientation),
        },
        "bs_id": bs_id,
        "bs_position": list(bs_position),
        "outputs": list(OUTPUT_NAMES),
    }
    (directory / "inputs.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    outputs = run_writer(
        cfg, view, aperture, offsets_hz, valid_mask, bs_id=bs_id, bs_position=bs_position
    )
    for name, array in outputs.items():
        np.save(directory / f"{name}.npy", array)
        print(f"wrote {name}.npy {array.dtype} {array.shape}")


def check_golden(directory: Path) -> int:
    """Re-run ``_write_view_bs`` on the stored inputs; return the number of mismatches."""
    cfg, view, aperture, offsets_hz, valid_mask, meta = load_inputs(directory)
    outputs = run_writer(
        cfg,
        view,
        aperture,
        offsets_hz,
        valid_mask,
        bs_id=meta["bs_id"],
        bs_position=tuple(meta["bs_position"]),
    )
    failures = 0
    for name in OUTPUT_NAMES:
        golden = np.load(directory / f"{name}.npy")
        actual = outputs[name]
        same = (
            golden.dtype == actual.dtype
            and golden.shape == actual.shape
            and golden.tobytes() == actual.tobytes()
        )
        status = "OK" if same else "MISMATCH"
        print(
            f"{status:8s} {name}: golden {golden.dtype}{golden.shape}, "
            f"actual {actual.dtype}{actual.shape}"
        )
        failures += 0 if same else 1
    return failures


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    parser.add_argument("--dir", type=Path, default=HERE, help="golden directory")
    args = parser.parse_args(argv)
    if args.check:
        failures = check_golden(args.dir)
        print("golden check:", "PASS" if failures == 0 else f"FAIL ({failures})")
        return 1 if failures else 0
    write_golden(args.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
