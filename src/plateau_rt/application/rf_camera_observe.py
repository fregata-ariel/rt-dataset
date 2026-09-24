"""Turn a multi-view RF-camera dataset into an observed single-channel dataset.

Reads ``dataset_manifest.json`` and each view's ideal two-hemisphere
``aperture_cfr.npy``, applies the ordered receiver impairments of
:mod:`plateau_rt.domain.rf_camera.impairments`, and writes each ``(view, BS)``
pair's observed CFR and its ground-truth parameters under
``views/<view_id>/rf/<bs_id>/observed/<name>/``. The manifest is rewritten in
place with the per-pair artifacts and an ``observations`` section.

Several observation variants (different ``name``) can coexist in one dataset:
re-running with the same ``name`` overwrites that variant idempotently and
leaves the others untouched. Only schema v3 (multi-BS) datasets are supported.

The element gain/phase error is a property of the UE receive array and is drawn
once per view and shared by all BSs; the timing offset, common phase and noise
belong to each ``(UE, BS)`` link and are drawn per pair. Noise is one
dataset-wide floor: ``--snr-db`` is relative to the maximum ideal isotropic
power over all pairs, so path loss and front-to-back attenuation show up as a
lower achieved SNR.

This module is NumPy-only and CPU-only; it never imports Sionna.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from plateau_rt.application.rf_dataset_manifest import (
    ManifestError,
    load_rf_dataset_manifest,
)
from plateau_rt.domain.rf_camera.impairments import (
    ImpairmentConfig,
    NoiseSpec,
    apply_impairments,
    draw_element_errors,
    isotropic_mean_power,
    resolve_noise_variance,
)

DEFAULT_OBSERVATION_NAME = "observed"

UE_ARRAY_STREAM = 1
LINK_STREAM = 2

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_MAX_SEED = 2**32 - 1

RNG_DESCRIPTION = (
    "element errors per view from SeedSequence([seed, view_index, 0, 1]); "
    "timing, common phase and noise per (view, BS) from "
    "SeedSequence([seed, view_index, bs_index, 2])"
)

OBSERVED_MODEL = (
    "Observed single-channel aperture CFR: front + g * back collapsed with a "
    "front-to-back gain g, multiplied by a per-element complex gain error (constant "
    "over frequency, drawn once per UE view and shared by all BSs), a per-link UE "
    "timing ramp and common phase, then corrupted by circular complex AWGN from one "
    "dataset-wide noise floor (SNR relative to the maximum ideal isotropic power over "
    "all (view, BS) pairs), in that order."
)


def ue_array_rng(seed: int, view_index: int) -> np.random.Generator:
    """RNG for the element gain/phase errors of view ``view_index``.

    The four-word entropy is padded with a fixed non-zero stream tag so it never
    collides with :func:`link_rng`.
    """
    return np.random.default_rng(np.random.SeedSequence([seed, view_index, 0, UE_ARRAY_STREAM]))


def link_rng(seed: int, view_index: int, bs_index: int) -> np.random.Generator:
    """RNG for the timing, common phase and noise of pair ``(view_index, bs_index)``."""
    return np.random.default_rng(np.random.SeedSequence([seed, view_index, bs_index, LINK_STREAM]))


def _validate_name(name: str) -> None:
    """Raise ``ValueError`` unless ``name`` is a valid observation name."""
    if not isinstance(name, str) or _NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(f"observation name must match ^[A-Za-z0-9][A-Za-z0-9_-]*$, got {name!r}")


def _validate_seed(seed: int) -> None:
    """Raise ``ValueError`` unless ``seed`` is an int in ``[0, 2**32 - 1]``."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer in [0, {_MAX_SEED}], got {seed!r}")
    if not 0 <= int(seed) <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}], got {seed!r}")


def observe_dataset(
    dataset_dir: Path,
    config: ImpairmentConfig,
    *,
    noise: NoiseSpec = NoiseSpec(),
    seed: int = 0,
    name: str = DEFAULT_OBSERVATION_NAME,
) -> Path:
    """Apply ``config`` to every pair and rewrite the manifest in place.

    Writes one observed CFR (``complex64``, ``[row, col, frequency]``) and one
    ``impairment_gt.json`` per ``(view, BS)`` pair under
    ``views/<view_id>/rf/<bs_id>/observed/<name>/``, registers them in the
    pair's ``artifacts`` and records the variant under ``raw["observations"]``.
    Returns the path of the rewritten manifest.
    """
    dataset_dir = Path(dataset_dir)
    dataset = load_rf_dataset_manifest(dataset_dir)
    if dataset.schema_version != 3:
        raise ManifestError(
            "rf-camera-observe needs a schema v3 dataset (multi-BS); "
            "re-generate it with rf-camera-multiview"
        )

    _validate_name(name)
    _validate_seed(seed)

    front_index = dataset.hemispheres.index("front")
    back_index = dataset.hemispheres.index("back")

    # Pass 1: load every view once and find the dataset-wide reference power.
    cfrs: dict[str, np.ndarray] = {}
    reference_power = 0.0
    reference_pair: tuple[str, str] | None = None
    for view in dataset.views:
        cfr = dataset.load_aperture_cfr(view)
        cfrs[view.view_id] = cfr
        for entry in view.bs:
            power = isotropic_mean_power(cfr[entry.bs_index][[front_index, back_index]])
            if reference_pair is None or power > reference_power:
                reference_power = power
                reference_pair = (view.view_id, entry.bs_id)

    noise_variance = resolve_noise_variance(noise, reference_power)

    # Pass 2: draw per-view element errors, then per-pair link impairments.
    raw: dict[str, Any] = dict(dataset.raw)
    raw.setdefault("observations", {})
    pairs: list[dict[str, Any]] = []
    for view in dataset.views:
        cfr = cfrs[view.view_id]
        errors = draw_element_errors(
            config, dataset.rx_rows, dataset.rx_cols, ue_array_rng(int(seed), view.index)
        )
        raw_view = raw["views"][view.index]
        for entry in view.bs:
            pair_cfr = cfr[entry.bs_index][[front_index, back_index]]
            observed, gt = apply_impairments(
                pair_cfr,
                dataset.frequency_offsets_hz,
                config,
                link_rng(int(seed), view.index, entry.bs_index),
                element_errors=errors,
                noise_variance=noise_variance,
            )
            gt["view_id"] = view.view_id
            gt["bs_id"] = entry.bs_id
            gt["observation"] = name
            gt["ideal_isotropic_power"] = isotropic_mean_power(pair_cfr)

            observed_dir = (
                dataset_dir / "views" / view.view_id / "rf" / entry.bs_id / "observed" / name
            )
            observed_dir.mkdir(parents=True, exist_ok=True)
            observed_path = observed_dir / "aperture_cfr.npy"
            gt_path = observed_dir / "impairment_gt.json"
            np.save(observed_path, observed.astype(np.complex64, copy=False))
            gt_path.write_text(json.dumps(gt, indent=2, allow_nan=False), encoding="utf-8")

            artifacts = raw_view["bs"][entry.bs_index].setdefault("artifacts", {})
            artifacts[f"observed.{name}.aperture_cfr"] = observed_path.relative_to(
                dataset_dir
            ).as_posix()
            artifacts[f"observed.{name}.impairment_gt"] = gt_path.relative_to(
                dataset_dir
            ).as_posix()

            pairs.append(
                {
                    "view_id": view.view_id,
                    "bs_id": entry.bs_id,
                    "ideal_isotropic_power": gt["ideal_isotropic_power"],
                    "signal_power": gt["signal_power"],
                    "expected_snr_db": gt["expected_snr_db"],
                    "achieved_snr_db": gt["achieved_snr_db"],
                }
            )

    raw["observations"][name] = {
        "artifact_keys": {
            "aperture_cfr": f"observed.{name}.aperture_cfr",
            "impairment_gt": f"observed.{name}.impairment_gt",
        },
        "axis_order": ["row", "col", "frequency_offset"],
        "seed": int(seed),
        "rng": RNG_DESCRIPTION,
        "config": asdict(config),
        "noise": {
            "mode": noise.mode,
            "snr_db": noise.snr_db,
            "reference": "max_pair_isotropic_mean_power",
            "reference_power": reference_power,
            "reference_pair": (
                None
                if reference_pair is None
                else {"view_id": reference_pair[0], "bs_id": reference_pair[1]}
            ),
            "noise_variance": noise_variance,
        },
        "model": OBSERVED_MODEL,
        "pairs": pairs,
    }

    dataset.manifest_path.write_text(json.dumps(raw, indent=2, allow_nan=False), encoding="utf-8")
    return dataset.manifest_path
