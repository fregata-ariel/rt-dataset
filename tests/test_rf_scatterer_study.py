import json

import numpy as np

from plateau_rt.experimental.rf_scatterer_fit import discrepancy_damp, reconstruct
from plateau_rt.experimental.rf_scatterer_study import (
    SweepConfig,
    _write_outputs,
    build_scene,
    evaluate_config,
    run_study,
)


def test_quick_study_writes_outputs(tmp_path):
    run_study(tmp_path, seed=0, quick=True)

    assert (tmp_path / "results.md").exists()
    assert (tmp_path / "results.json").exists()

    records = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["records"]
    ok_records = [record for record in records if record["status"] == "ok"]
    assert ok_records
    for record in ok_records:
        for key in ("damp_rel", "noisy_error", "bias_error", "oracle_error"):
            assert np.isfinite(record[key])


def test_discrepancy_damp_improves_well_conditioned_config():
    config = SweepConfig(num_views=8, rows=8, cols=8, bandwidth_hz=100e6)
    record = evaluate_config(config, seed=0)

    assert record["status"] == "ok"
    assert record["noisy_error"] < 0.1
    assert record["bias_error"] < record["noisy_error"]


def test_discrepancy_damp_unit():
    rng = np.random.default_rng(3)
    rows, cols = 60, 15
    matrix = rng.standard_normal((rows, cols)) + 1j * rng.standard_normal((rows, cols))
    truth = rng.standard_normal(cols) + 1j * rng.standard_normal(cols)
    sigma = 0.05
    noise = sigma / np.sqrt(2.0) * (rng.standard_normal(rows) + 1j * rng.standard_normal(rows))
    observations = matrix @ truth + noise

    s_max = float(np.linalg.svd(matrix, compute_uv=False)[0])
    damps = s_max * np.logspace(-8, 0, 33)
    chosen = discrepancy_damp(matrix, observations, sigma, damps, tau=1.0)
    threshold = sigma * np.sqrt(rows)

    residual = np.linalg.norm(
        matrix @ reconstruct(matrix, observations, damp=chosen, method="normal") - observations
    )
    assert residual <= threshold * (1.0 + 1e-9)

    larger = damps[damps > chosen]
    assert larger.size > 0
    next_larger = float(larger.min())
    residual_next = np.linalg.norm(
        matrix @ reconstruct(matrix, observations, damp=next_larger, method="normal") - observations
    )
    assert residual_next > threshold


def test_write_outputs_handles_only_skipped_records(tmp_path):
    _, rho_true, axes, _ = build_scene()
    records = [
        {"views": 1, "aperture": "4x4", "bandwidth_mhz": 20.0, "status": "skipped"},
    ]

    _write_outputs(tmp_path, records, {}, rho_true, axes, seed=0)

    assert (tmp_path / "results.md").exists()
    assert (tmp_path / "results.json").exists()
