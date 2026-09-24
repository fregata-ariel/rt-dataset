"""Tests for the Sionna path-GT adapter (canonical and explicit array modes)."""

from __future__ import annotations

import types

import numpy as np
import pytest

pytest.importorskip("sionna.rt")

from plateau_rt.adapters.sionna import rf_tracing
from plateau_rt.domain.rf_camera.paths import (
    PATH_GT_MODE_CANONICAL,
    PATH_GT_MODE_SIONNA_NATIVE,
    apply_path_order,
    canonical_path_order,
)

OBJECT_IDS = {"alpha": 7, "beta": 3}  # sorted names: alpha -> 0, beta -> 1
UNKNOWN_OBJECT_ID = 99


def _make_scene() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        objects={
            name: types.SimpleNamespace(object_id=object_id)
            for name, object_id in OBJECT_IDS.items()
        }
    )


def _make_synthetic_inputs(
    *,
    rx: int,
    tx: int,
    rows: int,
    cols: int,
    num_paths: int,
    depth: int = 2,
    seed: int = 0,
) -> dict:
    """Build a stub ``Paths`` with Sionna synthetic-array shapes."""
    num_patterns = 2
    fused = num_patterns * rows * cols
    rng = np.random.default_rng(seed)

    valid = np.ones((rx, tx, num_paths), dtype=bool)
    tau = rng.uniform(0.0, 500e-9, size=(rx, tx, num_paths)).astype(np.float32)
    # Invalidate the last path of every (rx, tx) pair.
    valid[..., -1] = False
    tau[..., -1] = -1.0

    angles = {
        name: rng.uniform(-1.0, 1.0, size=(rx, tx, num_paths)).astype(np.float32)
        for name in rf_tracing.PATH_ANGLE_FIELDS
    }
    a_b = (
        rng.standard_normal((rx, fused, tx, 1, num_paths))
        + 1j * rng.standard_normal((rx, fused, tx, 1, num_paths))
    ).astype(np.complex64)

    path_index = np.arange(num_paths)
    interactions = rng.integers(0, 4, size=(depth, rx, tx, num_paths)).astype(np.uint32)
    primitives = np.empty((depth, rx, tx, num_paths), dtype=np.uint32)
    vertices = np.empty((depth, rx, tx, num_paths, 3), dtype=np.float32)
    raw_objects = np.empty((depth, rx, tx, num_paths), dtype=np.uint32)
    for d in range(depth):
        primitives[d] = 100 * path_index + d
        vertices[d, ..., 0] = 1000 * d + path_index
        raw_objects[d] = np.where(
            path_index % 3 == 0,
            OBJECT_IDS["alpha"],
            np.where(path_index % 3 == 1, OBJECT_IDS["beta"], UNKNOWN_OBJECT_ID),
        ).astype(np.uint32)

    def cir(*, normalize_delays: bool = False, out_type: str = "numpy"):
        return (a_b[..., None], tau)

    patterns = [object(), object()]
    paths = types.SimpleNamespace(
        synthetic_array=True,
        valid=valid,
        tau=tau,
        a=(a_b.real.copy(), a_b.imag.copy()),
        cir=cir,
        interactions=interactions,
        objects=raw_objects,
        primitives=primitives,
        vertices=vertices,
        rx_array=types.SimpleNamespace(antenna_pattern=types.SimpleNamespace(patterns=patterns)),
        **angles,
    )
    return {
        "paths": paths,
        "scene": _make_scene(),
        "a_b": a_b,
        "valid": valid,
        "tau": tau,
        "angles": angles,
        "interactions": interactions,
        "objects": raw_objects,
        "primitives": primitives,
        "vertices": vertices,
    }


def _verify_canonical(inputs: dict, *, rows: int, cols: int) -> None:
    paths = inputs["paths"]
    result = rf_tracing.path_ground_truth(paths, inputs["scene"], rx_rows=rows, rx_cols=cols)
    assert result.mode == PATH_GT_MODE_CANONICAL
    assert set(result.arrays) == {
        "valid",
        "tau",
        "theta_t",
        "phi_t",
        "theta_r",
        "phi_r",
        "a_baseband",
        "interactions",
        "object_index",
        "primitives",
        "vertices",
        "num_interactions",
    }
    assert "a" not in result.arrays

    a_b = inputs["a_b"]
    tau = inputs["tau"]
    valid = inputs["valid"]
    rx, tx, num_paths = tau.shape
    depth = inputs["interactions"].shape[0]

    power = np.sum(np.abs(a_b[:, :, :, 0, :].astype(np.complex128)) ** 2, axis=1)
    order = canonical_path_order(tau, power, valid)

    known = {OBJECT_IDS["alpha"]: 0, OBJECT_IDS["beta"]: 1}
    for r in range(rx):
        for t in range(tx):
            pair_order = order[r, t]
            assert np.array_equal(result.arrays["valid"][r, t], valid[r, t][pair_order])
            assert np.array_equal(result.arrays["tau"][r, t], tau[r, t][pair_order])
            for name in rf_tracing.PATH_ANGLE_FIELDS:
                assert np.array_equal(
                    result.arrays[name][r, t], inputs["angles"][name][r, t][pair_order]
                )

            # a_baseband uses the fused column-first channel numbering, permuted.
            for pattern in range(2):
                for row in range(rows):
                    for col in range(cols):
                        channel = pattern * rows * cols + col * rows + row
                        expected = a_b[r, channel, t, 0, :][pair_order]
                        assert np.array_equal(
                            result.arrays["a_baseband"][r, t, pattern, row, col], expected
                        )

            expected_interactions = inputs["interactions"][:, r, t, :][:, pair_order].T
            assert np.array_equal(result.arrays["interactions"][r, t], expected_interactions)
            expected_primitives = inputs["primitives"][:, r, t, :][:, pair_order].T
            assert np.array_equal(result.arrays["primitives"][r, t], expected_primitives)
            expected_vertices = inputs["vertices"][:, r, t, :, :][:, pair_order, :].transpose(
                1, 0, 2
            )
            assert np.array_equal(result.arrays["vertices"][r, t], expected_vertices)
            assert np.array_equal(
                result.arrays["num_interactions"][r, t],
                np.sum(expected_interactions != 0, axis=-1).astype(np.int32),
            )

            expected_raw = inputs["objects"][:, r, t, :][:, pair_order].T
            expected_objects = np.full((num_paths, depth), -1, dtype=np.int32)
            for path in range(num_paths):
                for d in range(depth):
                    index = known.get(int(expected_raw[path, d]))
                    if index is not None:
                        expected_objects[path, d] = index
            assert np.array_equal(result.arrays["object_index"][r, t], expected_objects)


def test_num_tx_two_path_ground_truth():
    rows, cols = 2, 3
    num_paths = 5
    inputs = _make_synthetic_inputs(rx=2, tx=2, rows=rows, cols=cols, num_paths=num_paths)
    _verify_canonical(inputs, rows=rows, cols=cols)


def test_single_rx_tx_equals_fused_is_not_misordered():
    rows, cols = 1, 2
    num_paths = 5
    num_tx = 2 * rows * cols  # tx equals fused: the silent-misorder case
    inputs = _make_synthetic_inputs(
        rx=1, tx=num_tx, rows=rows, cols=cols, num_paths=num_paths, seed=1
    )
    _verify_canonical(inputs, rows=rows, cols=cols)


def _make_explicit_inputs(
    *, rx: int, rx_ant: int, tx: int, tx_ant: int, num_paths: int, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    shape = (rx, rx_ant, tx, tx_ant, num_paths)
    valid = rng.random(shape) > 0.3
    tau = rng.uniform(0.0, 500e-9, size=shape).astype(np.float32)
    angles = {
        name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        for name in rf_tracing.PATH_ANGLE_FIELDS
    }
    paths = types.SimpleNamespace(synthetic_array=False, valid=valid, tau=tau, **angles)
    return {
        "paths": paths,
        "scene": _make_scene(),
        "valid": valid,
        "tau": tau,
        "angles": angles,
    }


def test_explicit_array_returns_sionna_native_geometry():
    inputs = _make_explicit_inputs(rx=2, rx_ant=8, tx=2, tx_ant=1, num_paths=5)
    result = rf_tracing.path_ground_truth(inputs["paths"], inputs["scene"], rx_rows=2, rx_cols=2)
    assert result.mode == PATH_GT_MODE_SIONNA_NATIVE
    assert set(result.arrays) == {"valid", "tau", "theta_t", "phi_t", "theta_r", "phi_r"}
    assert np.array_equal(result.arrays["valid"], inputs["valid"])
    assert np.array_equal(result.arrays["tau"], inputs["tau"])
    for name in rf_tracing.PATH_ANGLE_FIELDS:
        assert np.array_equal(result.arrays[name], inputs["angles"][name])


def test_apply_path_order_broadcasts_middle_axis():
    rx, tx, middle, num_paths = 2, 3, 2, 4
    array = np.arange(rx * tx * middle * num_paths).reshape(rx, tx, middle, num_paths)
    rng = np.random.default_rng(3)
    order = np.stack([np.stack([rng.permutation(num_paths) for _ in range(tx)]) for _ in range(rx)])

    out = apply_path_order(array, order, path_axis=-1)

    assert out.shape == array.shape
    for a in range(rx):
        for b in range(tx):
            for k in range(middle):
                assert np.array_equal(out[a, b, k], array[a, b, k, order[a, b]])


def test_apply_path_order_rejects_mismatched_leading_shape():
    rx, tx, wrong, num_paths = 2, 3, 2, 4
    order = np.zeros((rx, tx, num_paths), dtype=np.int64)
    mismatched = np.zeros((rx, wrong, tx, num_paths))

    with pytest.raises(ValueError, match=r"leading dimensions .* do not match"):
        apply_path_order(mismatched, order, path_axis=-1)
