"""節目CI: パスGTから aperture CFR を再合成し、保存済み観測と一致することを確認する。

出力ディレクトリを共有の型付きリーダー (`load_rf_dataset_manifest`) 経由で読み、
`path_geometry_gt.npz` の `a_baseband` と `tau` から各 (view, BS) の aperture CFR を
`synthesize_cfr` で再合成して `views/*/rf/aperture_cfr.npy` と比較する。
GPU の実行ごとの揺らぎ分だけ許容誤差を持たせる。

明示的アレイ (`synthetic_array=False`, スキーマの mode が `sionna_native`) や
path schema を持たない古いデータセットは、[SKIP] を出して正常終了する。
参照された path schema が読めない場合は失敗とする。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from plateau_rt.application.rf_dataset_manifest import (
    ManifestError,
    load_rf_dataset_manifest,
)
from plateau_rt.domain.rf_camera.paths import TAU_QUANTUM_S, quantise_tau, synthesize_cfr

# GPU の実行ごとの揺らぎ・量子化分だけ許容誤差を持たせる
MAX_RELATIVE_ERROR = 1e-3
ZERO_TOLERANCE = 1e-12


def _relative_error(synthesized: np.ndarray, reference: np.ndarray) -> float:
    peak = float(np.max(np.abs(reference)))
    if peak == 0.0:
        return float(np.max(np.abs(synthesized)))
    return float(np.max(np.abs(synthesized - reference)) / peak)


def _check_canonical_order(valid: np.ndarray, tau: np.ndarray, *, num_views: int, num_bs: int):
    """Yield ``(label, message)`` for every (view, bs) violating canonical order."""
    for v_index in range(num_views):
        for b_index in range(num_bs):
            v_mask = np.asarray(valid[v_index, b_index], dtype=bool)
            count = int(np.count_nonzero(v_mask))
            if count and (not bool(np.all(v_mask[:count])) or bool(np.any(v_mask[count:]))):
                yield (v_index, b_index), "valid paths are not a contiguous prefix"
                continue
            if count < 2:
                continue
            valid_tau = np.asarray(tau[v_index, b_index, :count], dtype=np.float64)
            bins = quantise_tau(valid_tau)
            if bool((np.diff(bins) < 0).any()):
                yield (v_index, b_index), "quantised valid delays decrease"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("multiview_dir", type=Path, help="rf_camera_multiview output directory")
    args = parser.parse_args()

    dataset = load_rf_dataset_manifest(args.multiview_dir)
    path_gt = dataset.path_geometry_gt
    if path_gt is None:
        print("[SKIP] dataset has no path_geometry_gt")
        return
    if path_gt.schema_path is None:
        print("[SKIP] path_geometry_gt has no path_schema; canonical mode not confirmed")
        return
    try:
        schema = path_gt.load_schema()
    except ManifestError as exc:
        print(f"[NG] path schema unreadable: {exc}")
        sys.exit(1)
    mode = schema.get("mode")
    if mode != "canonical":
        print(f"[SKIP] path schema mode is {mode!r}, not 'canonical'")
        return

    arrays = path_gt.load_arrays()
    valid = np.asarray(arrays["valid"])
    tau = np.asarray(arrays["tau"])
    a_baseband = np.asarray(arrays["a_baseband"])
    offsets = dataset.frequency_offsets_hz
    print(f"tau_quantum_s={TAU_QUANTUM_S:g}, synthetic_array={schema.get('synthetic_array')}")

    failures: list[str] = []
    pair_labels: dict[tuple[int, int], str] = {
        (v_index, entry.bs_index): f"{view.view_id} {entry.bs_id}"
        for v_index, view in enumerate(dataset.views)
        for entry in view.bs
    }

    order_violations = list(
        _check_canonical_order(valid, tau, num_views=dataset.num_views, num_bs=dataset.num_bs)
    )
    if order_violations:
        for pair, message in order_violations:
            print(f"[NG] {pair_labels[pair]}: {message}")
            failures.append(pair_labels[pair])
    else:
        print("[OK] canonical path order holds for every (view, BS) pair")

    worst_error = -1.0
    worst_label = ""
    for v_index, view in enumerate(dataset.views):
        stored_all = dataset.load_aperture_cfr(view)
        for entry in view.bs:
            b_index = entry.bs_index
            stored = stored_all[b_index]
            synthesized = synthesize_cfr(
                a_baseband[v_index, b_index], tau[v_index, b_index], offsets
            )
            error = _relative_error(synthesized, stored)
            if float(np.max(np.abs(stored))) == 0.0:
                passed = float(np.max(np.abs(synthesized))) <= ZERO_TOLERANCE
            else:
                passed = error <= MAX_RELATIVE_ERROR
            label = f"{view.view_id} {entry.bs_id}"
            print(f"[{'OK' if passed else 'NG'}] {label}: max|H_syn - H| / max|H| = {error:.2e}")
            if not passed:
                failures.append(label)
            if error > worst_error:
                worst_error = error
                worst_label = label

    print(f"worst pair: {worst_label} = {worst_error:.2e}")

    if failures:
        print(f"❌ resynthesis/order failed for pairs: {sorted(set(failures))}")
        sys.exit(1)
    print("✅ Path GT resynthesis matches the stored aperture CFRs")


if __name__ == "__main__":
    main()
