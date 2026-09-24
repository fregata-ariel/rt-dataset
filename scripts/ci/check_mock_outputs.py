"""節目CI: モックE2Eパイプラインの生成物を検証する。

Makefile の *-mock ターゲットで生成した出力ディレクトリを受け取り、
ファイルの有無、配列の形状・有限性、LoSの角度/遅延の物理的な整合性を確認する。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from plateau_rt.application.rf_dataset_manifest import (
    APERTURE_CFR_AXIS_ORDER,
    ManifestError,
    RFDatasetManifest,
    load_rf_dataset_manifest,
)
from plateau_rt.domain.rf_camera.optical import (
    PinholeIntrinsics,
    local_to_world_rays,
    pinhole_ray_directions_local,
)

# 8x8 開口 / 128点FFTの角度ビン幅は 2/128 ≈ 0.016。LoS推定の許容誤差は約3ビンとする
MAX_LOS_YZ_PROJECTION_ERROR = 0.05

# mock シーンの箱形状 (CityJSON バウンディングボックス中心が原点になる):
# x, y in [-5, 5]、z in [0, 10] の1棟のみ
MOCK_BOX_MIN = np.array([-5.0, -5.0, 0.0])
MOCK_BOX_MAX = np.array([5.0, 5.0, 10.0])
# 光学レンダーの幾何チェック許容誤差
OPTICAL_HIT_AGREEMENT_MIN = 0.995
OPTICAL_RANGE_TOL = 1e-3
# レイ方向のある軸成分がこれ未満なら、その軸はスラブ法で「平行」とみなす
# (単位ベクトルの丸め誤差 ~1e-16 よりずっと大きく、実際に軸に平行なレイの角度
# 誤差よりずっと小さい)
RAY_BOX_PARALLEL_EPS = 1e-9
# LOS がクリアとみなす箱からの余裕 (ボックスをこの分だけ膨らませて判定する)
LOS_CLEARANCE_M = 1.0
# パスGTのLoS遅延チェック許容誤差 (3 cm 相当)
SPEED_OF_LIGHT_M_S = 299792458.0
LOS_TAU_TOL_S = 1e-10


def segment_clear_of_box(
    p0: Sequence[float],
    p1: Sequence[float],
    box_min: Sequence[float],
    box_max: Sequence[float],
) -> bool:
    """Return True when the segment p0-p1 does not intersect the box (slab method)."""
    p0_arr = np.asarray(p0, dtype=np.float64)
    p1_arr = np.asarray(p1, dtype=np.float64)
    lo = np.asarray(box_min, dtype=np.float64)
    hi = np.asarray(box_max, dtype=np.float64)
    t_enter = 0.0
    t_exit = 1.0
    for axis in range(3):
        origin = float(p0_arr[axis])
        direction = float(p1_arr[axis] - p0_arr[axis])
        if direction == 0.0:
            if origin < lo[axis] or origin > hi[axis]:
                return True
            continue
        t0 = (lo[axis] - origin) / direction
        t1 = (hi[axis] - origin) / direction
        if t0 > t1:
            t0, t1 = t1, t0
        t_enter = max(t_enter, t0)
        t_exit = min(t_exit, t1)
        if t_enter > t_exit:
            return True
    if t_exit < 0.0 or t_enter > 1.0:
        return True
    return False


class Checker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.failures: list[str] = []

    def check(self, ok: bool, message: str) -> bool:
        print(f"[{'OK' if ok else 'NG'}] {message}")
        if not ok:
            self.failures.append(message)
        return ok

    def exists(self, path: Path) -> bool:
        return self.check(path.is_file(), f"exists: {path.relative_to(self.root)}")

    def finite(self, path: Path, *, allow_nan: bool = False) -> np.ndarray | None:
        if not self.exists(path):
            return None
        array = np.load(path)
        values = array[~np.isnan(array)] if allow_nan else array
        self.check(
            bool(np.isfinite(values).all()), f"finite: {path.relative_to(self.root)} {array.shape}"
        )
        return array

    def finite_nonzero(self, path: Path, *, allow_nan: bool = False) -> np.ndarray | None:
        array = self.finite(path, allow_nan=allow_nan)
        if array is not None:
            values = array[~np.isnan(array)] if allow_nan else array
            self.check(bool(np.abs(values).max() > 0), f"nonzero: {path.relative_to(self.root)}")
        return array


def check_scene_and_coverage(c: Checker, out: Path) -> None:
    print("--- scene build / coverage ---")
    c.exists(out / "mock_building.city.xml")
    if c.exists(out / "manifest.json"):
        manifest = json.loads((out / "manifest.json").read_text())
        c.check(manifest["outputs"]["mesh_count"] >= 1, "manifest: mesh_count >= 1")
    c.finite_nonzero(out / "mock_building.city_coverage.npy")
    for png in ("render_2d_heatmap.png", "heatmap_path_gain.png", "heatmap_rss_linear.png"):
        c.exists(out / png)


def check_rf_camera(c: Checker, rf: Path) -> None:
    print("--- 1BS/1UE RF camera + calibration + angle-delay ---")
    c.exists(rf / "rf_camera_metadata.json")
    c.finite_nonzero(rf / "aperture_cfr.npy")
    c.finite_nonzero(rf / "angular_cfr_calibrated.npy")
    c.finite_nonzero(rf / "angular_delay_cir.npy")

    if c.exists(rf / "angular_calibration.json"):
        calibration = json.loads((rf / "angular_calibration.json").read_text())
        error = calibration["center_frequency_peak"]["yz_projection_error_to_geometric_los"]
        c.check(
            error <= MAX_LOS_YZ_PROJECTION_ERROR,
            f"calibrated peak vs geometric LoS: {error:.4f} <= {MAX_LOS_YZ_PROJECTION_ERROR}",
        )

    if c.exists(rf / "angle_delay_report.json"):
        report = json.loads((rf / "angle_delay_report.json").read_text())
        voxel = report["strongest_physical_voxel"]
        delay_error = abs(voxel["circular_delay_error_to_geometric_los_s"])
        resolution = report["delay_resolution_s"]
        c.check(
            delay_error <= resolution,
            f"strongest voxel delay vs LoS: {delay_error * 1e9:.2f} ns "
            f"<= resolution {resolution * 1e9:.2f} ns",
        )


def check_multiview(c: Checker, mv: Path, num_views: int, num_bs: int) -> None:
    print("--- multi-BS/multi-UE RF camera dataset ---")
    if not c.exists(mv / "dataset_manifest.json"):
        return
    manifest = json.loads((mv / "dataset_manifest.json").read_text())
    config = manifest["config"]
    views = manifest["views"]
    c.check(manifest["schema_version"] == 3, f"schema_version: {manifest['schema_version']} == 3")
    c.check(
        manifest["mode"] == "multibs_multiue_rf_camera_dataset",
        f"mode: {manifest['mode']}",
    )
    c.check(len(views) == num_views, f"views: {len(views)} == {num_views}")
    base_stations = manifest.get("base_stations", [])
    c.check(len(base_stations) == num_bs, f"base_stations: {len(base_stations)} == {num_bs}")
    hemispheres = manifest["raw_observation"]["hemispheres"]
    c.check(hemispheres == ["front", "back"], f"hemispheres: {hemispheres}")
    axis_order = manifest["raw_observation"]["axis_order"]
    c.check(
        axis_order == ["bs", "hemisphere", "row", "col", "frequency_offset"],
        f"raw axis_order: {axis_order}",
    )
    c.check(
        manifest["raw_observation"]["bs_ids"] == [bs["bs_id"] for bs in base_stations],
        f"bs_ids: {manifest['raw_observation']['bs_ids']}",
    )

    if c.exists(mv / "camera_model.npz"):
        model = np.load(mv / "camera_model.npz")
        expected = (config["fft_rows"], config["fft_cols"], 3)
        shape = model["ray_directions_local"].shape
        c.check(shape == expected, f"camera rays shape: {shape} == {expected}")
        weight_shape = model["solid_angle_weight"].shape
        c.check(weight_shape == expected[:2], f"solid-angle weight shape: {weight_shape}")
    # 以降は共有の型付きリーダー経由で読む (レイアウトの整合性もここで検証される)
    dataset = load_dataset(c, mv)
    if dataset is None:
        return
    c.check(
        dataset.aperture_cfr_axis_order == APERTURE_CFR_AXIS_ORDER,
        f"reader aperture axis order: {list(dataset.aperture_cfr_axis_order)}",
    )
    aperture_shape = dataset.aperture_cfr_shape
    expected_shape = (
        num_bs,
        len(hemispheres),
        config["rx_rows"],
        config["rx_cols"],
        config["num_frequency_bins"],
    )
    c.check(
        aperture_shape == expected_shape,
        f"reader aperture_cfr_shape {aperture_shape} == {expected_shape}",
    )
    for view in dataset.views:
        view_id = view.view_id
        for path in view.artifacts.values():
            c.exists(path)
        for bs_entry in view.bs:
            for path in bs_entry.artifacts.values():
                c.exists(path)
        # 全BS合計で何らかのエネルギーが届いている
        cfr = c.finite(view.aperture_cfr_path)
        shape_ok = cfr is not None and c.check(
            cfr.shape == aperture_shape, f"{view_id} aperture_cfr {cfr.shape}"
        )
        if shape_ok:
            total_energy = float(np.sum(np.abs(cfr) ** 2))
            c.check(total_energy > 0, f"{view_id} total energy nonzero ({total_energy:.3e})")
            for bs_entry in view.bs:
                bs_id = bs_entry.bs_id
                energy = {
                    h: float(np.sum(np.abs(cfr[bs_entry.bs_index, i]) ** 2))
                    for i, h in enumerate(hemispheres)
                }
                recorded = bs_entry.hemisphere_energy
                c.check(
                    all(np.isclose(energy[h], recorded[h], rtol=1e-4) for h in hemispheres),
                    f"{view_id} {bs_id} hemisphere energy matches manifest",
                )
                # mock は直接波が支配的なので、エネルギーの大半は BS のある半球から届く
                # (前後の分割が入れ替わっていないことの確認)。BSからの到達が皆無の
                # 場合は向きの判定ができないため、エネルギーが正のときだけ確認する
                if sum(energy.values()) > 0:
                    dominant = max(energy, key=energy.get)
                    bs_side = "front" if bs_entry.bs_in_front_hemisphere else "back"
                    c.check(
                        dominant == bs_side,
                        f"{view_id} {bs_id} dominant hemisphere {dominant} == {bs_side}",
                    )
        # 前面から何も届かない(BS, 視点)では現像画像が空になる (背面の光源と同じ扱い)
        for bs_entry in view.bs:
            c.finite(bs_entry.artifact("angular_power_center"))
            # 位相が有効でない画素は NaN で埋められる
            c.finite(bs_entry.artifact("dominant_delay_s"), allow_nan=True)

    # 全ビュー合計で各BSが何らかのエネルギーを届けている (全ゼロBSの検出)
    bs_total_energy = {bs_id: 0.0 for bs_id in dataset.bs_ids}
    for _view, bs_entry in dataset.pairs():
        bs_total_energy[bs_entry.bs_id] += bs_entry.total_energy
    for bs_id, total in bs_total_energy.items():
        c.check(total > 0, f"{bs_id} total energy over all views nonzero ({total:.3e})")

    # 幾何学的LOSがクリアな (view, BS) ペアはエネルギーが正でなければならない
    grown_min = tuple(MOCK_BOX_MIN - LOS_CLEARANCE_M)
    grown_max = tuple(MOCK_BOX_MAX + LOS_CLEARANCE_M)
    required_counts = {bs_id: 0 for bs_id in dataset.bs_ids}
    for view, bs_entry in dataset.pairs():
        station = dataset.base_stations[bs_entry.bs_index]
        if segment_clear_of_box(station.position_m, view.position_m, grown_min, grown_max):
            required_counts[bs_entry.bs_id] += 1
            c.check(
                bs_entry.total_energy > 0,
                f"{view.view_id} {bs_entry.bs_id} clear-LOS energy nonzero "
                f"({bs_entry.total_energy:.3e})",
            )
    for bs_id, count in required_counts.items():
        c.check(True, f"{bs_id} clear-LOS pairs required nonzero: {count}/{dataset.num_views}")

    check_path_geometry(c, dataset, grown_min=grown_min, grown_max=grown_max)


def check_path_geometry(
    c: Checker,
    dataset: RFDatasetManifest,
    *,
    grown_min: Sequence[float],
    grown_max: Sequence[float],
) -> None:
    """Path-GT artifact: schema consistency and physical LoS delays via the reader."""
    print("--- path-level ground truth ---")
    path_gt = dataset.path_geometry_gt
    if path_gt is None:
        c.check(False, "manifest has path_geometry_gt")
        return
    c.exists(path_gt.path)
    if path_gt.schema_path is None:
        c.check(False, "manifest has path_schema")
        return
    c.exists(path_gt.schema_path)
    try:
        schema = path_gt.load_schema()
    except ManifestError as exc:
        c.check(False, f"path schema readable: {exc}")
        return
    c.check(
        schema.get("mode") == "canonical",
        f"path schema mode: {schema.get('mode')!r} == 'canonical'",
    )
    try:
        arrays = path_gt.load_arrays()
    except ManifestError as exc:
        c.check(False, f"path GT arrays readable: {exc}")
        return

    c.check(
        schema.get("bs_ids") == list(dataset.bs_ids),
        f"path schema bs_ids {schema.get('bs_ids')} == {list(dataset.bs_ids)}",
    )
    c.check(
        schema.get("view_ids") == list(dataset.view_ids),
        f"path schema view_ids {schema.get('view_ids')} == {list(dataset.view_ids)}",
    )

    for name, spec in schema["arrays"].items():
        if name not in arrays:
            c.check(False, f"path GT array {name} present in npz")
            continue
        array = np.asarray(arrays[name])
        shape_ok = tuple(array.shape) == tuple(spec.get("shape", ()))
        dtype_ok = str(array.dtype) == spec.get("dtype")
        c.check(
            shape_ok and dtype_ok,
            f"path GT {name} matches schema {spec.get('dtype')} {spec.get('shape')} "
            f"(got {array.dtype} {array.shape})",
        )

    def axis_size(name: str, axis: str) -> int | None:
        axes = path_gt.array_axes(name)
        if axis not in axes or name not in arrays:
            c.check(False, f"path GT {name} has axis {axis!r} (axes {list(axes)})")
            return None
        return int(np.asarray(arrays[name]).shape[axes.index(axis)])

    axis_checks = {
        ("valid", "view"): dataset.num_views,
        ("valid", "bs"): dataset.num_bs,
        ("a_baseband", "hemisphere"): len(dataset.hemispheres),
        ("a_baseband", "row"): dataset.rx_rows,
        ("a_baseband", "col"): dataset.rx_cols,
    }
    for (name, axis), expected in axis_checks.items():
        size = axis_size(name, axis)
        c.check(size == expected, f"path GT {name} axis {axis}: {size} == {expected}")

    valid = np.asarray(arrays["valid"])
    tau = np.asarray(arrays["tau"], dtype=np.float64)
    num_interactions = np.asarray(arrays["num_interactions"])

    max_deviation = 0.0
    clear_pairs = 0
    for v_index, view in enumerate(dataset.views):
        for b_index, entry in enumerate(view.bs):
            station = dataset.base_stations[entry.bs_index]
            distance = float(
                np.linalg.norm(
                    np.asarray(station.position_m, dtype=np.float64)
                    - np.asarray(view.position_m, dtype=np.float64)
                )
            )
            los_tau = distance / SPEED_OF_LIGHT_M_S
            v_mask = np.asarray(valid[v_index, b_index], dtype=bool)
            count = int(np.count_nonzero(v_mask))
            clear = segment_clear_of_box(station.position_m, view.position_m, grown_min, grown_max)
            label = f"{view.view_id} {entry.bs_id}"
            if count == 0:
                if clear:
                    c.check(False, f"{label} clear-LOS pair has at least one valid path")
                continue

            pair_tau = tau[v_index, b_index, :count]
            pair_interactions = np.asarray(num_interactions[v_index, b_index, :count])
            c.check(
                bool((pair_tau >= los_tau - LOS_TAU_TOL_S).all()),
                f"{label} no valid path arrives before |BS-UE|/c - tol",
            )
            los_paths = pair_interactions == 0
            if los_paths.any():
                deviation = np.abs(pair_tau[los_paths] - los_tau)
                max_deviation = max(max_deviation, float(np.max(deviation)))
                c.check(
                    bool((deviation <= LOS_TAU_TOL_S).all()),
                    f"{label} zero-interaction delays match |BS-UE|/c within {LOS_TAU_TOL_S:g} s",
                )
            if clear:
                clear_pairs += 1
                first_los = bool(pair_interactions[0] == 0)
                first_deviation = abs(float(pair_tau[0]) - los_tau)
                max_deviation = max(max_deviation, first_deviation)
                c.check(
                    first_los and first_deviation <= LOS_TAU_TOL_S,
                    f"{label} first valid path is LoS (num_interactions="
                    f"{int(pair_interactions[0])}, |tau-d/c|={first_deviation:.3e} s)",
                )

    c.check(
        True,
        f"path GT LoS delay check: {clear_pairs} clear-LOS pairs, "
        f"max |tau - |BS-UE|/c| = {max_deviation:.3e} s",
    )


def load_dataset(c: Checker, mv: Path) -> RFDatasetManifest | None:
    """共有リーダーで dataset_manifest.json を読む。失敗は NG として記録する。"""
    try:
        dataset = load_rf_dataset_manifest(mv)
    except (ManifestError, OSError) as exc:
        c.check(False, f"dataset manifest readable by rf_dataset_manifest: {exc}")
        return None
    c.check(
        True,
        f"dataset manifest readable by rf_dataset_manifest "
        f"(schema v{dataset.schema_version}, {dataset.num_views} views, {dataset.num_bs} BS)",
    )
    return dataset


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Max absolute difference, or 0.0 if there is nothing to compare."""
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def ray_box_intersect(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slab法で mock ボックスとの交差を解析的に求める。

    ``origins``/``directions`` は ``[N, 3]``。戻り値は ``(hit, range_m)``
    (``directions`` は単位ベクトルを前提とし ``range_m`` はそのまま距離になる)。

    ある軸のレイ方向成分がほぼ0 (``RAY_BOX_PARALLEL_EPS`` 未満) の場合は素直な
    ``1/direction`` を避け、その軸を「平行」として扱う: 原点がその軸のスラブ内
    (境界含む) ならその軸は t を制約せず、外なら不命中にする。単純な
    ``1/direction`` のまま処理すると、原点がちょうどそのスラブの境界面上にあり
    かつ方向成分が丸め誤差程度 (~1e-16) のときに ``inv_direction`` が極端に
    大きくなり、他の軸の本来の近傍交点を覆い隠してしまう (実測: 望遠方向が
    ちょうど mock ボックスの壁面を含む平面と一致する視点で発生)。
    """
    near = np.full(origins.shape[0], -np.inf)
    far = np.full(origins.shape[0], np.inf)
    blocked = np.zeros(origins.shape[0], dtype=bool)
    for axis in range(3):
        o = origins[:, axis]
        d = directions[:, axis]
        parallel = np.abs(d) < RAY_BOX_PARALLEL_EPS
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_d = 1.0 / np.where(parallel, 1.0, d)
        t1 = (MOCK_BOX_MIN[axis] - o) * inv_d
        t2 = (MOCK_BOX_MAX[axis] - o) * inv_d
        axis_lo = np.where(parallel, -np.inf, np.minimum(t1, t2))
        axis_hi = np.where(parallel, np.inf, np.maximum(t1, t2))
        outside_slab = parallel & (
            (o < MOCK_BOX_MIN[axis] - RAY_BOX_PARALLEL_EPS)
            | (o > MOCK_BOX_MAX[axis] + RAY_BOX_PARALLEL_EPS)
        )
        blocked |= outside_slab
        near = np.maximum(near, axis_lo)
        far = np.minimum(far, axis_hi)
    hit = ~blocked & np.isfinite(near) & np.isfinite(far) & (far >= near) & (far >= 0.0)
    range_m = np.where(hit, np.where(near >= 0.0, near, far), np.nan)
    return hit, range_m


def check_optical(c: Checker, mv: Path) -> None:
    """光学参照レンダー (rf-camera-optical) の生成物を検証する。

    ``dataset_manifest.json`` に ``optical_reference`` が無ければ何もしない
    (このステージを実行していない場合は静かにスキップする)。
    """
    manifest_path = mv / "dataset_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    if "optical_reference" not in manifest:
        return

    print("--- optical reference renders (rf-camera-optical) ---")
    dataset = load_dataset(c, mv)
    if dataset is None:
        return
    optical_cfg = manifest["optical_reference"]
    pinhole_cfg = optical_cfg["pinhole"]
    width, height = int(pinhole_cfg["width"]), int(pinhole_cfg["height"])
    intrinsics = PinholeIntrinsics.from_horizontal_fov(width, height, pinhole_cfg["fov_x_deg"])
    pinhole_dirs_local = pinhole_ray_directions_local(intrinsics)

    transforms_path = mv / pinhole_cfg["transforms"]
    if not c.exists(transforms_path):
        return
    transforms = json.loads(transforms_path.read_text())
    c.check(
        len(transforms["frames"]) == dataset.num_views,
        f"transforms.json: {len(transforms['frames'])} frames == {dataset.num_views} views",
    )

    if not c.exists(dataset.camera_model_path):
        return
    camera_model = np.load(dataset.camera_model_path)
    valid_mask = camera_model["valid_mask"]
    hemisphere_dirs_local = camera_model["ray_directions_local"]

    optical_names = (
        "optical_pinhole_rgba",
        "optical_pinhole_depth_m",
        "optical_pinhole_range_m",
        "optical_hemisphere_rgba",
        "optical_hemisphere_range_m",
    )
    for view, frame in zip(dataset.views, transforms["frames"]):
        view_id = view.view_id
        # 光学レンダーは姿勢だけに依存するので、BSごとではなくビュー単位で1組だけ持つ
        missing = [name for name in optical_names if name not in view.artifacts]
        if not c.check(not missing, f"{view_id} view-level optical artifacts present {missing}"):
            continue
        per_bs = [
            f"{entry.bs_id}:{name}"
            for entry in view.bs
            for name in entry.artifacts
            if name.startswith("optical_")
        ]
        c.check(not per_bs, f"{view_id} no optical artifacts in per-BS entries {per_bs}")
        artifacts = {name: view.artifact(name) for name in optical_names}
        for path in artifacts.values():
            c.exists(path)

        pose = json.loads(view.pose_path.read_text())
        rotation = np.asarray(pose["world_from_local_rotation"], dtype=np.float64)
        position = np.asarray(pose["position_m"], dtype=np.float64)

        matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
        rotation_part = matrix[:3, :3]
        det = float(np.linalg.det(rotation_part))
        c.check(abs(det - 1.0) < 1e-6, f"{view_id} transforms.json: rotation det~+1 ({det:.6f})")
        c.check(
            bool(np.allclose(matrix[:3, 3], position, atol=1e-6)),
            f"{view_id} transforms.json: translation == view position",
        )
        gl_forward = -rotation_part[:, 2]
        c.check(
            bool(np.allclose(gl_forward, rotation[:, 0], atol=1e-6)),
            f"{view_id} transforms.json: GL forward (-col 2) == world_from_local forward",
        )

        # --- pinhole: shape + 幾何 (analytic box intersection) ---
        pinhole_rgba = np.asarray(Image.open(artifacts["optical_pinhole_rgba"]))
        c.check(
            pinhole_rgba.shape == (height, width, 4),
            f"{view_id} pinhole rgba shape {pinhole_rgba.shape} == {(height, width, 4)}",
        )
        depth = np.load(artifacts["optical_pinhole_depth_m"])
        c.check(depth.shape == (height, width), f"{view_id} pinhole depth shape {depth.shape}")
        range_m = np.load(artifacts["optical_pinhole_range_m"])
        c.check(range_m.shape == (height, width), f"{view_id} pinhole range shape {range_m.shape}")

        origins, directions = local_to_world_rays(pinhole_dirs_local, rotation, position)
        expected_hit, expected_range = ray_box_intersect(
            origins.reshape(-1, 3), directions.reshape(-1, 3)
        )
        expected_hit = expected_hit.reshape(height, width)
        expected_range = expected_range.reshape(height, width)
        expected_depth = expected_range * pinhole_dirs_local[..., 0]

        alpha = pinhole_rgba[..., 3]
        rgb = pinhole_rgba[..., :3]
        rendered_hit = alpha > 0
        agreement = float(np.mean(rendered_hit == expected_hit))
        c.check(
            agreement >= OPTICAL_HIT_AGREEMENT_MIN,
            f"{view_id} pinhole hit-mask agreement {agreement:.4f} >= {OPTICAL_HIT_AGREEMENT_MIN}",
        )
        both_hit = rendered_hit & expected_hit
        depth_err = _max_abs_diff(depth[both_hit], expected_depth[both_hit])
        c.check(
            depth_err <= OPTICAL_RANGE_TOL,
            f"{view_id} pinhole z-depth vs analytic box: max err {depth_err:.2e} m",
        )
        range_err = _max_abs_diff(range_m[both_hit], expected_range[both_hit])
        c.check(
            range_err <= OPTICAL_RANGE_TOL,
            f"{view_id} pinhole range vs analytic box: max err {range_err:.2e} m",
        )
        c.check(bool(np.all(alpha[rendered_hit] == 255)), f"{view_id} pinhole alpha == 255 on hits")
        c.check(bool(np.all(rgb[rendered_hit] > 0)), f"{view_id} pinhole rgb > 0 on hits")
        c.check(bool(np.all(alpha[~rendered_hit] == 0)), f"{view_id} pinhole alpha == 0 on misses")

        # --- hemisphere: shape + 幾何 (analytic box intersection) ---
        hemisphere_rgba_png = np.asarray(Image.open(artifacts["optical_hemisphere_rgba"]))
        c.check(
            hemisphere_rgba_png.shape == valid_mask.shape + (4,),
            f"{view_id} hemisphere rgba shape {hemisphere_rgba_png.shape}",
        )
        hemisphere_range = np.load(artifacts["optical_hemisphere_range_m"])
        c.check(
            hemisphere_range.shape == valid_mask.shape,
            f"{view_id} hemisphere range shape {hemisphere_range.shape}",
        )
        # PNG は表示用に行を反転しているので、配列と同じ並びに戻す
        hemisphere_rgba = np.flipud(hemisphere_rgba_png)

        valid_dirs_local = hemisphere_dirs_local[valid_mask]
        h_origins, h_directions = local_to_world_rays(valid_dirs_local, rotation, position)
        h_expected_hit_valid, h_expected_range_valid = ray_box_intersect(h_origins, h_directions)
        h_expected_hit = np.zeros(valid_mask.shape, dtype=bool)
        h_expected_hit[valid_mask] = h_expected_hit_valid
        h_expected_range = np.full(valid_mask.shape, np.nan)
        h_expected_range[valid_mask] = h_expected_range_valid

        h_alpha = hemisphere_rgba[..., 3]
        h_rgb = hemisphere_rgba[..., :3]
        h_rendered_hit = (h_alpha > 0) & valid_mask
        h_agreement = float(np.mean(h_rendered_hit[valid_mask] == h_expected_hit[valid_mask]))
        c.check(
            h_agreement >= OPTICAL_HIT_AGREEMENT_MIN,
            f"{view_id} hemisphere hit-mask agreement {h_agreement:.4f} "
            f">= {OPTICAL_HIT_AGREEMENT_MIN}",
        )
        h_both_hit = h_rendered_hit & h_expected_hit
        h_range_err = _max_abs_diff(hemisphere_range[h_both_hit], h_expected_range[h_both_hit])
        c.check(
            h_range_err <= OPTICAL_RANGE_TOL,
            f"{view_id} hemisphere range vs analytic box: max err {h_range_err:.2e} m",
        )
        c.check(
            bool(np.all(h_alpha[h_rendered_hit] == 255)),
            f"{view_id} hemisphere alpha == 255 on hits",
        )
        c.check(bool(np.all(h_rgb[h_rendered_hit] > 0)), f"{view_id} hemisphere rgb > 0 on hits")
        miss_in_valid = valid_mask & ~h_rendered_hit
        c.check(
            bool(np.all(h_alpha[miss_in_valid] == 0)),
            f"{view_id} hemisphere alpha == 0 on misses (within valid_mask)",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mock_out", type=Path, help="MOCK_OUT directory of the mock pipeline")
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--num-bs", type=int, default=2)
    args = parser.parse_args()

    c = Checker(args.mock_out)
    check_scene_and_coverage(c, args.mock_out)
    check_rf_camera(c, args.mock_out / "rf_camera")
    check_multiview(c, args.mock_out / "rf_camera_multiview", args.num_views, args.num_bs)
    check_optical(c, args.mock_out / "rf_camera_multiview")

    if c.failures:
        print(f"\n❌ {len(c.failures)} check(s) failed:")
        for failure in c.failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\n✅ All mock pipeline outputs look sane")


if __name__ == "__main__":
    main()
