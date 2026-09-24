"""節目CI: モックE2Eパイプラインの生成物を検証する。

Makefile の *-mock ターゲットで生成した出力ディレクトリを受け取り、
ファイルの有無、配列の形状・有限性、LoSの角度/遅延の物理的な整合性を確認する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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


def check_multiview(c: Checker, mv: Path, num_views: int) -> None:
    print("--- 1BS/multi-UE RF camera dataset ---")
    if not c.exists(mv / "dataset_manifest.json"):
        return
    manifest = json.loads((mv / "dataset_manifest.json").read_text())
    config = manifest["config"]
    views = manifest["views"]
    c.check(manifest["schema_version"] == 2, f"schema_version: {manifest['schema_version']} == 2")
    c.check(len(views) == num_views, f"views: {len(views)} == {num_views}")
    hemispheres = manifest["raw_observation"]["hemispheres"]
    c.check(hemispheres == ["front", "back"], f"hemispheres: {hemispheres}")

    if c.exists(mv / "camera_model.npz"):
        model = np.load(mv / "camera_model.npz")
        expected = (config["fft_rows"], config["fft_cols"], 3)
        shape = model["ray_directions_local"].shape
        c.check(shape == expected, f"camera rays shape: {shape} == {expected}")
        weight_shape = model["solid_angle_weight"].shape
        c.check(weight_shape == expected[:2], f"solid-angle weight shape: {weight_shape}")
    c.exists(mv / "path_geometry_gt.npz")

    aperture_shape = (
        len(hemispheres),
        config["rx_rows"],
        config["rx_cols"],
        config["num_frequency_bins"],
    )
    for view in views:
        view_id = view["view_id"]
        for name, rel in view["artifacts"].items():
            c.exists(mv / rel)
        # 前面・背面の少なくとも一方にはエネルギーが届いている
        cfr = c.finite_nonzero(mv / view["artifacts"]["aperture_cfr"])
        shape_ok = cfr is not None and c.check(
            cfr.shape == aperture_shape, f"{view_id} aperture_cfr {cfr.shape}"
        )
        if shape_ok:
            energy = {h: float(np.sum(np.abs(cfr[i]) ** 2)) for i, h in enumerate(hemispheres)}
            recorded = view["hemisphere_energy"]
            c.check(
                all(np.isclose(energy[h], recorded[h], rtol=1e-4) for h in hemispheres),
                f"{view_id} hemisphere energy matches manifest",
            )
            # mock は直接波が支配的なので、エネルギーの大半は BS のある半球から届く
            # (前後の分割が入れ替わっていないことの確認)
            dominant = max(energy, key=energy.get)
            bs_side = "front" if view["bs_in_front_hemisphere"] else "back"
            c.check(dominant == bs_side, f"{view_id} dominant hemisphere {dominant} == {bs_side}")
        # 前面から何も届かない視点では現像画像が空になる (背面の光源と同じ扱い)
        c.finite(mv / view["artifacts"]["angular_power_center"])
        # 位相が有効でない画素は NaN で埋められる
        c.finite(mv / view["artifacts"]["dominant_delay_s"], allow_nan=True)


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
    optical_cfg = manifest["optical_reference"]
    pinhole_cfg = optical_cfg["pinhole"]
    width, height = int(pinhole_cfg["width"]), int(pinhole_cfg["height"])
    intrinsics = PinholeIntrinsics.from_horizontal_fov(width, height, pinhole_cfg["fov_x_deg"])
    pinhole_dirs_local = pinhole_ray_directions_local(intrinsics)

    transforms_path = mv / pinhole_cfg["transforms"]
    if not c.exists(transforms_path):
        return
    transforms = json.loads(transforms_path.read_text())
    views = manifest["views"]
    c.check(
        len(transforms["frames"]) == len(views),
        f"transforms.json: {len(transforms['frames'])} frames == {len(views)} views",
    )

    if not c.exists(mv / "camera_model.npz"):
        return
    camera_model = np.load(mv / "camera_model.npz")
    valid_mask = camera_model["valid_mask"]
    hemisphere_dirs_local = camera_model["ray_directions_local"]

    for view, frame in zip(views, transforms["frames"]):
        view_id = view["view_id"]
        artifacts = view["artifacts"]
        for name in (
            "optical_pinhole_rgba",
            "optical_pinhole_depth_m",
            "optical_pinhole_range_m",
            "optical_hemisphere_rgba",
            "optical_hemisphere_range_m",
        ):
            c.exists(mv / artifacts[name])

        pose = json.loads((mv / "views" / view_id / "pose.json").read_text())
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
        pinhole_rgba = np.asarray(Image.open(mv / artifacts["optical_pinhole_rgba"]))
        c.check(
            pinhole_rgba.shape == (height, width, 4),
            f"{view_id} pinhole rgba shape {pinhole_rgba.shape} == {(height, width, 4)}",
        )
        depth = np.load(mv / artifacts["optical_pinhole_depth_m"])
        c.check(depth.shape == (height, width), f"{view_id} pinhole depth shape {depth.shape}")
        range_m = np.load(mv / artifacts["optical_pinhole_range_m"])
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
        hemisphere_rgba_png = np.asarray(Image.open(mv / artifacts["optical_hemisphere_rgba"]))
        c.check(
            hemisphere_rgba_png.shape == valid_mask.shape + (4,),
            f"{view_id} hemisphere rgba shape {hemisphere_rgba_png.shape}",
        )
        hemisphere_range = np.load(mv / artifacts["optical_hemisphere_range_m"])
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
    args = parser.parse_args()

    c = Checker(args.mock_out)
    check_scene_and_coverage(c, args.mock_out)
    check_rf_camera(c, args.mock_out / "rf_camera")
    check_multiview(c, args.mock_out / "rf_camera_multiview", args.num_views)
    check_optical(c, args.mock_out / "rf_camera_multiview")

    if c.failures:
        print(f"\n❌ {len(c.failures)} check(s) failed:")
        for failure in c.failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\n✅ All mock pipeline outputs look sane")


if __name__ == "__main__":
    main()
