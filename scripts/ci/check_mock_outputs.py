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

# 8x8 開口 / 128点FFTの角度ビン幅は 2/128 ≈ 0.016。LoS推定の許容誤差は約3ビンとする
MAX_LOS_YZ_PROJECTION_ERROR = 0.05


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mock_out", type=Path, help="MOCK_OUT directory of the mock pipeline")
    parser.add_argument("--num-views", type=int, default=8)
    args = parser.parse_args()

    c = Checker(args.mock_out)
    check_scene_and_coverage(c, args.mock_out)
    check_rf_camera(c, args.mock_out / "rf_camera")
    check_multiview(c, args.mock_out / "rf_camera_multiview", args.num_views)

    if c.failures:
        print(f"\n❌ {len(c.failures)} check(s) failed:")
        for failure in c.failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\n✅ All mock pipeline outputs look sane")


if __name__ == "__main__":
    main()
