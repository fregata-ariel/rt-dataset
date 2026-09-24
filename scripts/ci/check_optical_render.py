"""節目CI: 光学レイレンダラーが mock ボックス形状と一致することを確認する。

mock シーン (x in [-5, 5]、y in [-5, 5]、z in [0, 10] の1棟) に対して
RayRenderer でワールド座標のレイを直接飛ばし、壁グリッド・屋根・ミス・
非正規化方向・シード決定性・混合バッチの6項目を検証する。地形は存在せず、
建物に当たらないレイは (hide_emitters の) 空へ抜ける。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mitsuba as mi
import numpy as np
from sionna.rt import load_scene

from plateau_rt.adapters.sionna.optical_render import RayRenderer

# 壁グリッド: x=35 の平面から -x 方向へ飛ばし +x 壁に range 30 で命中させる
WALL_X = 35.0
WALL_RANGE = 30.0
# 屋根: z=20 の平面から -z 方向へ飛ばし屋根 (z=10) に range 10 で命中させる
ROOF_Z = 20.0
ROOF_RANGE = 10.0
# 地面ポリゴン: 建物直下 (z=0) のみ存在する。下から飛ばすと range 5 で命中する
BELOW_RANGE = 5.0
# 幾何の許容誤差 (壁・屋根とも軸平行な面への垂直入射なので厳しめにできる)
RANGE_TOL = 1e-3
# 非正規化方向は正規化後に同じレイになるはずなので許容誤差は小さめでよい
UNNORMALISED_TOL = 1e-4
# 法線と期待方向の内積の下限
NORMAL_DOT_MIN = 0.999
SPP = 64
SEED = 0
# シードを変えたときの相対 RGB 変動の上限 (実測は最大で約 0.09)
SEED_REL_TOL = 0.2


def wall_rays() -> tuple[np.ndarray, np.ndarray]:
    """チェック1の壁グリッド用レイ (origins, directions) を作る。"""
    ys = np.linspace(-4.0, 4.0, 5)
    zs = np.linspace(1.0, 9.0, 5)
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    origins = np.stack([np.full(yy.size, WALL_X), yy.ravel(), zz.ravel()], axis=1)
    directions = np.zeros_like(origins)
    directions[:, 0] = -1.0
    return origins, directions


def raises_value_error(fn: object) -> bool:
    """fn() が ValueError で失敗すれば True、それ以外は False を返す。"""
    try:
        fn()  # type: ignore[operator]
    except ValueError:
        return True
    except Exception:
        return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_xml", type=Path, help="build-mock が生成した mock scene XML")
    args = parser.parse_args()

    scene = load_scene(str(args.scene_xml))
    # 後で「入力シーン不変」を確認するためのスナップショット (構築前の状態)
    attrs_before = set(dir(scene))
    variant_before = mi.variant()
    names_before = sorted(scene.objects.keys())
    verts_before = {
        name: np.array(np.asarray(obj.mi_mesh.vertex_positions_buffer()))
        for name, obj in scene.objects.items()
    }
    materials_before = {name: obj.radio_material.name for name, obj in scene.objects.items()}
    renderer = RayRenderer(scene)
    variant_after_init = mi.variant()

    failures = 0

    def report(ok: bool, message: str) -> None:
        nonlocal failures
        print(f"[{'OK' if ok else 'NG'}] {message}")
        if not ok:
            failures += 1

    # チェック1: 壁グリッド (5x5、spp=64、seed=0)
    origins, directions = wall_rays()
    wall = renderer.render(origins, directions, spp=SPP, seed=SEED)
    wall_range_err = float(np.max(np.abs(wall.range_m - WALL_RANGE)))
    wall_dot = float(np.min(wall.normal_world @ np.array([1.0, 0.0, 0.0])))
    wall_rgb_min = float(np.min(wall.rgb))
    ok = (
        bool(np.all(wall.hit))
        and wall_range_err <= RANGE_TOL
        and wall_dot > NORMAL_DOT_MIN
        and wall_rgb_min > 0.0
    )
    report(
        ok,
        f"check1 壁グリッド: 全命中={bool(np.all(wall.hit))}, "
        f"max|range-30|={wall_range_err:.2e}, min(n.x)={wall_dot:.5f}, "
        f"min(rgb)={wall_rgb_min:.4f}",
    )

    # チェック2: 屋根 (3点、真下方向)
    roof_origins = np.array([[0.0, 0.0, ROOF_Z], [3.0, -2.0, ROOF_Z], [-4.0, 4.0, ROOF_Z]])
    roof_directions = np.zeros_like(roof_origins)
    roof_directions[:, 2] = -1.0
    roof = renderer.render(roof_origins, roof_directions, spp=SPP, seed=SEED)
    roof_range_err = float(np.max(np.abs(roof.range_m - ROOF_RANGE)))
    roof_dot = float(np.min(roof.normal_world @ np.array([0.0, 0.0, 1.0])))
    ok = bool(np.all(roof.hit)) and roof_range_err <= RANGE_TOL and roof_dot > NORMAL_DOT_MIN
    report(
        ok,
        f"check2 屋根: 全命中={bool(np.all(roof.hit))}, "
        f"max|range-10|={roof_range_err:.2e}, min(n.z)={roof_dot:.5f}",
    )

    # チェック3: ミス (真上方向は空へ抜ける)
    miss = renderer.render(
        np.array([[WALL_X, 0.0, 5.0]]), np.array([[0.0, 0.0, 1.0]]), spp=SPP, seed=SEED
    )
    ok = (
        not bool(miss.hit[0])
        and bool(np.isnan(miss.range_m[0]))
        and bool(np.all(np.isnan(miss.normal_world[0])))
        and bool(np.all(miss.rgb == 0.0))
    )
    report(
        ok,
        f"check3 ミス: hit={bool(miss.hit[0])}, range=nanか={bool(np.isnan(miss.range_m[0]))}, "
        f"normal全nanか={bool(np.all(np.isnan(miss.normal_world[0])))}, "
        f"rgb全0か={bool(np.all(miss.rgb == 0.0))}",
    )

    # チェック4: 非正規化方向 (長さ 0.5〜7.5 倍) でも同じ結果になる
    factors = np.linspace(0.5, 7.5, origins.shape[0])[:, None]
    scaled = renderer.render(origins, directions * factors, spp=SPP, seed=SEED)
    scaled_range_err = float(np.max(np.abs(scaled.range_m - wall.range_m)))
    ok = scaled_range_err <= UNNORMALISED_TOL and bool(np.array_equal(scaled.hit, wall.hit))
    report(
        ok,
        f"check4 非正規化方向: max|range差|={scaled_range_err:.2e}, "
        f"hit一致={bool(np.array_equal(scaled.hit, wall.hit))}",
    )

    # チェック5: シード決定性 (同一シードはビット一致、別シードは近くて異なる)
    wall_again = renderer.render(origins, directions, spp=SPP, seed=SEED)
    identical = bool(np.array_equal(wall_again.rgb, wall.rgb))
    wall_seed1 = renderer.render(origins, directions, spp=SPP, seed=1)
    differs = not bool(np.array_equal(wall_seed1.rgb, wall.rgb))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(wall_seed1.rgb - wall.rgb) / wall.rgb
    max_rel = float(np.max(rel[np.isfinite(rel)]))
    ok = identical and differs and max_rel < SEED_REL_TOL
    report(
        ok,
        f"check5 シード: 同一シード一致={identical}, 別シード相違={differs}, "
        f"max相対差={max_rel:.4f}",
    )

    # チェック6a: 混合バッチ (壁グリッド + ミス1 + 下からの地面1)。法線の向きも見る
    n_wall = origins.shape[0]  # 壁レイ数 (グリッド変更時にずれないよう名前で保持)
    i_miss = n_wall  # ミスレイの添字
    i_ground = n_wall + 1  # 下からの地面レイの添字
    mixed_origins = np.concatenate(
        [origins, np.array([[WALL_X, 0.0, 5.0], [0.0, 0.0, -5.0]])], axis=0
    )
    mixed_directions = np.concatenate(
        [directions, np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])], axis=0
    )
    mixed = renderer.render(mixed_origins, mixed_directions, spp=SPP, seed=SEED)
    mixed_wall_err = float(np.max(np.abs(mixed.range_m[:n_wall] - WALL_RANGE)))
    mixed_wall_dot = float(np.min(mixed.normal_world[:n_wall] @ np.array([1.0, 0.0, 0.0])))
    miss_normal_nan = bool(np.all(np.isnan(mixed.normal_world[i_miss])))
    miss_rgb_zero = bool(np.all(mixed.rgb[i_miss] == 0.0))
    wall_rgb_min = float(np.min(mixed.rgb[:n_wall]))
    wall_rgb_all_pos = bool(np.all(mixed.rgb[:n_wall] > 0.0))
    ground_range_err = float(abs(mixed.range_m[i_ground] - BELOW_RANGE))
    ground_dot = float(mixed.normal_world[i_ground] @ np.array([0.0, 0.0, -1.0]))
    # 地面レイは間接光のみで照らされるため rgb > 0 は要求せず、有限・非負のみ見る
    ground_rgb_finite = bool(np.all(np.isfinite(mixed.rgb[i_ground])))
    ground_rgb_nonneg = bool(np.all(mixed.rgb[i_ground] >= 0.0))
    ground_rgb_min = float(np.min(mixed.rgb[i_ground]))
    ok = (
        bool(np.all(mixed.hit[:n_wall]))
        and mixed_wall_err <= RANGE_TOL
        and mixed_wall_dot > NORMAL_DOT_MIN
        and wall_rgb_all_pos
        and not bool(mixed.hit[i_miss])
        and bool(np.isnan(mixed.range_m[i_miss]))
        and miss_normal_nan
        and miss_rgb_zero
        and bool(mixed.hit[i_ground])
        and ground_range_err <= RANGE_TOL
        and ground_dot > NORMAL_DOT_MIN
        and ground_rgb_finite
        and ground_rgb_nonneg
    )
    report(
        ok,
        f"check6a 混合バッチ: 壁max|range-30|={mixed_wall_err:.2e}, "
        f"壁min(n.x)={mixed_wall_dot:.5f}, 壁min(rgb)={wall_rgb_min:.4f}, "
        f"ミスhit={bool(mixed.hit[i_miss])}, ミスnormal全nanか={miss_normal_nan}, "
        f"ミスrgb全0か={miss_rgb_zero}, "
        f"地面|range-5|={ground_range_err:.2e}, 地面n.(-z)={ground_dot:.5f}, "
        f"地面min(rgb)={ground_rgb_min:.4f}",
    )

    # チェック6c: N=1 の形状・dtype
    single = renderer.render(origins[:1], directions[:1], spp=SPP, seed=SEED)
    ok = (
        single.rgb.shape == (1, 3)
        and single.rgb.dtype == np.float32
        and single.rgb.flags["C_CONTIGUOUS"]
        and single.hit.shape == (1,)
        and single.hit.dtype == np.bool_
        and single.range_m.shape == (1,)
        and single.range_m.dtype == np.float32
        and single.normal_world.shape == (1, 3)
        and single.normal_world.dtype == np.float32
    )
    report(
        ok,
        f"check6c N=1: rgb{single.rgb.shape}/{single.rgb.dtype}, hit{single.hit.shape}/"
        f"{single.hit.dtype}, range{single.range_m.shape}/{single.range_m.dtype}, "
        f"normal{single.normal_world.shape}/{single.normal_world.dtype}",
    )

    # チェック6d: 不正入力は ValueError になる
    bad_shape = raises_value_error(lambda: renderer.render(np.zeros((4, 2)), np.zeros((4, 2))))
    bad_spp = raises_value_error(lambda: renderer.render(origins[:1], directions[:1], spp=0))
    ok = bad_shape and bad_spp
    report(ok, f"check6d 不正入力: 形状[N,2]がValueError={bad_shape}, spp=0がValueError={bad_spp}")

    # チェック6b: 入力シーンが変更されていない (全レンダー後に比較する)
    # 属性・メッシュ頂点・マテリアル・variant を構築前のスナップショットと比べる
    attrs_after = set(dir(scene))
    added = sorted(attrs_after - attrs_before)
    removed = sorted(attrs_before - attrs_after)
    names_after = sorted(scene.objects.keys())
    names_same = names_after == names_before
    verts_same = names_same and all(
        np.array_equal(
            np.asarray(scene.objects[name].mi_mesh.vertex_positions_buffer()),
            verts_before[name],
        )
        for name in names_before
    )
    materials_after = {name: obj.radio_material.name for name, obj in scene.objects.items()}
    materials_same = materials_after == materials_before
    variant_after = mi.variant()
    variant_same = variant_after_init == variant_before and variant_after == variant_before
    variant_expected = variant_before == "cuda_ad_mono_polarized"
    ok = (
        not added
        and not removed
        and names_same
        and verts_same
        and materials_same
        and variant_same
        and variant_expected
    )
    report(
        ok,
        f"check6b 入力シーン不変: 追加={added}, 削除={removed}, "
        f"オブジェクト名一致={names_same}, 頂点一致={verts_same}, "
        f"マテリアル一致={materials_same}({materials_after}), "
        f"variant={variant_before}->{variant_after_init}->{variant_after}",
    )

    if failures:
        sys.exit(1)
    print("✅ Optical ray renderer matches the mock box geometry")


if __name__ == "__main__":
    main()
