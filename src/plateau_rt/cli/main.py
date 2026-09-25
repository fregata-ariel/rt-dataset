from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from plateau_rt.application.build_scene import SceneBuilder
from plateau_rt.domain.rf_camera.camera import RFViewSpec
from plateau_rt.domain.rf_camera.placement import (
    AGGREGATIONS,
    ORIENTATION_POLICIES,
    THRESHOLD_MODES,
)


@click.group()
def cli():
    """PLATEAU CityJSON to Sionna-RT Dataset Generator"""
    pass


@cli.command("build")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--ground-plane-size-m",
    type=click.FloatRange(min=0.0),
    default=0.0,
    show_default=True,
    help=(
        "Side length in metres of an optional square ground plane centred on "
        "the scene origin at z = -0.01 m. 0 disables the plane. "
        "Its material itu_medium_dry_ground is only defined for carriers from 1 to 10 GHz."
    ),
)
def build_scene(input_file: Path, output_dir: Path, ground_plane_size_m: float):
    """Step 1: CityJSONからSionna-RT用シーン(PLY/XML)とマニフェストを生成します。"""
    click.echo(f"Building scene from {input_file} into {output_dir}...")
    builder = SceneBuilder(input_file, output_dir, ground_plane_size_m=ground_plane_size_m)
    xml_path = builder.run()
    click.echo(click.style(f"Success! Scene XML generated at: {xml_path}", fg="green"))


@cli.command("simulate")
@click.argument("xml_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("manifest_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
def simulate_coverage(xml_file: Path, manifest_file: Path, output_dir: Path):
    """Step 2: 生成されたXMLとマニフェストを用いて電波カバレッジマップを計算します。"""
    from plateau_rt.adapters.sionna.simulator import SionnaSimulator

    click.echo(f"Running simulation for {xml_file}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    simulator = SionnaSimulator(xml_file, manifest_file)
    result_path = simulator.run_coverage_simulation(output_dir)
    click.echo(click.style(f"Success! Coverage map generated at: {result_path}", fg="green"))


@cli.command("rf-camera")
@click.argument("xml_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--ue-position", nargs=3, type=float, default=(0.0, 0.0, 1.5), show_default=True)
@click.option("--ue-orientation", nargs=3, type=float, default=(0.0, 0.0, 0.0), show_default=True)
@click.option("--bs-position", nargs=3, type=float, default=(-50.0, -50.0, 30.0), show_default=True)
@click.option("--rx-rows", type=int, default=8, show_default=True)
@click.option("--rx-cols", type=int, default=8, show_default=True)
@click.option("--carrier-ghz", type=float, default=3.5, show_default=True)
@click.option("--bandwidth-mhz", type=float, default=100.0, show_default=True)
@click.option("--frequency-bins", type=int, default=64, show_default=True)
@click.option("--max-depth", type=int, default=5, show_default=True)
@click.option("--synthetic-array/--explicit-array", default=True, show_default=True)
def rf_camera(
    xml_file: Path,
    output_dir: Path,
    ue_position: tuple[float, float, float],
    ue_orientation: tuple[float, float, float],
    bs_position: tuple[float, float, float],
    rx_rows: int,
    rx_cols: int,
    carrier_ghz: float,
    bandwidth_mhz: float,
    frequency_bins: int,
    max_depth: int,
    synthetic_array: bool,
):
    """1 BS / 1 UEの複素RFカメラ画像を生成します。

    Raw aperture CFRと、2-D spatial FFTによる最初のangular-spectrum画像を出力します。
    MVPではBS側は1 active antenna/port、UE側は既定で8x8 planar apertureです。
    """
    from plateau_rt.application.scene_checks import check_scene_carrier_frequency

    try:
        check_scene_carrier_frequency(xml_file, carrier_ghz * 1e9)
    except ValueError as err:
        raise click.BadParameter(str(err), param_hint="--carrier-ghz") from None
    from plateau_rt.adapters.sionna.rf_camera import RFCameraConfig, RFCameraMVP

    config = RFCameraConfig(
        carrier_frequency_hz=carrier_ghz * 1e9,
        bandwidth_hz=bandwidth_mhz * 1e6,
        num_frequency_bins=frequency_bins,
        tx_position=tuple(bs_position),
        ue_position=tuple(ue_position),
        ue_orientation=tuple(ue_orientation),
        rx_rows=rx_rows,
        rx_cols=rx_cols,
        max_depth=max_depth,
        synthetic_array=synthetic_array,
    )

    click.echo(click.style("=== 1 BS / 1 UE RF Camera MVP ===", fg="cyan", bold=True))
    artifacts = RFCameraMVP(xml_file, config).run(output_dir)
    click.echo(click.style("RF camera development complete", fg="green", bold=True))
    click.echo(f"  aperture CFR : {artifacts.aperture_cfr}")
    click.echo(f"  angular CFR  : {artifacts.angular_cfr}")
    click.echo(f"  power image  : {artifacts.power_png}")
    click.echo(f"  phase image  : {artifacts.phase_png}")
    click.echo(f"  path GT      : {artifacts.path_gt}")
    click.echo(f"  metadata     : {artifacts.metadata}")


@cli.command("rf-camera-calibrate")
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--phase-floor-db",
    type=float,
    default=-35.0,
    show_default=True,
    help="Mask phase below this power level relative to the peak",
)
def rf_camera_calibrate(output_dir: Path, phase_floor_db: float):
    """rf-camera の出力ディレクトリの角度FFTを物理座標に校正します (GPU不要)。"""
    from plateau_rt.application.rf_camera_calibration import calibrate_directory

    calibrate_directory(output_dir, phase_floor_db=phase_floor_db)


@cli.command("rf-camera-delay")
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--power-floor-db",
    type=float,
    default=-35.0,
    show_default=True,
    help="Mask dominant-delay visualization below this relative peak power",
)
def rf_camera_delay(output_dir: Path, power_floor_db: float):
    """校正済みRFカメラ出力を角度-遅延ボリュームに展開します (GPU不要)。"""
    from plateau_rt.application.rf_camera_delay import develop_angle_delay

    develop_angle_delay(output_dir, power_floor_db=power_floor_db)


def _plan_coverage_views(
    *,
    xml_file: Path,
    output_dir: Path,
    config: Any,
    num_views: int,
    placement_seed: int,
    threshold_mode: str,
    threshold_value: float,
    aggregation: str,
    orientation_policy: str,
    face_bs: int | str,
    pitch_deg: float,
    building_clearance_m: float,
    min_bs_distance_m: float,
    min_ue_spacing_m: float,
    jitter_fraction: float,
    rm_center: tuple[float, float] | None,
    rm_size: tuple[float, float],
    rm_cell_size: tuple[float, float],
    rm_max_depth: int,
    rm_samples_per_tx: int,
    rm_seed: int,
    radio_map: Path | None,
    ue_height_m: float,
    target: tuple[float, float, float],
) -> tuple[list[RFViewSpec], dict[str, Any]]:
    """Plan coverage-map UE views and build the manifest ``placement`` section.

    The Sionna adapter is imported lazily. When ``--radio-map`` is given the
    saved map is reused (the ``--rm-*`` options are ignored); otherwise a radio
    map is computed and saved under ``output_dir/placement/``. Both branches
    then reload the saved float32 arrays, so the NumPy placement is identical.
    """
    from plateau_rt.adapters.sionna.radio_map import (
        RadioMapSolverSettings,
        compute_radio_map,
    )
    from plateau_rt.application.ue_placement import (
        building_exclusion_mask,
        check_radio_map_matches,
        copy_radio_map,
        load_radio_map,
        placement_manifest_section,
        save_radio_map,
    )
    from plateau_rt.domain.rf_camera.placement import (
        CoveragePlacementSettings,
        CoverageThreshold,
        RadioMapGrid,
        plan_coverage_placement,
    )

    base_stations = config.resolve_base_stations()
    solver = RadioMapSolverSettings(
        max_depth=rm_max_depth,
        samples_per_tx=rm_samples_per_tx,
        seed=rm_seed,
    )
    if radio_map is not None:
        saved = load_radio_map(radio_map)
        check_radio_map_matches(
            saved,
            carrier_frequency_hz=config.carrier_frequency_hz,
            base_stations=base_stations,
            ue_height_m=ue_height_m,
        )
        recorded_scene = str(saved.metadata.get("source_scene", ""))
        if Path(recorded_scene).resolve() != Path(xml_file).resolve():
            click.echo(
                f"warning: the saved radio map was computed on {recorded_scene!r}, "
                f"not on {str(xml_file)!r}; only carrier, base stations and UE height "
                "are checked",
                err=True,
            )
        metadata_path = copy_radio_map(saved, output_dir)
        radio_map_source = "loaded"
        radio_map_origin: str | None = str(radio_map)
        click.echo("using the saved radio-map grid/solver settings (--rm-* are ignored)")
    else:
        center = (
            (float(target[0]), float(target[1]))
            if rm_center is None
            else (float(rm_center[0]), float(rm_center[1]))
        )
        grid = RadioMapGrid(
            center_m=(center[0], center[1], float(ue_height_m)),
            size_m=(float(rm_size[0]), float(rm_size[1])),
            cell_size_m=(float(rm_cell_size[0]), float(rm_cell_size[1])),
        )
        result = compute_radio_map(xml_file, dataset_config=config, grid=grid, solver=solver)
        metadata_path = save_radio_map(
            output_dir,
            path_gain=result.path_gain,
            indoor_mask=result.indoor_mask,
            grid=result.grid,
            solver=solver.to_dict(),
            base_stations=[
                {"bs_id": bs_id, "position_m": list(position), "look_at_m": list(look_at)}
                for bs_id, position, look_at in base_stations
            ],
            carrier_frequency_hz=config.carrier_frequency_hz,
            source_scene=str(xml_file),
        )
        radio_map_source = "computed"
        radio_map_origin = None

    saved = load_radio_map(metadata_path)
    settings = CoveragePlacementSettings(
        num_views=num_views,
        placement_seed=placement_seed,
        threshold=CoverageThreshold(mode=threshold_mode, value=threshold_value),
        aggregation=aggregation,
        min_bs_distance_m=min_bs_distance_m,
        min_spacing_m=min_ue_spacing_m,
        jitter_fraction=jitter_fraction,
        orientation_policy=orientation_policy,
        face_bs=face_bs,
        target=tuple(target),
        pitch_deg=pitch_deg,
    )
    exclusion = building_exclusion_mask(
        saved.indoor_mask, saved.grid, clearance_m=building_clearance_m
    )
    placement = plan_coverage_placement(
        saved.path_gain,
        saved.grid,
        settings,
        exclusion_mask=exclusion,
        bs_positions=[position for _, position, _ in base_stations],
    )
    section = placement_manifest_section(
        placement,
        saved=saved,
        dataset_dir=output_dir,
        radio_map_source=radio_map_source,
        radio_map_origin=radio_map_origin,
        building_clearance_m=building_clearance_m,
    )
    click.echo(
        f"coverage candidates={placement.candidates.count} "
        f"threshold_db={list(placement.candidates.threshold_db)}"
    )
    for row, view in enumerate(placement.views):
        chosen = int(placement.sampled.candidate_index[row])
        iy, ix = (int(v) for v in placement.candidates.indices[chosen])
        click.echo(
            f"  [{row + 1:02d}/{len(placement.views):02d}] {view.view_id}: "
            f"cell=({iy},{ix}) position={view.position} "
            f"gain_db={float(placement.candidates.gain_db[chosen]):.2f} "
            f"facing_bs={placement.facing_bs_index[row]}"
        )
    return placement.views, section


@cli.command("rf-camera-multiview")
@click.argument("xml_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--num-views", type=int, default=8, show_default=True)
@click.option("--radius-m", type=float, default=30.0, show_default=True)
@click.option("--ue-height-m", type=float, default=1.5, show_default=True)
@click.option("--target", nargs=3, type=float, default=(5.0, 5.0, 5.0), show_default=True)
@click.option(
    "--bs-position",
    multiple=True,
    nargs=3,
    type=float,
    default=[(-50.0, -50.0, 30.0)],
    show_default=True,
)
@click.option("--bs-look-at", multiple=True, nargs=3, type=float, default=None)
@click.option("--rx-rows", type=int, default=8, show_default=True)
@click.option("--rx-cols", type=int, default=8, show_default=True)
@click.option("--carrier-ghz", type=float, default=3.5, show_default=True)
@click.option("--bandwidth-mhz", type=float, default=100.0, show_default=True)
@click.option("--frequency-bins", type=int, default=64, show_default=True)
@click.option("--max-depth", type=int, default=5, show_default=True)
@click.option("--synthetic-array/--explicit-array", default=True, show_default=True)
@click.option(
    "--tx-power-dbm",
    type=float,
    default=44.0,
    show_default=True,
    help="BS transmit power [dBm]; recorded in the manifest, not applied to the stored CFR",
)
@click.option(
    "--placement",
    type=click.Choice(["ring", "coverage"]),
    default="ring",
    show_default=True,
    help="UE placement method (ring or coverage-map based)",
)
@click.option(
    "--placement-seed",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Dedicated coverage placement seed (coverage only)",
)
@click.option(
    "--pl-threshold-mode",
    type=click.Choice(THRESHOLD_MODES),
    default="relative_to_max_db",
    show_default=True,
    help="Path-gain threshold reference mode (coverage only)",
)
@click.option(
    "--pl-threshold",
    type=float,
    default=30.0,
    show_default=True,
    help="Threshold in dB (absolute / below max) or percentile (coverage only)",
)
@click.option(
    "--bs-aggregation",
    type=click.Choice(AGGREGATIONS),
    default="max",
    show_default=True,
    help="Multi-BS path-gain aggregation (coverage only)",
)
@click.option(
    "--orientation-policy",
    type=click.Choice(ORIENTATION_POLICIES),
    default="face_bs",
    show_default=True,
    help="UE orientation policy (coverage only)",
)
@click.option(
    "--face-bs",
    type=str,
    default="strongest",
    show_default=True,
    help="BS index or 'strongest' for --orientation-policy face_bs (coverage only)",
)
@click.option(
    "--pitch-deg",
    type=float,
    default=0.0,
    show_default=True,
    help="Forward elevation for --orientation-policy random_yaw (coverage only)",
)
@click.option(
    "--building-clearance-m",
    type=click.FloatRange(min=0.0),
    default=1.0,
    show_default=True,
    help="Dilation of the indoor mask [m] (coverage only)",
)
@click.option(
    "--min-bs-distance-m",
    type=click.FloatRange(min=0.0),
    default=5.0,
    show_default=True,
    help="Minimum 3D distance from a UE to any BS [m] (coverage only)",
)
@click.option(
    "--min-ue-spacing-m",
    type=click.FloatRange(min=0.0),
    default=2.0,
    show_default=True,
    help="Minimum horizontal spacing between UEs [m] (coverage only)",
)
@click.option(
    "--cell-jitter",
    type=click.FloatRange(min=0.0, max=1.0, max_open=True),
    default=0.0,
    show_default=True,
    help="Intra-cell jitter fraction (coverage only)",
)
@click.option(
    "--rm-center",
    nargs=2,
    type=float,
    default=None,
    help="Radio-map centre x y [m]; defaults to the target x y (coverage only)",
)
@click.option(
    "--rm-size",
    nargs=2,
    type=float,
    default=(100.0, 100.0),
    show_default=True,
    help="Radio-map size x y [m] (coverage only)",
)
@click.option(
    "--rm-cell-size",
    nargs=2,
    type=float,
    default=(1.0, 1.0),
    show_default=True,
    help="Radio-map cell size x y [m] (coverage only)",
)
@click.option(
    "--rm-max-depth",
    type=int,
    default=5,
    show_default=True,
    help="RadioMapSolver max_depth (coverage only)",
)
@click.option(
    "--rm-samples-per-tx",
    type=int,
    default=1_000_000,
    show_default=True,
    help="RadioMapSolver samples_per_tx (coverage only)",
)
@click.option(
    "--rm-seed",
    type=int,
    default=42,
    show_default=True,
    help="RadioMapSolver seed, not the placement seed (coverage only)",
)
@click.option(
    "--radio-map",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=Path),
    default=None,
    help="Reuse a saved radio map (json, placement dir or dataset dir)",
)
def rf_camera_multiview(
    xml_file: Path,
    output_dir: Path,
    num_views: int,
    radius_m: float,
    ue_height_m: float,
    target: tuple[float, float, float],
    bs_position: tuple[tuple[float, float, float], ...],
    bs_look_at: tuple[tuple[float, float, float], ...],
    rx_rows: int,
    rx_cols: int,
    carrier_ghz: float,
    bandwidth_mhz: float,
    frequency_bins: int,
    max_depth: int,
    synthetic_array: bool,
    tx_power_dbm: float,
    placement: str,
    placement_seed: int,
    pl_threshold_mode: str,
    pl_threshold: float,
    bs_aggregation: str,
    orientation_policy: str,
    face_bs: str,
    pitch_deg: float,
    building_clearance_m: float,
    min_bs_distance_m: float,
    min_ue_spacing_m: float,
    cell_jitter: float,
    rm_center: tuple[float, float] | None,
    rm_size: tuple[float, float],
    rm_cell_size: tuple[float, float],
    rm_max_depth: int,
    rm_samples_per_tx: int,
    rm_seed: int,
    radio_map: Path | None,
):
    """複数BS / multi-UEのRFカメラデータセットを生成します。

    targetを中心とする半径radius-mのリング上にUE(RFカメラ)を配置し、
    全BSと全UEを1回のPathSolver呼び出しでトレースします。--bs-positionを
    繰り返すと複数のBSを配置でき、各BSは既定でtargetを向きます
    (--bs-look-atで個別の注視点も指定可能)。

    --placement coverage ではUE高さの2Dパスゲイン(ラジオマップ)を計算し、
    しきい値を超える建物外のセルから --placement-seed でUE姿勢を抽選します。
    ラジオマップは placement/ に保存され、--radio-map で再利用すると
    「保存マップ+seed」から同一の姿勢を再現できます。
    """
    from plateau_rt.application.scene_checks import check_scene_carrier_frequency

    try:
        check_scene_carrier_frequency(xml_file, carrier_ghz * 1e9)
    except ValueError as err:
        raise click.BadParameter(str(err), param_hint="--carrier-ghz") from None
    from plateau_rt.adapters.sionna.rf_camera_dataset import (
        RFMultiViewConfig,
        RFMultiViewDataset,
    )
    from plateau_rt.domain.rf_camera.camera import generate_ring_views

    target = tuple(target)
    if bs_look_at and len(bs_look_at) != len(bs_position):
        raise click.BadParameter(
            f"--bs-look-at count ({len(bs_look_at)}) must match "
            f"--bs-position count ({len(bs_position)})"
        )
    config = RFMultiViewConfig(
        carrier_frequency_hz=carrier_ghz * 1e9,
        bandwidth_hz=bandwidth_mhz * 1e6,
        num_frequency_bins=frequency_bins,
        tx_positions=tuple(tuple(p) for p in bs_position),
        tx_look_at=target,
        tx_look_ats=tuple(tuple(p) for p in bs_look_at) if bs_look_at else None,
        rx_rows=rx_rows,
        rx_cols=rx_cols,
        max_depth=max_depth,
        synthetic_array=synthetic_array,
        tx_power_dbm=tx_power_dbm,
    )

    if placement == "ring":
        if radio_map is not None:
            raise click.UsageError("--radio-map requires --placement coverage")
        views = generate_ring_views(
            target=target,
            radius_m=radius_m,
            ue_height_m=ue_height_m,
            num_views=num_views,
        )
        RFMultiViewDataset(xml_file, views=views, config=config).run(output_dir)
        return

    if face_bs.isdigit():
        face_bs_value: int | str = int(face_bs)
    elif face_bs == "strongest":
        face_bs_value = "strongest"
    else:
        raise click.BadParameter(
            "must be a non-negative BS index or 'strongest'", param_hint="--face-bs"
        )
    try:
        views, section = _plan_coverage_views(
            xml_file=xml_file,
            output_dir=output_dir,
            config=config,
            num_views=num_views,
            placement_seed=placement_seed,
            threshold_mode=pl_threshold_mode,
            threshold_value=pl_threshold,
            aggregation=bs_aggregation,
            orientation_policy=orientation_policy,
            face_bs=face_bs_value,
            pitch_deg=pitch_deg,
            building_clearance_m=building_clearance_m,
            min_bs_distance_m=min_bs_distance_m,
            min_ue_spacing_m=min_ue_spacing_m,
            jitter_fraction=cell_jitter,
            rm_center=rm_center,
            rm_size=rm_size,
            rm_cell_size=rm_cell_size,
            rm_max_depth=rm_max_depth,
            rm_samples_per_tx=rm_samples_per_tx,
            rm_seed=rm_seed,
            radio_map=radio_map,
            ue_height_m=ue_height_m,
            target=target,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    RFMultiViewDataset(xml_file, views=views, config=config, placement=section).run(output_dir)


@cli.command("rf-camera-optical")
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--scene-xml",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="既定ではマニフェストの source_scene を使用",
)
@click.option("--width", type=int, default=512, show_default=True)
@click.option("--height", type=int, default=512, show_default=True)
@click.option("--fov-x-deg", type=float, default=90.0, show_default=True)
@click.option("--spp", type=int, default=64, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def rf_camera_optical(
    dataset_dir: Path,
    scene_xml: Path | None,
    width: int,
    height: int,
    fov_x_deg: float,
    spp: int,
    seed: int,
):
    """rf-camera-multiview の出力に位置合わせ済みの光学参照レンダーを追加します。"""
    from plateau_rt.application.optical_reference import render_optical_references
    from plateau_rt.application.rf_dataset_manifest import ManifestError

    click.echo(click.style("=== Optical reference renders ===", fg="cyan", bold=True))
    try:
        transforms_path = render_optical_references(
            dataset_dir,
            scene_xml=scene_xml,
            width=width,
            height=height,
            fov_x_deg=fov_x_deg,
            spp=spp,
            seed=seed,
        )
    except ManifestError as exc:
        # 非対応スキーマや壊れたマニフェストは描画前にエラー終了する
        raise click.ClickException(f"{dataset_dir}/dataset_manifest.json: {exc}") from exc
    click.echo(click.style(f"Success! transforms written to: {transforms_path}", fg="green"))


@cli.command("rf-camera-observe")
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--front-to-back-db", type=float, default=None, help="正面/背面比 [dB] (未指定で理想)"
)
@click.option("--common-phase-deg", type=float, default=0.0, show_default=True)
@click.option("--random-common-phase/--fixed-common-phase", default=False)
@click.option("--timing-offset-ns", type=float, default=0.0, show_default=True)
@click.option("--timing-offset-std-ns", type=float, default=0.0, show_default=True)
@click.option("--element-gain-std-db", type=float, default=0.0, show_default=True)
@click.option("--element-phase-std-deg", type=float, default=0.0, show_default=True)
@click.option(
    "--snr-db",
    type=float,
    default=None,
    help=(
        "データセット全体の参照電力(全(view, BS)ペアの理想等方平均電力の最大値)に対する"
        " SNR [dB] (未指定で雑音なし)"
    ),
)
@click.option(
    "--noise-variance",
    type=float,
    default=None,
    help="素子・ビンあたりの絶対複素雑音分散 (--snr-dbとは排他)",
)
@click.option(
    "--name",
    default="observed",
    show_default=True,
    help="観測バリアント名 (既存の同名バリアントは上書き)",
)
@click.option("--seed", type=click.IntRange(0, 2**32 - 1), default=0, show_default=True)
def rf_camera_observe(
    dataset_dir: Path,
    front_to_back_db: float | None,
    common_phase_deg: float,
    random_common_phase: bool,
    timing_offset_ns: float,
    timing_offset_std_ns: float,
    element_gain_std_db: float,
    element_phase_std_deg: float,
    snr_db: float | None,
    noise_variance: float | None,
    name: str,
    seed: int,
):
    """マルチビューRFカメラデータセットに受信機劣化を付与し、単一チャネル観測を生成します。"""
    import json

    from plateau_rt.application.rf_camera_observe import observe_dataset
    from plateau_rt.application.rf_dataset_manifest import ManifestError
    from plateau_rt.domain.rf_camera.impairments import ImpairmentConfig, NoiseSpec

    if random_common_phase and common_phase_deg != 0.0:
        raise click.UsageError("--random-common-phase は --common-phase-deg と同時に指定できません")
    if snr_db is not None and noise_variance is not None:
        raise click.UsageError("--snr-db と --noise-variance は同時に指定できません")

    try:
        config = ImpairmentConfig(
            front_to_back_db=front_to_back_db,
            common_phase_deg=common_phase_deg,
            random_common_phase=random_common_phase,
            timing_offset_ns=timing_offset_ns,
            timing_offset_std_ns=timing_offset_std_ns,
            element_gain_std_db=element_gain_std_db,
            element_phase_std_deg=element_phase_std_deg,
        )
        noise = NoiseSpec(snr_db=snr_db, noise_variance=noise_variance)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    try:
        manifest_path = observe_dataset(dataset_dir, config, noise=noise, seed=seed, name=name)
    except (ManifestError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    section = json.loads(manifest_path.read_text(encoding="utf-8"))["observations"][name]
    noise_section = section["noise"]
    if noise_section["mode"] == "none":
        click.echo("noise: none")
    else:
        reference = noise_section["reference_pair"]
        reference_text = (
            "none" if reference is None else f"{reference['view_id']}/{reference['bs_id']}"
        )
        click.echo(
            f"noise: mode={noise_section['mode']} reference_power="
            f"{noise_section['reference_power']:.6g} reference_pair={reference_text} "
            f"noise_variance={noise_section['noise_variance']:.6g}"
        )
    for pair in section["pairs"]:
        expected = pair["expected_snr_db"]
        achieved = pair["achieved_snr_db"]
        if noise_section["noise_variance"] == 0.0:
            status = "no noise"
        elif pair["signal_power"] == 0.0 or expected is None:
            status = "zero signal"
        else:
            achieved_text = "n/a" if achieved is None else f"{achieved:.3f}"
            status = f"expected_snr_db={expected:.3f} achieved_snr_db={achieved_text}"
        click.echo(f"{pair['view_id']} {pair['bs_id']} {status}")

    click.echo(click.style(f"Observed dataset manifest: {manifest_path}", fg="green"))


@cli.command("rf-camera-partial")
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--view-fraction", type=float, default=1.0, show_default=True)
@click.option(
    "--element-mask",
    "element_mask_kind",
    type=click.Choice(["none", "random", "every_other_row", "every_other_col", "checkerboard"]),
    default="none",
    show_default=True,
)
@click.option("--mask-fraction", type=float, default=0.5, show_default=True)
@click.option("--subband", type=str, default=None, show_default=True)
@click.option(
    "--summary", type=click.Choice(["none", "power", "delay"]), default="none", show_default=True
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="既存の部分出力 (partial_manifest.json / element_mask.npy / views) を置き換える",
)
def rf_camera_partial(
    dataset_dir: Path,
    out_dir: Path,
    view_fraction: float,
    element_mask_kind: str,
    mask_fraction: float,
    subband: str | None,
    summary: str,
    seed: int,
    overwrite: bool,
):
    """マルチビューRFカメラデータセットから部分/要約観測データセットを生成します。"""
    from plateau_rt.application.rf_camera_partial import build_partial_dataset
    from plateau_rt.application.rf_dataset_manifest import ManifestError

    try:
        manifest_path = build_partial_dataset(
            dataset_dir,
            out_dir,
            view_fraction=view_fraction,
            element_mask_kind=element_mask_kind,
            mask_fraction=mask_fraction,
            subband=subband,
            summary=summary,
            seed=seed,
            overwrite=overwrite,
        )
    except (ManifestError, ValueError, FileExistsError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(click.style(f"Success! Partial manifest generated at: {manifest_path}", fg="green"))


@cli.command("render")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def render_heatmaps(input_dir: Path):
    """シミュレーション結果(npy)から2Dヒートマップ画像群を生成します。"""
    import numpy as np

    from plateau_rt.adapters.sionna.renderer import CoverageRenderer

    npy_files = list(input_dir.glob("*coverage*.npy"))
    if not npy_files:
        click.echo(click.style("Error: No coverage .npy file found.", fg="red"))
        raise SystemExit(1)

    path_gain = np.load(npy_files[0])
    click.echo(f"Loaded {npy_files[0]} (shape={path_gain.shape})")

    results = CoverageRenderer.render_all(input_dir, path_gain)
    for name, path in results.items():
        click.echo(click.style(f"  {name}: {path}", fg="green"))


@cli.command("view")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--metric", default="path_gain_db", help="初期表示メトリクス")
@click.option("--interactive/--no-interactive", default=True, help="インタラクティブモード")
def view_data(input_dir: Path, metric: str, interactive: bool):
    """インタラクティブ・ビューアでシミュレーション結果を閲覧します。"""
    import matplotlib

    matplotlib.use("TkAgg")

    from plateau_rt.application.viewer import DatasetViewer

    viewer = DatasetViewer(input_dir)

    if interactive:
        click.echo("Starting interactive viewer... (close window to exit)")
        viewer.interactive()
    else:
        viewer.show(metric)


@cli.command("run-all")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--num-rx", default=4, help="PathSolver用テスト受信点数")
@click.option("--keep-intermediates", is_flag=True, help="中間ファイルを保持する")
def run_all(input_file: Path, output_dir: Path, num_rx: int, keep_intermediates: bool):
    """一気貫通: CityJSONのパースから電波シミュレーション、画像生成までを実行します。"""
    from plateau_rt.adapters.sionna.simulator import SionnaSimulator

    click.echo(click.style("=== Starting End-to-End Pipeline ===", fg="cyan"))

    builder = SceneBuilder(input_file, output_dir)
    xml_path = builder.run()
    manifest_path = output_dir / "manifest.json"

    click.echo(click.style("=== Proceeding to Full Simulation ===", fg="cyan"))
    simulator = SionnaSimulator(xml_path, manifest_path)
    results = simulator.run_full_simulation(
        output_dir,
        num_rx=num_rx,
        keep_intermediates=keep_intermediates,
    )

    click.echo(click.style("=== Pipeline Finished! ===", fg="green", bold=True))
    for name, path in results.items():
        click.echo(f"  {name}: {path}")


@cli.command("rf-tomo-bench")
@click.option("--dataset", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--suite", type=click.Choice(["full", "smoke", "unit"]), default="smoke", show_default=True
)
@click.option(
    "--tracks", multiple=True, type=click.Choice(["ideal-S", "ideal-N", "N-sep", "S_tau"])
)
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--configs", multiple=True)
@click.option("--strategies", multiple=True)
@click.option("--spaces", multiple=True, type=click.Choice(["bv", "vs"]))
@click.option("--gt", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--grid-center", nargs=3, type=float, default=None)
@click.option("--grid-half-size", nargs=3, type=float, default=None)
@click.option("--grid-spacing", type=float, default=None)
@click.option("--overwrite", is_flag=True, default=False, show_default=True)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Chains run in parallel processes; 1 keeps runtime_s uncontended.",
)
def rf_tomo_bench(
    dataset: Path,
    suite: str,
    tracks: tuple[str, ...],
    out: Path,
    configs: tuple[str, ...],
    strategies: tuple[str, ...],
    spaces: tuple[str, ...],
    gt: Path | None,
    seed: int,
    grid_center: tuple[float, float, float] | None,
    grid_half_size: tuple[float, float, float] | None,
    grid_spacing: float | None,
    overwrite: bool,
    workers: int,
):
    """Run the RF tomography baseline benchmark (design §6, §8 T16)."""
    from plateau_rt.application.rf_dataset_manifest import ManifestError
    from plateau_rt.application.rf_tomography_benchmark import run_benchmark

    try:
        run = run_benchmark(
            dataset,
            out,
            suite,
            tracks=list(tracks) or None,
            configs=list(configs) or None,
            strategies=list(strategies) or None,
            spaces=list(spaces) or None,
            gt_path=gt,
            grid_center=grid_center,
            grid_half_size=grid_half_size,
            grid_spacing=grid_spacing,
            dataset_seed=seed,
            overwrite=overwrite,
            workers=workers,
        )
    except (ValueError, FileExistsError, FileNotFoundError, ManifestError) as exc:
        raise click.ClickException(str(exc)) from exc
    counts = {"ok": 0, "n/a": 0, "error": 0}
    for row in run.rows:
        counts[str(row["status"])] += 1
    click.echo(f"results: {run.results_path}")
    click.echo(f"run manifest: {run.run_manifest_path}")
    click.echo(
        f"rows: {len(run.rows)} (ok={counts['ok']} n/a={counts['n/a']} error={counts['error']})"
    )


@cli.command("rf-tomo-gt")
@click.argument("dataset", type=click.Path(exists=True, path_type=Path))
@click.option("--scene", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--out", type=click.Path(path_type=Path), default=None)
@click.option("--no-register", is_flag=True, default=False, show_default=True)
@click.option("--no-surfaces", is_flag=True, default=False, show_default=True)
@click.option("--surface-spacing", type=float, default=0.25, show_default=True)
@click.option("--surface-margin", type=float, default=5.0, show_default=True)
@click.option(
    "--los-polarization", type=click.Choice(["none", "vv"]), default="none", show_default=True
)
@click.option("--cluster-tol", type=float, default=0.01, show_default=True)
def rf_tomo_gt(
    dataset: Path,
    scene: Path | None,
    out: Path | None,
    no_register: bool,
    no_surfaces: bool,
    surface_spacing: float,
    surface_margin: float,
    los_polarization: str,
    cluster_tol: float,
):
    """Build tomography_gt.npz from a dataset's path ground truth and scene mesh (T17)."""
    import json

    import numpy as np

    from plateau_rt.application.rf_dataset_manifest import ManifestError
    from plateau_rt.application.rf_tomography_gt import (
        summarize_tomography_gt,
        write_tomography_gt,
    )

    try:
        path = write_tomography_gt(
            dataset,
            scene=scene,
            out=out,
            register=not no_register,
            surfaces=not no_surfaces,
            surface_spacing=surface_spacing,
            surface_margin=surface_margin,
            los_polarization=los_polarization,
            cluster_tol_m=cluster_tol,
        )
    except (ValueError, FileNotFoundError, ManifestError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"tomography_gt: {path}")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    click.echo(json.dumps(summarize_tomography_gt(arrays)))


if __name__ == "__main__":
    cli()
