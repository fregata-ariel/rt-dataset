from pathlib import Path

import click

from plateau_rt.application.build_scene import SceneBuilder


@click.group()
def cli():
    """PLATEAU CityJSON to Sionna-RT Dataset Generator"""
    pass


@cli.command("build")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
def build_scene(input_file: Path, output_dir: Path):
    """Step 1: CityJSONからSionna-RT用シーン(PLY/XML)とマニフェストを生成します。"""
    click.echo(f"Building scene from {input_file} into {output_dir}...")
    builder = SceneBuilder(input_file, output_dir)
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


@cli.command("rf-camera-multiview")
@click.argument("xml_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--num-views", type=int, default=8, show_default=True)
@click.option("--radius-m", type=float, default=30.0, show_default=True)
@click.option("--ue-height-m", type=float, default=1.5, show_default=True)
@click.option("--target", nargs=3, type=float, default=(5.0, 5.0, 5.0), show_default=True)
@click.option("--bs-position", nargs=3, type=float, default=(-50.0, -50.0, 30.0), show_default=True)
@click.option("--rx-rows", type=int, default=8, show_default=True)
@click.option("--rx-cols", type=int, default=8, show_default=True)
@click.option("--carrier-ghz", type=float, default=3.5, show_default=True)
@click.option("--bandwidth-mhz", type=float, default=100.0, show_default=True)
@click.option("--frequency-bins", type=int, default=64, show_default=True)
@click.option("--max-depth", type=int, default=5, show_default=True)
@click.option("--synthetic-array/--explicit-array", default=True, show_default=True)
def rf_camera_multiview(
    xml_file: Path,
    output_dir: Path,
    num_views: int,
    radius_m: float,
    ue_height_m: float,
    target: tuple[float, float, float],
    bs_position: tuple[float, float, float],
    rx_rows: int,
    rx_cols: int,
    carrier_ghz: float,
    bandwidth_mhz: float,
    frequency_bins: int,
    max_depth: int,
    synthetic_array: bool,
):
    """1 BS / multi-UEのRFカメラデータセットを生成します。

    targetを中心とする半径radius-mのリング上にUE(RFカメラ)を配置し、
    全UEを1回のPathSolver呼び出しでトレースします。BSもtargetを向きます。
    """
    from plateau_rt.adapters.sionna.rf_camera_dataset import (
        RFMultiViewConfig,
        RFMultiViewDataset,
    )
    from plateau_rt.domain.rf_camera.camera import generate_ring_views

    target = tuple(target)
    views = generate_ring_views(
        target=target,
        radius_m=radius_m,
        ue_height_m=ue_height_m,
        num_views=num_views,
    )
    config = RFMultiViewConfig(
        carrier_frequency_hz=carrier_ghz * 1e9,
        bandwidth_hz=bandwidth_mhz * 1e6,
        num_frequency_bins=frequency_bins,
        tx_position=tuple(bs_position),
        tx_look_at=target,
        rx_rows=rx_rows,
        rx_cols=rx_cols,
        max_depth=max_depth,
        synthetic_array=synthetic_array,
    )
    RFMultiViewDataset(xml_file, views=views, config=config).run(output_dir)


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

    click.echo(click.style("=== Optical reference renders ===", fg="cyan", bold=True))
    transforms_path = render_optical_references(
        dataset_dir,
        scene_xml=scene_xml,
        width=width,
        height=height,
        fov_x_deg=fov_x_deg,
        spp=spp,
        seed=seed,
    )
    click.echo(click.style(f"Success! transforms written to: {transforms_path}", fg="green"))


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


if __name__ == "__main__":
    cli()
