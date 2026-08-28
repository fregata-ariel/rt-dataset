import click
from pathlib import Path

from plateau_rt.application.build_scene import SceneBuilder
from plateau_rt.adapters.sionna.simulator import SionnaSimulator

@click.group()
def cli():
    """PLATEAU CityJSON to Sionna-RT Dataset Generator"""
    pass

@cli.command("build")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
def build_scene(input_file: Path, output_dir: Path):
    """
    Step 1: CityJSONからSionna-RT用シーン(PLY/XML)とマニフェストを生成します。
    """
    click.echo(f"Building scene from {input_file} into {output_dir}...")
    builder = SceneBuilder(input_file, output_dir)
    xml_path = builder.run()
    click.echo(click.style(f"Success! Scene XML generated at: {xml_path}", fg="green"))

@cli.command("simulate")
@click.argument("xml_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("manifest_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
def simulate_coverage(xml_file: Path, manifest_file: Path, output_dir: Path):
    """
    Step 2: 生成されたXMLとマニフェストを用いて電波カバレッジマップを計算します。
    """
    click.echo(f"Running simulation for {xml_file}...")
    
    # 出力先ディレクトリの確保
    output_dir.mkdir(parents=True, exist_ok=True)
    
    simulator = SionnaSimulator(xml_file, manifest_file)
    result_path = simulator.run_coverage_simulation(output_dir)
    click.echo(click.style(f"Success! Coverage map generated at: {result_path}", fg="green"))

@cli.command("render")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def render_heatmaps(input_dir: Path):
    """
    シミュレーション結果(npy)から2Dヒートマップ画像群を生成します。
    """
    import numpy as np
    from plateau_rt.adapters.sionna.renderer import CoverageRenderer

    # coverage npy を探す
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
    """
    インタラクティブ・ビューアでシミュレーション結果を閲覧します。
    デフォルトは強度(path_gain dB)表示。
    """
    import matplotlib
    matplotlib.use('TkAgg')  # Interactive backend

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
    """
    一気貫通: CityJSONのパースから電波シミュレーション、画像生成までを全自動で実行します。
    """
    click.echo(click.style("=== Starting End-to-End Pipeline ===", fg="cyan"))
    
    # 1. シーン構築
    builder = SceneBuilder(input_file, output_dir)
    xml_path = builder.run()
    manifest_path = output_dir / "manifest.json"
    
    # 2. フルシミュレーション実行 (カバレッジ + パス分解 + レンダリング)
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