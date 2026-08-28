"""2Dヒートマップと3Dシーンのレンダリング機能"""
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class CoverageRenderer:
    """カバレッジデータを各種カラーマップで画像化するレンダラー"""

    # Default colormap and figure settings
    DEFAULT_CMAP = 'jet'
    DEFAULT_DPI = 150
    DEFAULT_FIGSIZE = (10, 8)

    @staticmethod
    def render_path_gain(
        path_gain: np.ndarray,
        output_path: Path,
        vmin_db: float = -120.0,
        vmax_db: float = -40.0,
        cmap: str = DEFAULT_CMAP,
        title: str = 'Path Gain [dB]',
        dpi: int = DEFAULT_DPI,
    ) -> Path:
        """パスゲインの2DヒートマップをdBスケールでPNGに描画する
        
        Args:
            path_gain: path_gain array, shape [cells_y, cells_x] or [num_tx, cells_y, cells_x].
                       If 3D, the first TX (index 0) is used.
            output_path: Output PNG file path.
            vmin_db: Minimum value for colorbar (dB).
            vmax_db: Maximum value for colorbar (dB).
            cmap: Matplotlib colormap name.
            title: Plot title.
            dpi: Output image DPI.
            
        Returns:
            Path to the saved PNG file.
        """
        # Handle [num_tx, H, W] shape by selecting TX 0
        if path_gain.ndim == 3:
            path_gain = path_gain[0]
        
        # Convert to dB, handling zeros/negatives
        with np.errstate(divide='ignore', invalid='ignore'):
            gain_db = 10.0 * np.log10(np.maximum(path_gain, 1e-30))
        
        fig, ax = plt.subplots(1, 1, figsize=CoverageRenderer.DEFAULT_FIGSIZE)
        im = ax.imshow(
            gain_db,
            cmap=cmap,
            vmin=vmin_db,
            vmax=vmax_db,
            origin='lower',
            aspect='equal',
        )
        ax.set_title(title)
        ax.set_xlabel('Cell X')
        ax.set_ylabel('Cell Y')
        fig.colorbar(im, ax=ax, label='dB', shrink=0.8)
        fig.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Heatmap saved to {output_path}")
        return output_path

    @staticmethod
    def render_metric(
        data: np.ndarray,
        output_path: Path,
        metric_name: str,
        cmap: str = DEFAULT_CMAP,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: Optional[str] = None,
        dpi: int = DEFAULT_DPI,
    ) -> Path:
        """汎用メトリクス用の2Dヒートマップ描画
        
        Args:
            data: 2D array [cells_y, cells_x]
            output_path: Output PNG path
            metric_name: Name of the metric (used for colorbar label)
            cmap: Colormap
            vmin/vmax: Colorbar range
            title: Title (defaults to metric_name)
            dpi: Output DPI
        """
        if data.ndim == 3:
            data = data[0]
        
        fig, ax = plt.subplots(1, 1, figsize=CoverageRenderer.DEFAULT_FIGSIZE)
        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin='lower',
            aspect='equal',
        )
        ax.set_title(title or metric_name)
        ax.set_xlabel('Cell X')
        ax.set_ylabel('Cell Y')
        fig.colorbar(im, ax=ax, label=metric_name, shrink=0.8)
        fig.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Heatmap ({metric_name}) saved to {output_path}")
        return output_path

    @staticmethod
    def render_all(output_dir: Path, path_gain: np.ndarray) -> dict:
        """path_gain データから生成可能な全ヒートマップを一括レンダリング
        
        Returns:
            Dict of {metric_name: output_path}
        """
        results = {}
        
        # 1. Path Gain (dB)
        results['path_gain_db'] = CoverageRenderer.render_path_gain(
            path_gain, output_dir / 'heatmap_path_gain.png'
        )
        
        # 2. RSS (linear, assuming default tx power)
        results['rss'] = CoverageRenderer.render_metric(
            path_gain if path_gain.ndim == 2 else path_gain[0],
            output_dir / 'heatmap_rss_linear.png',
            metric_name='RSS (linear)',
            cmap='viridis',
        )
        
        return results


def render_3d_scene(
    scene,  # sionna.rt.Scene
    output_path: Path,
    paths=None,  # sionna.rt.Paths
    radio_map=None,  # sionna.rt.RadioMap  
    camera_position: list = None,
    camera_look_at: list = None,
    resolution: tuple = (1024, 768),
    num_samples: int = 256,
) -> Path:
    """Sionnaのレンダラーを使用して3Dシーン画像を生成
    
    Args:
        scene: Sionna Scene object
        output_path: Output PNG path
        paths: Optional Paths object to overlay ray paths
        radio_map: Optional RadioMap to overlay
        camera_position: Camera position [x, y, z]. If None, auto-determined from scene.
        camera_look_at: Point to look at [x, y, z]. If None, looks at scene center.
        resolution: Image resolution (width, height)
        num_samples: Rays per pixel for rendering quality
    """
    from sionna.rt import Camera
    
    # Auto-determine camera position if not provided
    if camera_position is None:
        # Get scene bounding box to position camera
        # Default: overhead diagonal view
        camera_position = [150, -150, 200]
    if camera_look_at is None:
        camera_look_at = [0, 0, 0]
    
    cam = Camera(position=camera_position, look_at=camera_look_at)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    scene.render_to_file(
        camera=cam,
        filename=str(output_path),
        paths=paths,
        radio_map=radio_map,
        resolution=resolution,
        num_samples=num_samples,
        show_devices=True,
    )
    print(f"3D render saved to {output_path}")
    return output_path
