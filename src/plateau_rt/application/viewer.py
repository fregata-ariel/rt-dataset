"""インタラクティブデータセットビューア

強度(path_gain)をデフォルトに、各種データをカラーマップで閲覧できる。
"""
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from PIL import Image


class DatasetViewer:
    """シミュレーション結果をインタラクティブに可視化するビューア"""

    # Available display metrics and their settings
    METRICS = {
        'path_gain_db': {
            'label': 'Path Gain [dB]',
            'cmap': 'jet',
            'vmin': -120.0,
            'vmax': -40.0,
            'transform': lambda pg: 10.0 * np.log10(np.maximum(pg, 1e-30)),
        },
        'path_gain_linear': {
            'label': 'Path Gain (linear)',
            'cmap': 'viridis',
            'vmin': None,
            'vmax': None,
            'transform': lambda pg: pg,
        },
        'rss_dbm': {
            'label': 'RSS [dBm]',
            'cmap': 'plasma',
            'vmin': -130.0,
            'vmax': -30.0,
            'transform': lambda pg: 10.0 * np.log10(np.maximum(pg * 1000, 1e-30)),  # Assuming 1W tx
        },
    }

    def __init__(self, record_dir: Path):
        """データディレクトリからデータを読み込む
        
        Args:
            record_dir: シミュレーション結果ディレクトリのパス
        """
        self.record_dir = Path(record_dir)
        self.path_gain: Optional[np.ndarray] = None
        self.paths_data: Optional[dict] = None
        self.render_images: dict = {}
        
        self._load_data()

    def _load_data(self):
        """ディレクトリ内の利用可能なデータを読み込む"""
        # Load path_gain (look for any *coverage*.npy file)
        npy_files = list(self.record_dir.glob('*coverage*.npy'))
        if npy_files:
            self.path_gain = np.load(npy_files[0])
            if self.path_gain.ndim == 3:
                self.path_gain = self.path_gain[0]  # TX 0
            print(f"Loaded path_gain: shape={self.path_gain.shape}")
        
        # Load paths data if available
        npz_files = list(self.record_dir.glob('paths*.npz'))
        if npz_files:
            self.paths_data = dict(np.load(npz_files[0], allow_pickle=True))
            print(f"Loaded paths data: keys={list(self.paths_data.keys())}")
            
            # Add paths-derived metrics if complex channel data available
            if 'a' in self.paths_data:
                a = self.paths_data['a']  # complex channel coefficients
                # Compute per-path metrics for the viewer
                self.METRICS['channel_amplitude'] = {
                    'label': 'Channel Amplitude |a|',
                    'cmap': 'hot',
                    'vmin': None,
                    'vmax': None,
                    'source': 'paths',  # Marker that this is paths data
                }
                self.METRICS['channel_phase'] = {
                    'label': 'Channel Phase ∠a [rad]',
                    'cmap': 'hsv',
                    'vmin': -np.pi,
                    'vmax': np.pi,
                    'source': 'paths',
                }
        
        # Load rendered images
        for png_path in self.record_dir.glob('render_*.png'):
            name = png_path.stem
            self.render_images[name] = png_path
            print(f"Found render image: {name}")
        for png_path in self.record_dir.glob('heatmap_*.png'):
            name = png_path.stem
            self.render_images[name] = png_path
            print(f"Found heatmap image: {name}")

    def show(self, metric: str = 'path_gain_db'):
        """指定されたメトリクスで静的表示
        
        Args:
            metric: Display metric name (key from METRICS dict)
        """
        if self.path_gain is None:
            print("Error: No path_gain data loaded.")
            return
        
        if metric not in self.METRICS:
            print(f"Unknown metric: {metric}. Available: {list(self.METRICS.keys())}")
            return
        
        m = self.METRICS[metric]
        
        # Check if this is a paths-derived metric
        if m.get('source') == 'paths':
            self._show_paths_metric(metric)
            return
        
        data = m['transform'](self.path_gain)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            data, cmap=m['cmap'], vmin=m['vmin'], vmax=m['vmax'],
            origin='lower', aspect='equal'
        )
        ax.set_title(m['label'])
        ax.set_xlabel('Cell X')
        ax.set_ylabel('Cell Y')
        fig.colorbar(im, ax=ax, label=m['label'], shrink=0.8)
        plt.show()

    def _show_paths_metric(self, metric: str):
        """PathSolver結果由来のメトリクスを棒グラフで表示"""
        if self.paths_data is None or 'a' not in self.paths_data:
            print("Error: No paths data with channel coefficients loaded.")
            return
        
        a = self.paths_data['a']  # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
        
        # For display, sum over antenna dims and show per-path for first Rx-Tx pair
        # Simplify: take first Rx, first Tx, all antennas averaged
        if a.ndim >= 5:
            a_view = a[0, 0, 0, 0, :]  # [num_paths]
        elif a.ndim == 3:
            a_view = a[0, 0, :]  # [num_paths]
        else:
            a_view = a.flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Amplitude
        axes[0].stem(np.abs(a_view), linefmt='C0-', markerfmt='C0o', basefmt='k-')
        axes[0].set_title('Channel Amplitude |a|')
        axes[0].set_xlabel('Path index')
        axes[0].set_ylabel('|a|')
        
        # Phase
        axes[1].stem(np.angle(a_view), linefmt='C1-', markerfmt='C1o', basefmt='k-')
        axes[1].set_title('Channel Phase ∠a [rad]')
        axes[1].set_xlabel('Path index')
        axes[1].set_ylabel('Phase [rad]')
        axes[1].set_ylim(-np.pi, np.pi)
        
        fig.suptitle(f'PathSolver Results (Rx 0 → Tx 0, {len(a_view)} paths)')
        fig.tight_layout()
        plt.show()

    def interactive(self):
        """matplotlib widgets を用いたインタラクティブ表示
        
        左側にデータ表示、右側にメトリクス切替ボタンを配置。
        デフォルトは path_gain (dB) 表示。
        """
        if self.path_gain is None:
            print("Error: No path_gain data loaded.")
            return

        # Only use coverage-map metrics for the interactive view
        coverage_metrics = {
            k: v for k, v in self.METRICS.items()
            if v.get('source') != 'paths'
        }
        
        # Add rendered images as options
        image_labels = {}
        for name, path in self.render_images.items():
            image_labels[name] = path

        all_labels = list(coverage_metrics.keys()) + list(image_labels.keys())
        
        fig, ax_main = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.05, right=0.72)

        # Initial display: path_gain in dB
        default_metric = 'path_gain_db'
        m = coverage_metrics[default_metric]
        data = m['transform'](self.path_gain)
        im = ax_main.imshow(
            data, cmap=m['cmap'], vmin=m['vmin'], vmax=m['vmax'],
            origin='lower', aspect='equal'
        )
        ax_main.set_title(m['label'])
        cb = fig.colorbar(im, ax=ax_main, label=m['label'], shrink=0.8)

        # RadioButtons for metric selection
        ax_radio = fig.add_axes([0.74, 0.3, 0.24, 0.5])
        display_labels = []
        for k in all_labels:
            if k in coverage_metrics:
                display_labels.append(coverage_metrics[k]['label'])
            else:
                display_labels.append(f"📷 {k}")
        
        radio = RadioButtons(ax_radio, display_labels, active=0)

        def on_select(label):
            # Find which key corresponds to this label
            idx = display_labels.index(label)
            key = all_labels[idx]
            
            ax_main.clear()
            nonlocal cb
            if cb:
                cb.remove()
                cb = None
            
            if key in coverage_metrics:
                m = coverage_metrics[key]
                data = m['transform'](self.path_gain)
                new_im = ax_main.imshow(
                    data, cmap=m['cmap'], vmin=m['vmin'], vmax=m['vmax'],
                    origin='lower', aspect='equal'
                )
                ax_main.set_title(m['label'])
                cb = fig.colorbar(new_im, ax=ax_main, label=m['label'], shrink=0.8)
            else:
                # Show rendered PNG image
                img = Image.open(image_labels[key])
                ax_main.imshow(np.array(img))
                ax_main.set_title(key)
                ax_main.axis('off')
            
            fig.canvas.draw_idle()

        radio.on_clicked(on_select)
        plt.show()
