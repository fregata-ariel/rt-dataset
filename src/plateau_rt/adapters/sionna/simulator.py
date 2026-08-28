import json
import gc
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

from sionna.rt import (
    load_scene, Transmitter, Receiver, PlanarArray,
    Scene, RadioMapSolver, PathSolver, Camera,
)


class SionnaSimulator:
    """Sionna-RT を用いた電波伝搬シミュレータ

    RadioMapSolver（2Dカバレッジ）と PathSolver（位相情報付きパス分解）
    の両方を実行し、結果の保存と3Dレンダリングまでを担当する。
    """

    def __init__(self, xml_path: Path, manifest_path: Path):
        self.xml_path = xml_path
        self.manifest_path = manifest_path
        self.scene: Optional[Scene] = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def run_coverage_simulation(self, output_dir: Path) -> Path:
        """後方互換: 既存の2Dカバレッジのみを実行する"""
        print("--- Starting Sionna-RT Coverage Simulation ---")
        self._load_scene()
        rm = self._run_radio_map()
        cm_path = self._save_path_gain(rm, output_dir)
        return cm_path

    def run_full_simulation(
        self,
        output_dir: Path,
        *,
        num_rx: int = 4,
        keep_intermediates: bool = False,
    ) -> Dict[str, Path]:
        """拡張シミュレーション: カバレッジ + パス分解 + 3Dレンダリング

        Args:
            output_dir: 出力ディレクトリ
            num_rx: PathSolver 用のテスト受信点数
            keep_intermediates: True の場合、中間 .npy も保存する

        Returns:
            生成されたファイルパスの辞書
        """
        print("--- Starting Sionna-RT Full Simulation ---")
        results: Dict[str, Path] = {}
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Phase 1: RadioMapSolver (2D Coverage) ---
        self._load_scene()
        rm = self._run_radio_map()
        path_gain = np.array(rm.path_gain)

        # カバレッジ npy を保存
        cm_path = self._save_path_gain(rm, output_dir)
        results["coverage_npy"] = cm_path

        # 2D ヒートマップ画像を生成
        from plateau_rt.adapters.sionna.renderer import CoverageRenderer
        results["heatmap_path_gain"] = CoverageRenderer.render_path_gain(
            path_gain, output_dir / "render_2d_heatmap.png"
        )

        # カバレッジ付き3Dレンダリング
        from plateau_rt.adapters.sionna.renderer import render_3d_scene
        try:
            cam_pos, cam_look = self._auto_camera_position()
            results["render_3d_coverage"] = render_3d_scene(
                self.scene,
                output_dir / "render_3d_coverage.png",
                radio_map=rm,
                camera_position=cam_pos,
                camera_look_at=cam_look,
            )
        except Exception as e:
            print(f"Warning: 3D coverage render failed: {e}")

        # VRAM 解放
        del rm
        gc.collect()

        # --- Phase 2: PathSolver (パス分解・位相情報) ---
        print("Computing propagation paths (PathSolver)...")
        rx_positions = self._generate_rx_positions(num_rx)
        self._add_receivers(rx_positions)

        paths = self._run_path_solver()

        # パスデータを保存
        paths_path = self._save_paths(paths, rx_positions, output_dir)
        results["paths_npz"] = paths_path

        # パス付き3Dレンダリング
        try:
            results["render_3d_paths"] = render_3d_scene(
                self.scene,
                output_dir / "render_3d_paths.png",
                paths=paths,
                camera_position=cam_pos,
                camera_look_at=cam_look,
            )
        except Exception as e:
            print(f"Warning: 3D paths render failed: {e}")

        # VRAM 解放
        del paths
        gc.collect()

        # --- GC: 中間ファイル管理 ---
        if not keep_intermediates:
            print("Keeping all files (GC deferred to Phase 2).")

        print(f"--- Full Simulation Complete. Files: {list(results.keys())} ---")
        return results

    # ------------------------------------------------------------------ #
    #  Internal: Scene Setup
    # ------------------------------------------------------------------ #

    def _load_scene(self) -> None:
        """シーンのロードとTx設定"""
        if self.scene is not None:
            return  # 再ロード防止

        print(f"Loading scene from {self.xml_path}...")
        self.scene = load_scene(str(self.xml_path))
        assert self.scene is not None, "Failed to load Sionna scene."

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self._setup_transceivers(manifest.get("center_lat_lon", [0, 0]))

    def _setup_transceivers(self, center_lat_lon: list) -> None:
        """Tx/Rxアンテナ設定"""
        assert self.scene is not None

        self.scene.tx_array = PlanarArray(
            num_rows=4, num_cols=4,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="tr38901", polarization="V",
        )
        self.scene.rx_array = PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="dipole", polarization="V",
        )

        tx = Transmitter(
            name="tx_base_station",
            position=[-50, -50, 30.0],
        )
        self.scene.add(tx)
        self.scene.frequency = 3.5e9

    # ------------------------------------------------------------------ #
    #  Internal: RadioMapSolver
    # ------------------------------------------------------------------ #

    def _run_radio_map(self):
        """RadioMapSolver を実行してカバレッジマップを計算"""
        print("Computing coverage map (RadioMapSolver)...")
        rm_solver = RadioMapSolver()
        rm = rm_solver(
            self.scene,
            max_depth=3,
            cell_size=[1.0, 1.0],
            center=[0.0, 0.0, 1.5],
            size=[200.0, 200.0],
            orientation=[0.0, 0.0, 0.0],
        )
        return rm

    def _save_path_gain(self, rm, output_dir: Path) -> Path:
        """path_gain を npy で保存"""
        cm_path = output_dir / f"{self.xml_path.stem}_coverage.npy"
        np.save(cm_path, np.array(rm.path_gain))
        print(f"Coverage map saved to {cm_path}")
        return cm_path

    # ------------------------------------------------------------------ #
    #  Internal: PathSolver
    # ------------------------------------------------------------------ #

    def _generate_rx_positions(self, num_rx: int) -> List[list]:
        """シーン内にテスト受信点を格子状に配置"""
        # シーンの概算サイズから等間隔に配置
        half = 80.0  # ±80m 範囲
        if num_rx == 1:
            return [[0.0, 0.0, 1.5]]

        side = int(np.ceil(np.sqrt(num_rx)))
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) >= num_rx:
                    break
                x = -half + (2 * half) * i / max(side - 1, 1)
                y = -half + (2 * half) * j / max(side - 1, 1)
                positions.append([x, y, 1.5])
        return positions[:num_rx]

    def _add_receivers(self, rx_positions: List[list]) -> None:
        """レシーバーをシーンに追加"""
        assert self.scene is not None
        for i, pos in enumerate(rx_positions):
            rx = Receiver(name=f"rx_{i}", position=pos)
            self.scene.add(rx)
        print(f"Added {len(rx_positions)} receivers")

    def _run_path_solver(self):
        """PathSolver を実行してパスデータを取得"""
        p_solver = PathSolver()
        paths = p_solver(
            scene=self.scene,
            max_depth=5,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            synthetic_array=True,
            seed=42,
        )
        return paths

    def _save_paths(self, paths, rx_positions: List[list], output_dir: Path) -> Path:
        """PathSolver の結果を npz で保存

        保存内容:
        - a: 複素チャネル係数 (complex64)
        - tau: パス遅延 (float32)
        - theta_t, phi_t: 出射角 (float32)
        - theta_r, phi_r: 到来角 (float32)
        - rx_positions: レシーバー位置 (float32)
        """
        paths_path = output_dir / "paths.npz"

        # CIR 取得 (complex numpy array)
        a, tau = paths.cir(out_type="numpy")

        save_dict = {
            "a": a,
            "tau": tau,
            "rx_positions": np.array(rx_positions, dtype=np.float32),
        }

        # 角度情報の取得（利用可能な場合）
        try:
            save_dict["theta_t"] = np.array(paths.theta_t)
            save_dict["phi_t"] = np.array(paths.phi_t)
            save_dict["theta_r"] = np.array(paths.theta_r)
            save_dict["phi_r"] = np.array(paths.phi_r)
        except Exception as e:
            print(f"Warning: Could not extract angle data: {e}")

        np.savez_compressed(paths_path, **save_dict)
        print(f"Paths data saved to {paths_path} (keys: {list(save_dict.keys())})")
        return paths_path

    # ------------------------------------------------------------------ #
    #  Internal: Camera positioning
    # ------------------------------------------------------------------ #

    def _auto_camera_position(self) -> tuple:
        """シーンのサイズから俯瞰カメラ位置を自動決定"""
        # デフォルト: 斜め45度からの俯瞰
        cam_pos = [150.0, -150.0, 200.0]
        cam_look = [0.0, 0.0, 0.0]
        return cam_pos, cam_look