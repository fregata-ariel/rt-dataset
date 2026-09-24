from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# 外部ライブラリ
import trimesh

from plateau_rt.domain.models import MaterialType, Scene, Surface, SurfaceType


@dataclass
class MeshRecord:
    """エクスポートされたメッシュの情報（XMLコンパイラへ渡すための中間データ）"""

    object_id: str
    file_path: Path
    surface_type: SurfaceType
    material: MaterialType


class TrimeshAdapter:
    """ドメインのSceneモデルを3Dメッシュ(PLY)に変換・保存するクラス"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_scene(self, scene: Scene) -> List[MeshRecord]:
        """
        シーン内の全建物をPLYとして書き出し、そのメタデータを返す
        Sionnaで個別の部位にマテリアルを当てられるよう、「建物ID_部位」の単位でPLYを分割する。
        """
        records: List[MeshRecord] = []

        for bldg in scene.buildings:
            # 部位（屋根、壁、地面など）ごとにSurfaceをグループ化する
            # 例: {SurfaceType.WALL: [Surface, Surface, ...], SurfaceType.ROOF: [...]}
            surface_groups = {}
            for surf in bldg.surfaces:
                surface_groups.setdefault(surf.surface_type, []).append(surf)

            for surf_type, surfaces in surface_groups.items():
                # そのグループのベースマテリアル（基本はリストの先頭のもの）
                material = surfaces[0].material if surfaces else MaterialType.DEFAULT

                # 建物IDと部位から、Sionna上で識別するためのオブジェクト名を生成
                object_id = f"{bldg.building_id}_{surf_type.value}"

                # メッシュの生成と保存
                mesh = self._build_mesh(surfaces)
                if mesh.is_empty:
                    continue

                ply_path = self.output_dir / f"{object_id}.ply"
                mesh.export(str(ply_path), file_type="ply")

                records.append(
                    MeshRecord(
                        object_id=object_id,
                        file_path=ply_path,
                        surface_type=surf_type,
                        material=material,
                    )
                )

        print(f"Exported {len(records)} meshes to {self.output_dir}")
        return records

    def _build_mesh(self, surfaces: List[Surface]) -> trimesh.Trimesh:
        """複数の多角形Surfaceから、1つの三角形Trimeshを構築する"""
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        for surf in surfaces:
            pts = surf.vertices
            n_pts = len(pts)
            if n_pts < 3:
                continue  # 線や点はスキップ

            # 頂点を追加
            all_vertices.extend(pts)

            # --- Triangulation (三角形分割) ---
            # PLATEAUの面データは基本的に凸多角形（Convex Polygon）であるため、
            # 簡易的かつ高速な「Triangle Fan（扇状分割）」アルゴリズムを適用します。
            # v0を基点として、(v0, v1, v2), (v0, v2, v3)... と三角形を作ります。
            for i in range(1, n_pts - 1):
                face = (
                    vertex_offset,  # v0
                    vertex_offset + i,  # vi
                    vertex_offset + i + 1,  # vi+1
                )
                all_faces.append(face)

            vertex_offset += n_pts

        return trimesh.Trimesh(
            vertices=np.array(all_vertices, dtype=np.float32),
            faces=np.array(all_faces, dtype=np.int32),
            process=True,  # 重複頂点の結合(マージ)や不正な面のクリーンアップを自動実行
        )
