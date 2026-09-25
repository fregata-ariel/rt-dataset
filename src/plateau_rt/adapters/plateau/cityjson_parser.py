import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from plateau_rt.domain.models import Building, MaterialType, Scene, Surface, SurfaceType
from plateau_rt.domain.scene_transform import SceneTransform


class CityJSONAdapter:
    """CityJSONをパースし、ドメインモデルに変換するクラス"""

    # PLATEAUの構造種別(urf:buildingStructureType)からマテリアルへのマッピング
    STRUCTURE_TO_MATERIAL = {
        "木造": MaterialType.WOOD,
        "鉄筋コンクリート造": MaterialType.CONCRETE,
        "鉄骨造": MaterialType.METAL,
        "鉄骨鉄筋コンクリート造": MaterialType.CONCRETE,
        "レンガ造": MaterialType.CONCRETE,  # 便宜上のフォールバック
    }

    def __init__(self, file_path: Path):
        self.file_path = file_path

    def parse(self) -> Scene:
        """CityJSONを読み込み、ローカル座標化されたSceneモデルを返す"""
        print(f"Loading CityJSON: {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            cm_dict = json.load(f)

        # --- 1. 頂点座標の展開 (圧縮された整数値 -> 現実のメートル座標) ---
        raw_vertices = cm_dict.get("vertices", [])
        transform = cm_dict.get("transform", {})
        scale = transform.get("scale", [1.0, 1.0, 1.0])
        translate = transform.get("translate", [0.0, 0.0, 0.0])

        vertices = [
            [
                v[0] * scale[0] + translate[0],
                v[1] * scale[1] + translate[1],
                v[2] * scale[2] + translate[2],
            ]
            for v in raw_vertices
        ]

        if not vertices:
            raise ValueError("No vertices found in CityJSON.")

        # --- 2. バウンディングボックスとローカル原点の算出 ---
        # 展開した頂点群から直接BBox (min/max) を計算する
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)
        min_z = min(v[2] for v in vertices)

        # 中心点を計算 (Sionna-RTに渡す際のローカル原点 0,0,0 となる)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        center_z = min_z  # Z(高さ)は一番低い場所を0とする
        offset = (center_x, center_y, center_z)

        # -------------------------------------------------------------

        buildings: List[Building] = []

        # CityObjects から建物を抽出
        for obj_id, city_obj in cm_dict.get("CityObjects", {}).items():
            if city_obj.get("type") not in ["Building", "BuildingPart"]:
                continue

            # 構造種別からベースマテリアルを決定
            attributes = city_obj.get("attributes", {})
            structure_type = attributes.get("urf:buildingStructureType", "")
            base_material = self.STRUCTURE_TO_MATERIAL.get(structure_type, MaterialType.DEFAULT)

            surfaces: List[Surface] = []

            # ジオメトリのパース
            for geom_idx, geometry in enumerate(city_obj.get("geometry", [])):
                if geometry.get("type") not in ["MultiSurface", "Solid", "CompositeSurface"]:
                    continue

                semantics = geometry.get("semantics")
                if not semantics:
                    continue

                surfaces.extend(
                    self._extract_surfaces(
                        obj_id, geom_idx, geometry, semantics, vertices, offset, base_material
                    )
                )

            if surfaces:
                buildings.append(Building(building_id=obj_id, surfaces=surfaces))

        metadata = cm_dict.get("metadata")
        source_crs: str | None = None
        if isinstance(metadata, dict):
            reference = metadata.get("referenceSystem")
            if isinstance(reference, str) and reference:
                source_crs = reference
            elif reference is not None:
                source_crs = str(reference)
            else:
                source_crs = None

        return Scene(
            scene_id=self.file_path.stem,
            buildings=buildings,
            transform=SceneTransform(
                origin_projected_xyz=offset,
                source_crs=source_crs,
            ),
        )

    def _extract_surfaces(
        self,
        bldg_id: str,
        geom_idx: int,
        geometry: Dict[str, Any],
        semantics: Dict[str, Any],
        all_vertices: List[List[float]],
        offset: Tuple[float, float, float],
        base_material: MaterialType,
    ) -> List[Surface]:
        """ジオメトリのポリゴン配列とセマンティクス配列を照らし合わせてSurfaceリストを生成"""
        surfaces = []

        # cjioのセマンティクス定義リスト
        sem_surfaces = semantics.get("surfaces", [])
        # 各ポリゴンがどのセマンティクス定義に属するかを示すインデックス配列
        sem_values = semantics.get("values", [])

        boundaries = geometry["boundaries"]

        # Solid(3次元配列) か MultiSurface/CompositeSurface(2次元配列) かでネストを吸収
        if geometry["type"] == "Solid":
            poly_list = boundaries[0]
            sem_list = sem_values[0] if sem_values else []
        else:
            poly_list = boundaries
            sem_list = sem_values

        for poly_idx, polygon in enumerate(poly_list):
            if not sem_list or sem_list[poly_idx] is None:
                continue

            # セマンティクスの種類 (RoofSurface, WallSurface, GroundSurface等) を特定
            sem_def = sem_surfaces[sem_list[poly_idx]]
            sem_type = sem_def.get("type", "")

            surface_type = SurfaceType.UNKNOWN
            material = base_material

            if sem_type == "RoofSurface":
                surface_type = SurfaceType.ROOF
                # 屋根は構造に関わらずコンクリート等のデフォルトに上書きすることが多い
                material = MaterialType.DEFAULT
            elif sem_type == "WallSurface":
                surface_type = SurfaceType.WALL
            elif sem_type == "GroundSurface":
                surface_type = SurfaceType.GROUND
            else:
                continue  # 窓(Window)や扉(Door)などは一旦除外

            # 外形を形成する頂点（穴あきポリゴンの場合は最初の配列が外枠）
            exterior_ring_indices = polygon[0]

            # 頂点座標をローカル座標にオフセットして取得
            local_vertices = []
            for v_idx in exterior_ring_indices:
                v = all_vertices[v_idx]
                local_v = (v[0] - offset[0], v[1] - offset[1], v[2] - offset[2])
                local_vertices.append(local_v)

            surf = Surface(
                surface_id=f"{bldg_id}_{geom_idx}_{poly_idx}_{sem_type}",
                surface_type=surface_type,
                material=material,
                vertices=local_vertices,
            )
            surfaces.append(surf)

        return surfaces
