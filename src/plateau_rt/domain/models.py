from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from plateau_rt.domain.scene_transform import SceneTransform


class SurfaceType(Enum):
    """建物を構成する面の役割（Sionnaでの法線・マテリアル判定に利用）"""

    ROOF = "roof"
    WALL = "wall"
    GROUND = "ground"
    UNKNOWN = "unknown"


class MaterialType(Enum):
    """ドメイン内で扱う抽象化されたマテリアル（SionnaのITUプリセットに依存しない）"""

    CONCRETE = "concrete"
    WOOD = "wood"
    GLASS = "glass"
    METAL = "metal"
    DEFAULT = "default"
    GROUND = "ground"


@dataclass
class Surface:
    """建物を構成する単一の面（ポリゴン）"""

    surface_id: str
    surface_type: SurfaceType
    material: MaterialType
    vertices: List[Tuple[float, float, float]]  # ローカル座標系での頂点リスト


@dataclass
class Building:
    """単一の建物モデル"""

    building_id: str
    surfaces: List[Surface]


@dataclass
class Scene:
    """シミュレーション対象の全体シーン"""

    scene_id: str
    buildings: List[Building]
    transform: SceneTransform
