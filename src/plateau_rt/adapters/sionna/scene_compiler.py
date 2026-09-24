import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
from xml.dom import minidom

from plateau_rt.adapters.geometry.trimesh_adapter import MeshRecord
from plateau_rt.domain.models import MaterialType
from plateau_rt.domain.models import Scene as DomainScene


class SionnaSceneCompiler:
    MATERIAL_MAP = {
        MaterialType.CONCRETE: "itu_concrete",
        MaterialType.WOOD: "itu_wood",
        MaterialType.GLASS: "itu_glass",
        MaterialType.METAL: "itu_metal",
        MaterialType.DEFAULT: "itu_concrete",
    }

    def compile(
        self, domain_scene: DomainScene, mesh_records: List[MeshRecord], output_dir: Path
    ) -> Path:
        xml_path = output_dir / f"{domain_scene.scene_id}.xml"
        self._write_mitsuba_xml(mesh_records, xml_path)
        print(f"Sionna scene compiled at {xml_path}")
        return xml_path

    def _write_mitsuba_xml(self, mesh_records: List[MeshRecord], out_path: Path) -> None:
        scene_el = ET.Element("scene", version="3.0.0")

        # 1. 使用されるITUマテリアルの一覧を取得
        used_itu_materials = set(self.MATERIAL_MAP[r.material] for r in mesh_records)

        # 2. Sionnaが自動認識するプレースホルダー(mat-itu_...)を定義
        for itu_mat in used_itu_materials:
            bsdf_el = ET.SubElement(scene_el, "bsdf", type="diffuse", id=f"mat-{itu_mat}")
            ET.SubElement(bsdf_el, "rgb", name="reflectance", value="0.5, 0.5, 0.5")

        # 3. メッシュ (PLY) の参照追加
        for record in mesh_records:
            itu_mat = self.MATERIAL_MAP[record.material]
            shape_el = ET.SubElement(scene_el, "shape", type="ply", id=record.object_id)
            ET.SubElement(shape_el, "string", name="filename", value=record.file_path.name)
            # 対応するITUマテリアルを参照させる
            ET.SubElement(shape_el, "ref", id=f"mat-{itu_mat}", name="bsdf")

        xml_str = ET.tostring(scene_el, encoding="utf-8")
        parsed_xml = minidom.parseString(xml_str)
        pretty_xml = parsed_xml.toprettyxml(indent="    ")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)
