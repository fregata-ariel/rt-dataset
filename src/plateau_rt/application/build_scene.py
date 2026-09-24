import json
from datetime import datetime, timezone
from pathlib import Path

from plateau_rt.adapters.geometry.trimesh_adapter import TrimeshAdapter
from plateau_rt.adapters.plateau.cityjson_parser import CityJSONAdapter
from plateau_rt.adapters.sionna.scene_compiler import SionnaSceneCompiler
from plateau_rt.domain.models import Scene


class SceneBuilder:
    """CityJSONからSionna-RT用シーン一式を生成するパイプラインを管理するクラス"""

    def __init__(self, input_cityjson: Path, output_dir: Path):
        self.input_file = input_cityjson
        self.output_dir = output_dir
        # 出力先ディレクトリの確保
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        """パイプラインを一気通貫で実行し、生成されたXMLのパスを返す"""
        print("--- Starting Scene Build Pipeline ---")
        print(f"Input: {self.input_file}")
        print(f"Output Directory: {self.output_dir}")

        # 1. パース (外界: CityJSON -> ドメイン: Scene)
        parser = CityJSONAdapter(self.input_file)
        scene = parser.parse()

        # 2. メッシュ生成 (ドメイン: Scene -> 外界: PLYファイル群)
        mesher = TrimeshAdapter(self.output_dir)
        mesh_records = mesher.export_scene(scene)

        # 3. Mitsuba XML生成 (ドメイン: Scene + メッシュ情報 -> 外界: scene.xml)
        compiler = SionnaSceneCompiler()
        xml_path = compiler.compile(scene, mesh_records, self.output_dir)

        # 4. データセットのメタデータ(マニフェスト)を保存
        self._save_manifest(scene, mesh_records, xml_path)

        print("--- Pipeline Completed Successfully ---")
        return xml_path

    def _save_manifest(self, scene: Scene, mesh_records: list, xml_path: Path) -> None:
        """再現性を担保するためのメタデータ（マニフェスト）をJSONとして保存する"""

        # MeshRecordからSionna-RT実行時に必要なマテリアル割り当て情報を抽出
        material_mapping = {
            record.object_id: SionnaSceneCompiler.MATERIAL_MAP[record.material]
            for record in mesh_records
        }

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": self.input_file.name,
            "scene_id": scene.scene_id,
            "center_lat_lon": scene.center_lat_lon,
            "outputs": {"xml_file": xml_path.name, "mesh_count": len(mesh_records)},
            # Sionna-RTのシミュレーション層が読み込んで使うマテリアル辞書
            "material_mapping": material_mapping,
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        print(f"Manifest saved at {manifest_path}")
