"""
規格資料庫模組
管理產品規格、容差標準等
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import numpy as np


@dataclass
class ToleranceSpec:
    """容差規格"""

    dimension_tolerance: float = 1.0  # 尺寸容差 (mm)
    position_tolerance: float = 2.0  # 位置容差 (mm)
    angle_tolerance: float = 5.0  # 角度容差 (度)
    defect_depth_threshold: float = 0.5  # 缺陷深度閾值 (mm)
    defect_area_threshold: float = 100.0  # 缺陷面積閾值 (mm²)
    roughness_threshold: float = 0.5  # 粗糙度閾值 (mm RMS)


@dataclass
class ProductSpec:
    """產品規格"""

    product_id: str  # 產品 ID
    product_name: str  # 產品名稱
    category: str  # 類別 (例如: "電子零件", "機械零件")

    # 尺寸規格
    dimensions: Optional[Dict[str, float]] = None  # {"length": 100, "width": 50, ...}
    volume: Optional[float] = None  # 體積 (mm³)
    weight: Optional[float] = None  # 重量 (g)

    # 組裝規格
    required_parts: List[str] = field(default_factory=list)  # 必須零件列表
    assembly_positions: Dict[str, List[float]] = field(
        default_factory=dict
    )  # 組裝位置

    # 缺陷標準
    allowed_defect_types: List[str] = field(
        default_factory=list
    )  # 允許的缺陷類型 (例如小凹陷)
    critical_defect_types: List[str] = field(
        default_factory=list
    )  # 關鍵缺陷類型 (不允許)

    # 容差規格
    tolerance: ToleranceSpec = field(default_factory=ToleranceSpec)

    # 其他屬性
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpecificationDatabase:
    """
    規格資料庫
    管理產品規格、查詢、更新等
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化規格資料庫

        Args:
            db_path: 資料庫路徑 (JSON 檔案), None 則使用預設路徑
        """
        if db_path is None:
            db_path = "data/specifications.json"

        self.db_path = Path(db_path)
        self.specs: Dict[str, ProductSpec] = {}

        # 載入資料庫
        if self.db_path.exists():
            self.load_from_file()
        else:
            # 建立預設規格
            self._create_default_specs()

    # ==================== 資料庫操作 ====================

    def add_spec(self, spec: ProductSpec) -> bool:
        """
        新增產品規格

        Args:
            spec: 產品規格

        Returns:
            是否成功
        """
        self.specs[spec.product_id] = spec
        return True

    def get_spec(self, product_id: str) -> Optional[ProductSpec]:
        """
        取得產品規格

        Args:
            product_id: 產品 ID

        Returns:
            產品規格, 或 None
        """
        return self.specs.get(product_id)

    def update_spec(self, product_id: str, spec: ProductSpec) -> bool:
        """
        更新產品規格

        Args:
            product_id: 產品 ID
            spec: 新規格

        Returns:
            是否成功
        """
        if product_id not in self.specs:
            return False

        self.specs[product_id] = spec
        return True

    def delete_spec(self, product_id: str) -> bool:
        """
        刪除產品規格

        Args:
            product_id: 產品 ID

        Returns:
            是否成功
        """
        if product_id in self.specs:
            del self.specs[product_id]
            return True
        return False

    def list_all_specs(self) -> List[ProductSpec]:
        """
        列出所有產品規格

        Returns:
            規格列表
        """
        return list(self.specs.values())

    def search_specs(
        self, category: Optional[str] = None, name_pattern: Optional[str] = None
    ) -> List[ProductSpec]:
        """
        搜尋產品規格

        Args:
            category: 類別篩選
            name_pattern: 名稱模式 (部分匹配)

        Returns:
            符合的規格列表
        """
        results = []

        for spec in self.specs.values():
            # 類別篩選
            if category and spec.category != category:
                continue

            # 名稱篩選
            if name_pattern and name_pattern.lower() not in spec.product_name.lower():
                continue

            results.append(spec)

        return results

    # ==================== 檔案操作 ====================

    def save_to_file(self) -> bool:
        """
        儲存到檔案

        Returns:
            是否成功
        """
        try:
            # 建立目錄
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # 轉換為 JSON
            data = {}
            for product_id, spec in self.specs.items():
                data[product_id] = self._spec_to_dict(spec)

            # 寫入檔案
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"儲存失敗: {e}")
            return False

    def load_from_file(self) -> bool:
        """
        從檔案載入

        Returns:
            是否成功
        """
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 轉換為 ProductSpec
            self.specs = {}
            for product_id, spec_dict in data.items():
                self.specs[product_id] = self._dict_to_spec(spec_dict)

            return True

        except Exception as e:
            print(f"載入失敗: {e}")
            return False

    # ==================== 轉換方法 ====================

    def _spec_to_dict(self, spec: ProductSpec) -> Dict[str, Any]:
        """將 ProductSpec 轉為字典"""
        return {
            "product_id": spec.product_id,
            "product_name": spec.product_name,
            "category": spec.category,
            "dimensions": spec.dimensions,
            "volume": spec.volume,
            "weight": spec.weight,
            "required_parts": spec.required_parts,
            "assembly_positions": spec.assembly_positions,
            "allowed_defect_types": spec.allowed_defect_types,
            "critical_defect_types": spec.critical_defect_types,
            "tolerance": {
                "dimension_tolerance": spec.tolerance.dimension_tolerance,
                "position_tolerance": spec.tolerance.position_tolerance,
                "angle_tolerance": spec.tolerance.angle_tolerance,
                "defect_depth_threshold": spec.tolerance.defect_depth_threshold,
                "defect_area_threshold": spec.tolerance.defect_area_threshold,
                "roughness_threshold": spec.tolerance.roughness_threshold,
            },
            "metadata": spec.metadata,
        }

    def _dict_to_spec(self, data: Dict[str, Any]) -> ProductSpec:
        """將字典轉為 ProductSpec"""
        tolerance = ToleranceSpec(
            dimension_tolerance=data.get("tolerance", {}).get("dimension_tolerance", 1.0),
            position_tolerance=data.get("tolerance", {}).get("position_tolerance", 2.0),
            angle_tolerance=data.get("tolerance", {}).get("angle_tolerance", 5.0),
            defect_depth_threshold=data.get("tolerance", {}).get(
                "defect_depth_threshold", 0.5
            ),
            defect_area_threshold=data.get("tolerance", {}).get(
                "defect_area_threshold", 100.0
            ),
            roughness_threshold=data.get("tolerance", {}).get("roughness_threshold", 0.5),
        )

        return ProductSpec(
            product_id=data["product_id"],
            product_name=data["product_name"],
            category=data["category"],
            dimensions=data.get("dimensions"),
            volume=data.get("volume"),
            weight=data.get("weight"),
            required_parts=data.get("required_parts", []),
            assembly_positions=data.get("assembly_positions", {}),
            allowed_defect_types=data.get("allowed_defect_types", []),
            critical_defect_types=data.get("critical_defect_types", []),
            tolerance=tolerance,
            metadata=data.get("metadata", {}),
        )

    # ==================== 預設規格 ====================

    def _create_default_specs(self):
        """建立預設規格"""

        # 規格 1: 電子零件盒
        spec1 = ProductSpec(
            product_id="ELEC-BOX-001",
            product_name="電子零件盒",
            category="電子零件",
            dimensions={"length": 100.0, "width": 50.0, "height": 30.0},
            volume=150000.0,
            weight=150.0,
            required_parts=["pcb_board", "connector", "screw_m3"],
            assembly_positions={
                "pcb_board": [50.0, 25.0, 5.0],
                "connector": [80.0, 25.0, 10.0],
                "screw_m3": [10.0, 10.0, 2.0],
            },
            allowed_defect_types=["minor_scratch"],
            critical_defect_types=["crack", "deep_dent"],
            tolerance=ToleranceSpec(
                dimension_tolerance=0.5,
                position_tolerance=1.0,
                angle_tolerance=3.0,
                defect_depth_threshold=0.3,
                defect_area_threshold=50.0,
                roughness_threshold=0.3,
            ),
        )

        # 規格 2: 機械支架
        spec2 = ProductSpec(
            product_id="MECH-BRACKET-001",
            product_name="機械支架",
            category="機械零件",
            dimensions={"length": 200.0, "width": 100.0, "height": 50.0},
            volume=1000000.0,
            weight=500.0,
            required_parts=["mounting_hole", "slot"],
            assembly_positions={
                "mounting_hole": [20.0, 20.0, 0.0],
                "slot": [100.0, 50.0, 10.0],
            },
            allowed_defect_types=[],
            critical_defect_types=["crack", "deep_dent", "bump"],
            tolerance=ToleranceSpec(
                dimension_tolerance=1.0,
                position_tolerance=2.0,
                angle_tolerance=5.0,
                defect_depth_threshold=0.5,
                defect_area_threshold=100.0,
                roughness_threshold=0.5,
            ),
        )

        # 規格 3: 測試零件 (用於測試)
        spec3 = ProductSpec(
            product_id="TEST-PART-001",
            product_name="測試零件",
            category="測試",
            dimensions={"length": 100.0, "width": 50.0, "height": 30.0},
            volume=150000.0,
            required_parts=["screw_m6", "nut_hex", "washer"],
            assembly_positions={
                "screw_m6": [10.0, 20.0, 5.0],
                "nut_hex": [50.0, 60.0, 10.0],
                "washer": [100.0, 100.0, 15.0],
            },
            tolerance=ToleranceSpec(),
        )

        self.add_spec(spec1)
        self.add_spec(spec2)
        self.add_spec(spec3)


if __name__ == "__main__":
    # 測試規格資料庫
    print("規格資料庫測試\n")

    # 建立資料庫
    db = SpecificationDatabase(db_path="outputs/test_specifications.json")

    # 測試 1: 列出所有規格
    print("測試 1: 列出所有規格")
    specs = db.list_all_specs()
    print(f"  總共 {len(specs)} 個規格:")
    for spec in specs:
        print(f"    - {spec.product_id}: {spec.product_name} ({spec.category})")

    # 測試 2: 取得特定規格
    print(f"\n測試 2: 取得特定規格")
    spec = db.get_spec("ELEC-BOX-001")
    if spec:
        print(f"  產品: {spec.product_name}")
        print(f"  尺寸: {spec.dimensions}")
        print(f"  必須零件: {spec.required_parts}")
        print(f"  尺寸容差: {spec.tolerance.dimension_tolerance} mm")

    # 測試 3: 搜尋規格
    print(f"\n測試 3: 搜尋規格")
    results = db.search_specs(category="電子零件")
    print(f"  電子零件類別: {len(results)} 個")
    for r in results:
        print(f"    - {r.product_name}")

    results = db.search_specs(name_pattern="支架")
    print(f"  名稱包含'支架': {len(results)} 個")
    for r in results:
        print(f"    - {r.product_name}")

    # 測試 4: 新增規格
    print(f"\n測試 4: 新增規格")
    new_spec = ProductSpec(
        product_id="CUSTOM-001",
        product_name="自訂零件",
        category="自訂",
        dimensions={"diameter": 50.0, "height": 20.0},
    )
    success = db.add_spec(new_spec)
    print(f"  新增: {'成功' if success else '失敗'}")
    print(f"  總規格數: {len(db.list_all_specs())}")

    # 測試 5: 儲存與載入
    print(f"\n測試 5: 儲存與載入")
    success = db.save_to_file()
    print(f"  儲存: {'成功' if success else '失敗'}")
    print(f"  檔案: {db.db_path}")

    # 載入
    db2 = SpecificationDatabase(db_path="outputs/test_specifications.json")
    print(f"  載入: {len(db2.list_all_specs())} 個規格")

    # 測試 6: 容差檢查
    print(f"\n測試 6: 容差規格")
    spec = db.get_spec("MECH-BRACKET-001")
    if spec:
        print(f"  產品: {spec.product_name}")
        print(f"  尺寸容差: {spec.tolerance.dimension_tolerance} mm")
        print(f"  位置容差: {spec.tolerance.position_tolerance} mm")
        print(f"  角度容差: {spec.tolerance.angle_tolerance}°")
        print(f"  缺陷深度閾值: {spec.tolerance.defect_depth_threshold} mm")
        print(f"  粗糙度閾值: {spec.tolerance.roughness_threshold} mm RMS")

    print(f"\n✓ 規格資料庫測試完成")
