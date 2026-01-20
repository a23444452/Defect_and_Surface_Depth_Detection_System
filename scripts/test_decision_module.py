#!/usr/bin/env python3
"""
決策模組完整測試
測試規格資料庫、品質判斷、決策引擎
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.decision import (
    SpecificationDatabase,
    ProductSpec,
    ToleranceSpec,
    QualityJudge,
    JudgmentResult,
    QualityLevel,
    DecisionEngine,
    InspectionAction,
)
from src.measurement import (
    MeasurementResult,
    DefectResult,
    AssemblyResult,
)
from src.utils import setup_logger


def test_specification_database():
    """測試規格資料庫"""
    print("\n" + "=" * 70)
    print("  測試 1: 規格資料庫")
    print("=" * 70)

    db = SpecificationDatabase(db_path="outputs/test_specifications.json")

    # 測試 1.1: 列出所有規格
    print("\n測試 1.1: 列出所有規格")
    specs = db.list_all_specs()
    print(f"  總共 {len(specs)} 個規格:")
    for spec in specs:
        print(
            f"    - {spec.product_id}: {spec.product_name} ({spec.category})"
        )

    # 測試 1.2: 取得特定規格
    print(f"\n測試 1.2: 取得特定規格")
    spec = db.get_spec("ELEC-BOX-001")
    if spec:
        print(f"  ✓ 產品: {spec.product_name}")
        print(f"    尺寸: {spec.dimensions}")
        print(f"    必須零件: {spec.required_parts}")
        print(f"    容差: {spec.tolerance.dimension_tolerance} mm")

    # 測試 1.3: 新增規格
    print(f"\n測試 1.3: 新增自訂規格")
    custom_spec = ProductSpec(
        product_id="CUSTOM-TEST-001",
        product_name="自訂測試零件",
        category="測試",
        dimensions={"diameter": 60.0, "height": 25.0},
        tolerance=ToleranceSpec(dimension_tolerance=0.8),
    )
    success = db.add_spec(custom_spec)
    print(f"  新增: {'✓' if success else '✗'}")

    # 測試 1.4: 搜尋規格
    print(f"\n測試 1.4: 搜尋規格")
    results = db.search_specs(category="電子零件")
    print(f"  電子零件類別: {len(results)} 個")

    results = db.search_specs(name_pattern="測試")
    print(f"  名稱包含'測試': {len(results)} 個")

    # 測試 1.5: 儲存與載入
    print(f"\n測試 1.5: 儲存與載入")
    success = db.save_to_file()
    print(f"  儲存: {'✓' if success else '✗'}")
    print(f"  檔案: {db.db_path}")

    db2 = SpecificationDatabase(db_path="outputs/test_specifications.json")
    print(f"  載入: {len(db2.list_all_specs())} 個規格 ✓")


def test_quality_judge():
    """測試品質判斷"""
    print("\n" + "=" * 70)
    print("  測試 2: 品質判斷系統")
    print("=" * 70)

    judge = QualityJudge()

    # 建立測試規格
    spec = ProductSpec(
        product_id="TEST-001",
        product_name="測試零件",
        category="測試",
        dimensions={"length": 100.0, "width": 50.0, "height": 30.0},
        required_parts=["screw_m6", "nut_hex", "washer"],
        tolerance=ToleranceSpec(
            dimension_tolerance=1.0,
            position_tolerance=2.0,
            defect_depth_threshold=0.5,
        ),
        critical_defect_types=["crack", "deep_dent"],
    )

    # 測試 2.1: 合格尺寸
    print("\n測試 2.1: 尺寸判斷 (合格)")
    measurement = MeasurementResult(
        length=100.3,
        width=50.2,
        height=30.1,
        volume=150600.0,
        center=np.array([50, 25, 15]),
        rotation=np.eye(3),
        confidence=0.95,
        method="obb",
    )
    dim_pass, dim_score, dim_failures = judge.judge_dimension(measurement, spec)
    print(f"  合格: {'✓' if dim_pass else '✗'}")
    print(f"  分數: {dim_score:.1f}")
    print(f"  問題: {dim_failures if dim_failures else '無'}")

    # 測試 2.2: 不合格尺寸
    print(f"\n測試 2.2: 尺寸判斷 (不合格)")
    measurement_fail = MeasurementResult(
        length=103.0,
        width=52.0,
        height=28.5,
        volume=147000.0,
        center=np.array([50, 25, 15]),
        rotation=np.eye(3),
        confidence=0.92,
        method="obb",
    )
    dim_pass, dim_score, dim_failures = judge.judge_dimension(measurement_fail, spec)
    print(f"  合格: {'✓' if dim_pass else '✗'}")
    print(f"  分數: {dim_score:.1f}")
    print(f"  問題:")
    for f in dim_failures:
        print(f"    - {f}")

    # 測試 2.3: 缺陷判斷
    print(f"\n測試 2.3: 缺陷判斷")
    defects = [
        DefectResult(
            defect_type="dent",
            severity="moderate",
            depth=0.8,
            area=120.0,
            location=np.array([30, 40, 10]),
            confidence=0.90,
        ),
        DefectResult(
            defect_type="bump",
            severity="minor",
            depth=0.4,
            area=50.0,
            location=np.array([70, 80, 15]),
            confidence=0.85,
        ),
    ]
    defect_pass, defect_score, defect_failures = judge.judge_defects(defects, spec)
    print(f"  合格: {'✓' if defect_pass else '✗'}")
    print(f"  分數: {defect_score:.1f}")
    print(f"  問題: {defect_failures if defect_failures else '無'}")

    # 測試 2.4: 組裝判斷
    print(f"\n測試 2.4: 組裝判斷")
    assembly_results = [
        AssemblyResult(
            part_name="screw_m6",
            present=True,
            position_correct=True,
            orientation_correct=True,
            position=np.array([10.0, 20.0, 5.0]),
            expected_position=np.array([10.0, 20.0, 5.0]),
            position_error=0.1,
            confidence=0.95,
        ),
        AssemblyResult(
            part_name="nut_hex",
            present=True,
            position_correct=False,
            orientation_correct=True,
            position=np.array([50.0, 63.0, 10.0]),
            expected_position=np.array([50.0, 60.0, 10.0]),
            position_error=3.0,
            confidence=0.90,
        ),
        AssemblyResult(
            part_name="washer",
            present=False,
            position_correct=False,
            orientation_correct=False,
            expected_position=np.array([100.0, 100.0, 15.0]),
            confidence=0.0,
        ),
    ]
    asm_pass, asm_score, asm_failures = judge.judge_assembly(assembly_results, spec)
    print(f"  合格: {'✓' if asm_pass else '✗'}")
    print(f"  分數: {asm_score:.1f}")
    print(f"  問題:")
    for f in asm_failures:
        print(f"    - {f}")

    # 測試 2.5: 綜合判斷
    print(f"\n測試 2.5: 綜合判斷")
    result = judge.judge_overall(
        spec=spec,
        measurement=measurement,
        defects=defects,
        assembly_results=assembly_results,
    )

    print(f"  品質等級: {result.quality_level.value.upper()}")
    print(f"  總分: {result.overall_score:.1f}")
    print(f"  尺寸: {'✓' if result.dimension_pass else '✗'} ({result.dimension_score:.1f})")
    print(f"  缺陷: {'✓' if result.defect_pass else '✗'} ({result.defect_score:.1f})")
    print(f"  組裝: {'✓' if result.assembly_pass else '✗'} ({result.assembly_score:.1f})")

    if result.failure_reasons:
        print(f"  不合格原因:")
        for reason in result.failure_reasons:
            print(f"    - {reason}")

    if result.warnings:
        print(f"  警告:")
        for warning in result.warnings:
            print(f"    - {warning}")


def test_decision_engine():
    """測試決策引擎"""
    print("\n" + "=" * 70)
    print("  測試 3: 決策引擎")
    print("=" * 70)

    engine = DecisionEngine()

    # 測試案例 1: 完全合格
    print("\n測試 3.1: 完全合格產品")
    measurement = MeasurementResult(
        length=100.2,
        width=50.1,
        height=30.0,
        volume=150300.0,
        center=np.array([50, 25, 15]),
        rotation=np.eye(3),
        confidence=0.95,
        method="obb",
    )
    decision = engine.make_decision(
        product_id="TEST-PART-001",
        measurement=measurement,
        defects=[],
        assembly_results=[
            AssemblyResult(
                part_name="screw_m6",
                present=True,
                position_correct=True,
                orientation_correct=True,
                position=np.array([10.0, 20.0, 5.0]),
                expected_position=np.array([10.0, 20.0, 5.0]),
                position_error=0.0,
                confidence=0.95,
            ),
        ],
    )

    print(f"  產品: {decision.product_id}")
    print(f"  決策: {decision.action.value.upper()}")
    print(f"  品質等級: {decision.judgment.quality_level.value.upper()}")
    print(f"  總分: {decision.judgment.overall_score:.1f}")
    print(f"  建議: {', '.join(decision.recommendations)}")

    # 測試案例 2: 尺寸超差 (拒絕)
    print(f"\n測試 3.2: 尺寸超差產品")
    measurement_fail = MeasurementResult(
        length=105.0,
        width=55.0,
        height=35.0,
        volume=160000.0,
        center=np.array([50, 25, 15]),
        rotation=np.eye(3),
        confidence=0.92,
        method="obb",
    )
    decision = engine.make_decision(
        product_id="TEST-PART-001",
        measurement=measurement_fail,
        defects=[],
        assembly_results=[],
    )

    print(f"  產品: {decision.product_id}")
    print(f"  決策: {decision.action.value.upper()}")
    print(f"  品質等級: {decision.judgment.quality_level.value.upper()}")
    print(f"  總分: {decision.judgment.overall_score:.1f}")
    print(f"  不合格原因:")
    for reason in decision.judgment.failure_reasons:
        print(f"    - {reason}")

    # 測試案例 3: 組裝問題 (可返工)
    print(f"\n測試 3.3: 組裝問題產品 (可返工)")
    decision = engine.make_decision(
        product_id="TEST-PART-001",
        measurement=measurement,
        defects=[],
        assembly_results=[
            AssemblyResult(
                part_name="screw_m6",
                present=False,
                position_correct=False,
                orientation_correct=False,
                expected_position=np.array([10.0, 20.0, 5.0]),
                confidence=0.0,
            ),
        ],
    )

    print(f"  產品: {decision.product_id}")
    print(f"  決策: {decision.action.value.upper()}")
    print(f"  品質等級: {decision.judgment.quality_level.value.upper()}")
    print(f"  總分: {decision.judgment.overall_score:.1f}")
    print(f"  建議:")
    for rec in decision.recommendations:
        print(f"    - {rec}")

    # 測試案例 4: 關鍵缺陷 (拒絕)
    print(f"\n測試 3.4: 關鍵缺陷產品")
    decision = engine.make_decision(
        product_id="ELEC-BOX-001",
        measurement=measurement,
        defects=[
            DefectResult(
                defect_type="crack",
                severity="critical",
                depth=2.0,
                area=500.0,
                location=np.array([50, 50, 10]),
                confidence=0.95,
            ),
        ],
        assembly_results=[],
    )

    print(f"  產品: {decision.product_id}")
    print(f"  決策: {decision.action.value.upper()}")
    print(f"  品質等級: {decision.judgment.quality_level.value.upper()}")
    print(f"  總分: {decision.judgment.overall_score:.1f}")
    print(f"  不合格原因:")
    for reason in decision.judgment.failure_reasons:
        print(f"    - {reason}")

    # 測試案例 5: 輕微問題 (警告)
    print(f"\n測試 3.5: 輕微問題產品 (警告)")
    decision = engine.make_decision(
        product_id="TEST-PART-001",
        measurement=measurement,
        defects=[
            DefectResult(
                defect_type="minor_scratch",
                severity="minor",
                depth=0.2,
                area=20.0,
                location=np.array([20, 30, 5]),
                confidence=0.80,
            ),
        ],
        assembly_results=[
            AssemblyResult(
                part_name="screw_m6",
                present=True,
                position_correct=True,
                orientation_correct=True,
                position=np.array([10.0, 20.0, 5.0]),
                expected_position=np.array([10.0, 20.0, 5.0]),
                position_error=0.5,
                confidence=0.90,
            ),
        ],
    )

    print(f"  產品: {decision.product_id}")
    print(f"  決策: {decision.action.value.upper()}")
    print(f"  品質等級: {decision.judgment.quality_level.value.upper()}")
    print(f"  總分: {decision.judgment.overall_score:.1f}")
    if decision.judgment.warnings:
        print(f"  警告:")
        for warning in decision.judgment.warnings:
            print(f"    - {warning}")


def test_batch_decision():
    """測試批次決策"""
    print("\n" + "=" * 70)
    print("  測試 4: 批次決策")
    print("=" * 70)

    engine = DecisionEngine()

    # 建立 10 個測試案例
    inspections = []
    for i in range(10):
        # 隨機生成測試資料
        dimensions = np.array([100.0, 50.0, 30.0]) + np.random.normal(0, 0.5, 3)
        measurement = MeasurementResult(
            length=float(dimensions[0]),
            width=float(dimensions[1]),
            height=float(dimensions[2]),
            volume=float(np.prod(dimensions)),
            center=np.array([50, 25, 15]),
            rotation=np.eye(3),
            confidence=0.90 + np.random.uniform(0, 0.1),
            method="obb",
        )

        # 隨機生成缺陷
        defects = []
        if np.random.rand() > 0.6:  # 40% 機率有缺陷
            defect_types = ["dent", "bump", "minor_scratch"]
            defects.append(
                DefectResult(
                    defect_type=np.random.choice(defect_types),
                    severity="minor" if np.random.rand() > 0.3 else "moderate",
                    depth=np.random.uniform(0.1, 0.8),
                    area=np.random.uniform(20, 150),
                    location=np.random.uniform(0, 100, 3),
                    confidence=0.85,
                )
            )

        inspections.append(
            {
                "product_id": "TEST-PART-001",
                "measurement": measurement,
                "defects": defects,
                "assembly_results": [],
            }
        )

    # 批次決策
    print(f"\n處理 {len(inspections)} 個檢測案例...")
    decisions = engine.batch_decision(inspections)

    # 統計結果
    action_counts = {}
    for decision in decisions:
        action = decision.action.value
        action_counts[action] = action_counts.get(action, 0) + 1

    print(f"\n批次決策結果:")
    print(f"  總數: {len(decisions)}")
    for action, count in action_counts.items():
        print(f"  {action.upper()}: {count} ({count/len(decisions)*100:.1f}%)")

    # 顯示統計
    print(f"\n整體統計:")
    engine.print_statistics()


def test_end_to_end():
    """測試端到端流程"""
    print("\n" + "=" * 70)
    print("  測試 5: 端到端決策流程")
    print("=" * 70)

    # 建立決策引擎
    engine = DecisionEngine()

    # 模擬完整檢測流程
    print("\n模擬產品檢測流程...")

    # 1. 取得產品規格
    product_id = "ELEC-BOX-001"
    spec = engine.spec_db.get_spec(product_id)
    print(f"\n1. 產品規格: {spec.product_name}")
    print(f"   預期尺寸: {spec.dimensions}")
    print(f"   必須零件: {spec.required_parts}")

    # 2. 模擬量測結果
    print(f"\n2. 尺寸量測")
    measurement = MeasurementResult(
        length=100.3,
        width=50.2,
        height=30.1,
        volume=151500.0,
        center=np.array([50, 25, 15]),
        rotation=np.eye(3),
        confidence=0.94,
        method="obb",
    )
    print(f"   實際尺寸: [{measurement.length:.1f}, {measurement.width:.1f}, {measurement.height:.1f}] mm")
    print(f"   體積: {measurement.volume:.0f} mm³")

    # 3. 模擬缺陷檢測
    print(f"\n3. 缺陷檢測")
    defects = [
        DefectResult(
            defect_type="minor_scratch",
            severity="minor",
            depth=0.15,
            area=15.0,
            location=np.array([30, 40, 10]),
            confidence=0.82,
        ),
    ]
    print(f"   檢測到 {len(defects)} 個缺陷")
    for d in defects:
        print(f"   - {d.defect_type}: {d.severity} (深度 {d.depth:.2f}mm)")

    # 4. 模擬組裝驗證
    print(f"\n4. 組裝驗證")
    assembly_results = [
        AssemblyResult(
            part_name="pcb_board",
            present=True,
            position_correct=True,
            orientation_correct=True,
            position=np.array([50.0, 25.0, 5.0]),
            expected_position=np.array([50.0, 25.0, 5.0]),
            position_error=0.0,
            confidence=0.96,
        ),
        AssemblyResult(
            part_name="connector",
            present=True,
            position_correct=True,
            orientation_correct=True,
            position=np.array([80.0, 25.0, 10.0]),
            expected_position=np.array([80.0, 25.0, 10.0]),
            position_error=0.0,
            confidence=0.93,
        ),
        AssemblyResult(
            part_name="screw_m3",
            present=True,
            position_correct=True,
            orientation_correct=True,
            position=np.array([10.0, 10.0, 2.0]),
            expected_position=np.array([10.0, 10.0, 2.0]),
            position_error=0.0,
            confidence=0.91,
        ),
    ]
    print(f"   驗證 {len(assembly_results)} 個零件")
    for a in assembly_results:
        status = "✓" if a.present and a.position_correct else "✗"
        print(f"   {status} {a.part_name}: 位置誤差 {a.position_error:.2f}mm")

    # 5. 做出決策
    print(f"\n5. 品質決策")
    decision = engine.make_decision(
        product_id=product_id,
        measurement=measurement,
        defects=defects,
        assembly_results=assembly_results,
    )

    print(f"   決策結果: {decision.action.value.upper()}")
    print(f"   品質等級: {decision.judgment.quality_level.value.upper()}")
    print(f"   總分: {decision.judgment.overall_score:.1f}")
    print(f"   尺寸: {'✓' if decision.judgment.dimension_pass else '✗'} ({decision.judgment.dimension_score:.1f})")
    print(f"   缺陷: {'✓' if decision.judgment.defect_pass else '✗'} ({decision.judgment.defect_score:.1f})")
    print(f"   組裝: {'✓' if decision.judgment.assembly_pass else '✗'} ({decision.judgment.assembly_score:.1f})")

    if decision.recommendations:
        print(f"\n   建議:")
        for rec in decision.recommendations:
            print(f"   - {rec}")


def main():
    """主函數"""
    print("\n" + "=" * 70)
    print("  決策模組完整測試")
    print("=" * 70)

    logger = setup_logger(name="TestDecision", log_dir="outputs/logs")

    # 執行所有測試
    test_specification_database()
    test_quality_judge()
    test_decision_engine()
    test_batch_decision()
    test_end_to_end()

    print("\n" + "=" * 70)
    print("  ✓ 所有測試完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
