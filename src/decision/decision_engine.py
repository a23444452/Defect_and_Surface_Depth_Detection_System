"""
決策引擎
整合所有檢測結果進行最終決策
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from .specification import ProductSpec, SpecificationDatabase
from .quality_judge import QualityJudge, JudgmentResult, QualityLevel
from ..measurement import MeasurementResult, DefectResult, AssemblyResult


class InspectionAction(Enum):
    """檢測動作"""

    ACCEPT = "accept"  # 接受 (合格)
    REWORK = "rework"  # 返工 (可修復)
    REJECT = "reject"  # 拒絕 (不可修復)
    MANUAL_CHECK = "manual_check"  # 人工檢查


@dataclass
class InspectionDecision:
    """檢測決策"""

    product_id: str  # 產品 ID
    action: InspectionAction  # 決策動作
    judgment: JudgmentResult  # 品質判斷結果

    # 時間戳記
    timestamp: datetime = field(default_factory=datetime.now)

    # 建議
    recommendations: List[str] = field(default_factory=list)

    # 額外資訊
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionEngine:
    """
    決策引擎
    整合規格資料庫、品質判斷,做出最終檢測決策
    """

    def __init__(
        self,
        spec_db: Optional[SpecificationDatabase] = None,
        judge: Optional[QualityJudge] = None,
    ):
        """
        初始化決策引擎

        Args:
            spec_db: 規格資料庫
            judge: 品質判斷器
        """
        self.spec_db = spec_db if spec_db else SpecificationDatabase()
        self.judge = judge if judge else QualityJudge()

        # 決策歷史
        self.decision_history: List[InspectionDecision] = []

    # ==================== 核心決策方法 ====================

    def make_decision(
        self,
        product_id: str,
        measurement: Optional[MeasurementResult] = None,
        defects: Optional[List[DefectResult]] = None,
        assembly_results: Optional[List[AssemblyResult]] = None,
        ai_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> InspectionDecision:
        """
        做出檢測決策

        Args:
            product_id: 產品 ID
            measurement: 尺寸量測結果
            defects: 缺陷列表
            assembly_results: 組裝驗證結果
            ai_detections: AI 檢測結果 (可選,用於參考)

        Returns:
            檢測決策
        """
        # 1. 取得產品規格
        spec = self.spec_db.get_spec(product_id)
        if spec is None:
            # 找不到規格,要求人工檢查
            return InspectionDecision(
                product_id=product_id,
                action=InspectionAction.MANUAL_CHECK,
                judgment=JudgmentResult(
                    quality_level=QualityLevel.FAIL,
                    overall_score=0.0,
                    failure_reasons=[f"找不到產品規格: {product_id}"],
                ),
                recommendations=["請確認產品 ID 是否正確", "或新增產品規格到資料庫"],
            )

        # 2. 品質判斷
        judgment = self.judge.judge_overall(
            spec=spec,
            measurement=measurement,
            defects=defects,
            assembly_results=assembly_results,
        )

        # 3. 決定動作
        action = self._determine_action(judgment, spec, defects)

        # 4. 生成建議
        recommendations = self._generate_recommendations(
            judgment, spec, measurement, defects, assembly_results
        )

        # 5. 建立決策
        decision = InspectionDecision(
            product_id=product_id,
            action=action,
            judgment=judgment,
            recommendations=recommendations,
            metadata={
                "spec": spec.product_name,
                "has_measurement": measurement is not None,
                "has_defects": defects is not None and len(defects) > 0,
                "has_assembly": assembly_results is not None
                and len(assembly_results) > 0,
                "ai_detections_count": len(ai_detections) if ai_detections else 0,
            },
        )

        # 6. 記錄歷史
        self.decision_history.append(decision)

        return decision

    # ==================== 決策邏輯 ====================

    def _determine_action(
        self,
        judgment: JudgmentResult,
        spec: ProductSpec,
        defects: Optional[List[DefectResult]],
    ) -> InspectionAction:
        """
        根據判斷結果決定動作

        Args:
            judgment: 品質判斷結果
            spec: 產品規格
            defects: 缺陷列表

        Returns:
            檢測動作
        """
        # 合格 -> 接受
        if judgment.quality_level == QualityLevel.PASS:
            return InspectionAction.ACCEPT

        # 警告 -> 根據情況決定
        if judgment.quality_level == QualityLevel.WARNING:
            # 如果分數 > 85,接受
            if judgment.overall_score >= 85.0:
                return InspectionAction.ACCEPT
            # 否則人工檢查
            else:
                return InspectionAction.MANUAL_CHECK

        # 不合格 -> 判斷是否可返工
        if judgment.quality_level == QualityLevel.FAIL:
            # 檢查是否有關鍵缺陷
            if defects:
                for defect in defects:
                    if defect.defect_type in spec.critical_defect_types:
                        # 關鍵缺陷 -> 拒絕
                        return InspectionAction.REJECT

            # 檢查尺寸問題
            if not judgment.dimension_pass:
                # 尺寸問題通常不可返工
                return InspectionAction.REJECT

            # 組裝問題 -> 可返工
            if not judgment.assembly_pass and judgment.defect_pass:
                return InspectionAction.REWORK

            # 僅缺陷問題 -> 根據嚴重程度
            if not judgment.defect_pass and judgment.dimension_pass:
                if judgment.defect_score > 50.0:
                    return InspectionAction.REWORK
                else:
                    return InspectionAction.REJECT

            # 預設: 拒絕
            return InspectionAction.REJECT

        # 預設: 人工檢查
        return InspectionAction.MANUAL_CHECK

    def _generate_recommendations(
        self,
        judgment: JudgmentResult,
        spec: ProductSpec,
        measurement: Optional[MeasurementResult],
        defects: Optional[List[DefectResult]],
        assembly_results: Optional[List[AssemblyResult]],
    ) -> List[str]:
        """
        生成建議

        Args:
            judgment: 判斷結果
            spec: 產品規格
            measurement: 量測結果
            defects: 缺陷列表
            assembly_results: 組裝結果

        Returns:
            建議列表
        """
        recommendations = []

        # 尺寸問題建議
        if not judgment.dimension_pass:
            recommendations.append("檢查加工設備精度")
            recommendations.append("確認量測基準是否正確")

        # 缺陷問題建議
        if not judgment.defect_pass and defects:
            # 根據缺陷類型給建議
            defect_types = set(d.defect_type for d in defects)

            if "dent" in defect_types or "bump" in defect_types:
                recommendations.append("檢查表面處理流程")
                recommendations.append("確認搬運過程是否有碰撞")

            if "crack" in defect_types:
                recommendations.append("檢查材料品質")
                recommendations.append("確認是否有應力集中")

            if "rough_surface" in defect_types:
                recommendations.append("檢查表面拋光流程")
                recommendations.append("確認刀具或模具狀態")

        # 組裝問題建議
        if not judgment.assembly_pass and assembly_results:
            missing_parts = [r.part_name for r in assembly_results if not r.present]
            if missing_parts:
                recommendations.append(f"補充缺少的零件: {', '.join(missing_parts)}")

            position_errors = [
                r for r in assembly_results if r.present and not r.position_correct
            ]
            if position_errors:
                recommendations.append("調整零件安裝位置")
                recommendations.append("確認定位治具是否正確")

        # 通用建議
        if judgment.overall_score < 70.0:
            recommendations.append("建議重新加工或組裝")

        if not recommendations:
            recommendations.append("無特殊建議")

        return recommendations

    # ==================== 統計分析 ====================

    def get_statistics(self, product_id: Optional[str] = None) -> Dict[str, Any]:
        """
        取得決策統計

        Args:
            product_id: 產品 ID 篩選 (None 表示所有)

        Returns:
            統計資訊
        """
        # 篩選決策
        decisions = self.decision_history
        if product_id:
            decisions = [d for d in decisions if d.product_id == product_id]

        if len(decisions) == 0:
            return {
                "total": 0,
                "accept": 0,
                "rework": 0,
                "reject": 0,
                "manual_check": 0,
                "accept_rate": 0.0,
                "avg_score": 0.0,
            }

        # 統計
        total = len(decisions)
        accept = sum(1 for d in decisions if d.action == InspectionAction.ACCEPT)
        rework = sum(1 for d in decisions if d.action == InspectionAction.REWORK)
        reject = sum(1 for d in decisions if d.action == InspectionAction.REJECT)
        manual_check = sum(
            1 for d in decisions if d.action == InspectionAction.MANUAL_CHECK
        )

        avg_score = sum(d.judgment.overall_score for d in decisions) / total

        return {
            "total": total,
            "accept": accept,
            "rework": rework,
            "reject": reject,
            "manual_check": manual_check,
            "accept_rate": accept / total if total > 0 else 0.0,
            "reject_rate": reject / total if total > 0 else 0.0,
            "avg_score": avg_score,
        }

    def print_statistics(self, product_id: Optional[str] = None):
        """
        列印統計資訊

        Args:
            product_id: 產品 ID 篩選
        """
        stats = self.get_statistics(product_id)

        print("決策統計")
        if product_id:
            print(f"產品: {product_id}")
        print(f"  總檢測數: {stats['total']}")
        print(f"  接受: {stats['accept']} ({stats['accept_rate']*100:.1f}%)")
        print(f"  返工: {stats['rework']}")
        print(f"  拒絕: {stats['reject']} ({stats['reject_rate']*100:.1f}%)")
        print(f"  人工檢查: {stats['manual_check']}")
        print(f"  平均分數: {stats['avg_score']:.1f}")

    # ==================== 批次決策 ====================

    def batch_decision(
        self,
        inspections: List[Dict[str, Any]],
    ) -> List[InspectionDecision]:
        """
        批次決策

        Args:
            inspections: 檢測資料列表
                [
                    {
                        "product_id": "...",
                        "measurement": ...,
                        "defects": ...,
                        "assembly_results": ...,
                    },
                    ...
                ]

        Returns:
            決策列表
        """
        decisions = []

        for inspection in inspections:
            decision = self.make_decision(
                product_id=inspection["product_id"],
                measurement=inspection.get("measurement"),
                defects=inspection.get("defects"),
                assembly_results=inspection.get("assembly_results"),
                ai_detections=inspection.get("ai_detections"),
            )
            decisions.append(decision)

        return decisions


if __name__ == "__main__":
    # 測試決策引擎
    print("決策引擎測試\n")

    import numpy as np
    from ..measurement import (
        MeasurementResult,
        DefectResult,
        AssemblyResult,
    )
    from .specification import SpecificationDatabase

    # 建立決策引擎
    engine = DecisionEngine()

    # 測試 1: 合格產品
    print("測試 1: 合格產品決策")
    measurement = MeasurementResult(
        dimensions=np.array([100.2, 50.1, 30.0]),
        volume=150300.0,
        confidence=0.95,
    )
    defects = []
    assembly_results = [
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
    ]

    decision = engine.make_decision(
        product_id="TEST-PART-001",
        measurement=measurement,
        defects=defects,
        assembly_results=assembly_results,
    )

    print(f"  產品: {decision.product_id}")
    print(f"  決策: {decision.action.value.upper()}")
    print(f"  品質等級: {decision.judgment.quality_level.value.upper()}")
    print(f"  總分: {decision.judgment.overall_score:.1f}")
    print(f"  建議: {decision.recommendations}")

    # 測試 2: 不合格產品 (尺寸超差)
    print(f"\n測試 2: 不合格產品 (尺寸超差)")
    measurement_fail = MeasurementResult(
        dimensions=np.array([105.0, 55.0, 35.0]),  # 嚴重超差
        volume=160000.0,
        confidence=0.92,
    )

    decision = engine.make_decision(
        product_id="TEST-PART-001",
        measurement=measurement_fail,
        defects=[],
        assembly_results=assembly_results,
    )

    print(f"  產品: {decision.product_id}")
    print(f"  決策: {decision.action.value.upper()}")
    print(f"  品質等級: {decision.judgment.quality_level.value.upper()}")
    print(f"  總分: {decision.judgment.overall_score:.1f}")
    print(f"  不合格原因:")
    for reason in decision.judgment.failure_reasons:
        print(f"    - {reason}")
    print(f"  建議:")
    for rec in decision.recommendations:
        print(f"    - {rec}")

    # 測試 3: 可返工產品 (組裝問題)
    print(f"\n測試 3: 可返工產品 (組裝問題)")
    assembly_results_fail = [
        AssemblyResult(
            part_name="screw_m6",
            present=True,
            position_correct=False,
            orientation_correct=True,
            position=np.array([10.0, 25.0, 5.0]),
            expected_position=np.array([10.0, 20.0, 5.0]),
            position_error=5.0,  # 超出容差
            confidence=0.90,
        ),
    ]

    decision = engine.make_decision(
        product_id="TEST-PART-001",
        measurement=measurement,
        defects=[],
        assembly_results=assembly_results_fail,
    )

    print(f"  產品: {decision.product_id}")
    print(f"  決策: {decision.action.value.upper()}")
    print(f"  品質等級: {decision.judgment.quality_level.value.upper()}")
    print(f"  總分: {decision.judgment.overall_score:.1f}")
    print(f"  建議:")
    for rec in decision.recommendations:
        print(f"    - {rec}")

    # 測試 4: 關鍵缺陷 (拒絕)
    print(f"\n測試 4: 關鍵缺陷產品")
    defects_critical = [
        DefectResult(
            defect_type="crack",  # 關鍵缺陷
            severity="critical",
            depth=2.0,
            area=500.0,
            location=np.array([50, 50, 10]),
            confidence=0.95,
        ),
    ]

    decision = engine.make_decision(
        product_id="ELEC-BOX-001",  # 這個規格定義 crack 為關鍵缺陷
        measurement=measurement,
        defects=defects_critical,
        assembly_results=[],
    )

    print(f"  產品: {decision.product_id}")
    print(f"  決策: {decision.action.value.upper()}")
    print(f"  品質等級: {decision.judgment.quality_level.value.upper()}")
    print(f"  不合格原因:")
    for reason in decision.judgment.failure_reasons:
        print(f"    - {reason}")

    # 測試 5: 統計
    print(f"\n測試 5: 決策統計")
    engine.print_statistics()

    print(f"\n✓ 決策引擎測試完成")
