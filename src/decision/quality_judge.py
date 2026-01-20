"""
品質判斷系統
根據量測結果與規格進行品質判斷
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from .specification import ProductSpec
from ..measurement import MeasurementResult, DefectResult, AssemblyResult


class QualityLevel(Enum):
    """品質等級"""

    PASS = "pass"  # 合格
    WARNING = "warning"  # 警告 (可接受但接近容差)
    FAIL = "fail"  # 不合格


@dataclass
class JudgmentResult:
    """判斷結果"""

    quality_level: QualityLevel  # 品質等級
    overall_score: float  # 總分 [0-100]

    # 各項檢查結果
    dimension_pass: bool = True  # 尺寸是否合格
    defect_pass: bool = True  # 缺陷是否合格
    assembly_pass: bool = True  # 組裝是否合格

    # 不合格原因
    failure_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # 詳細資訊
    dimension_score: float = 100.0  # 尺寸分數
    defect_score: float = 100.0  # 缺陷分數
    assembly_score: float = 100.0  # 組裝分數

    metadata: Dict[str, Any] = field(default_factory=dict)


class QualityJudge:
    """
    品質判斷器
    根據規格對量測結果進行判斷
    """

    def __init__(self, warning_threshold: float = 0.8):
        """
        初始化品質判斷器

        Args:
            warning_threshold: 警告閾值 (0-1), 超過容差的此比例會發出警告
        """
        self.warning_threshold = warning_threshold

    # ==================== 尺寸判斷 ====================

    def judge_dimension(
        self, measurement: MeasurementResult, spec: ProductSpec
    ) -> tuple[bool, float, List[str]]:
        """
        判斷尺寸是否合格

        Args:
            measurement: 量測結果
            spec: 產品規格

        Returns:
            (是否合格, 分數, 不合格原因列表)
        """
        if spec.dimensions is None:
            return True, 100.0, []

        tolerance = spec.tolerance.dimension_tolerance
        failures = []
        errors = []

        # 檢查各個尺寸
        for dim_name, expected_value in spec.dimensions.items():
            if dim_name == "length":
                actual_value = measurement.length
                error = abs(actual_value - expected_value)
                errors.append(error / expected_value if expected_value > 0 else 0)

                if error > tolerance:
                    failures.append(
                        f"length 超出容差: {actual_value:.2f} mm "
                        f"(期望: {expected_value:.2f} ± {tolerance:.2f} mm, "
                        f"誤差: {error:.2f} mm)"
                    )

            elif dim_name == "width":
                actual_value = measurement.width
                error = abs(actual_value - expected_value)
                errors.append(error / expected_value if expected_value > 0 else 0)

                if error > tolerance:
                    failures.append(
                        f"width 超出容差: {actual_value:.2f} mm "
                        f"(期望: {expected_value:.2f} ± {tolerance:.2f} mm, "
                        f"誤差: {error:.2f} mm)"
                    )

            elif dim_name == "height":
                actual_value = measurement.height
                error = abs(actual_value - expected_value)
                errors.append(error / expected_value if expected_value > 0 else 0)

                if error > tolerance:
                    failures.append(
                        f"height 超出容差: {actual_value:.2f} mm "
                        f"(期望: {expected_value:.2f} ± {tolerance:.2f} mm, "
                        f"誤差: {error:.2f} mm)"
                    )

            elif dim_name == "diameter":
                # 直徑檢查 (如果有 diameter 屬性)
                if hasattr(measurement, "diameter") and measurement.metadata.get("diameter") is not None:
                    actual_value = measurement.metadata.get("diameter")
                    error = abs(actual_value - expected_value)
                    errors.append(error / expected_value if expected_value > 0 else 0)

                    if error > tolerance:
                        failures.append(
                            f"直徑超出容差: {actual_value:.2f} mm "
                            f"(期望: {expected_value:.2f} ± {tolerance:.2f} mm)"
                        )

        # 計算分數
        if len(errors) == 0:
            score = 100.0
        else:
            # 根據相對誤差計算分數
            avg_error = sum(errors) / len(errors)
            score = max(0, 100.0 * (1.0 - avg_error * 2))  # 誤差越大分數越低

        is_pass = len(failures) == 0

        return is_pass, score, failures

    # ==================== 缺陷判斷 ====================

    def judge_defects(
        self, defects: List[DefectResult], spec: ProductSpec
    ) -> tuple[bool, float, List[str]]:
        """
        判斷缺陷是否合格

        Args:
            defects: 缺陷列表
            spec: 產品規格

        Returns:
            (是否合格, 分數, 不合格原因列表)
        """
        failures = []
        total_score = 100.0

        for defect in defects:
            # 檢查是否為關鍵缺陷
            if defect.defect_type in spec.critical_defect_types:
                failures.append(
                    f"發現關鍵缺陷: {defect.defect_type} "
                    f"(深度: {defect.depth:.2f} mm, 面積: {defect.area:.0f} mm²)"
                )
                total_score -= 50.0  # 關鍵缺陷重扣分
                continue

            # 檢查是否為允許的缺陷
            if defect.defect_type in spec.allowed_defect_types:
                # 允許的缺陷,僅略扣分
                if defect.severity == "critical":
                    total_score -= 10.0
                elif defect.severity == "moderate":
                    total_score -= 5.0
                continue

            # 一般缺陷,根據嚴重程度扣分
            if defect.severity == "critical":
                failures.append(
                    f"嚴重缺陷: {defect.defect_type} "
                    f"(深度: {defect.depth:.2f} mm, 面積: {defect.area:.0f} mm²)"
                )
                total_score -= 30.0
            elif defect.severity == "moderate":
                total_score -= 15.0
            else:  # minor
                total_score -= 5.0

            # 檢查深度閾值
            if defect.depth > spec.tolerance.defect_depth_threshold:
                if (
                    f"缺陷深度超出閾值: {defect.defect_type}" not in failures
                ):  # 避免重複
                    failures.append(
                        f"缺陷深度超出閾值: {defect.defect_type} "
                        f"({defect.depth:.2f} mm > {spec.tolerance.defect_depth_threshold:.2f} mm)"
                    )

            # 檢查面積閾值
            if defect.area > spec.tolerance.defect_area_threshold:
                if (
                    f"缺陷面積超出閾值: {defect.defect_type}" not in failures
                ):  # 避免重複
                    failures.append(
                        f"缺陷面積超出閾值: {defect.defect_type} "
                        f"({defect.area:.0f} mm² > {spec.tolerance.defect_area_threshold:.0f} mm²)"
                    )

        # 確保分數不小於 0
        total_score = max(0.0, total_score)

        is_pass = len(failures) == 0 and total_score >= 60.0

        return is_pass, total_score, failures

    # ==================== 組裝判斷 ====================

    def judge_assembly(
        self, assembly_results: List[AssemblyResult], spec: ProductSpec
    ) -> tuple[bool, float, List[str]]:
        """
        判斷組裝是否合格

        Args:
            assembly_results: 組裝驗證結果
            spec: 產品規格

        Returns:
            (是否合格, 分數, 不合格原因列表)
        """
        failures = []
        total_score = 100.0

        # 檢查必須零件
        for required_part in spec.required_parts:
            # 找對應的組裝結果
            result = None
            for r in assembly_results:
                if r.part_name == required_part:
                    result = r
                    break

            if result is None or not result.present:
                failures.append(f"缺少必須零件: {required_part}")
                total_score -= 30.0
                continue

            # 檢查位置
            if not result.position_correct:
                failures.append(
                    f"零件位置錯誤: {required_part} "
                    f"(誤差: {result.position_error:.2f} mm, "
                    f"容差: {spec.tolerance.position_tolerance:.2f} mm)"
                )
                total_score -= 20.0

            # 檢查方向
            if not result.orientation_correct:
                failures.append(
                    f"零件方向錯誤: {required_part} "
                    f"(誤差: {result.orientation_error:.2f}°, "
                    f"容差: {spec.tolerance.angle_tolerance:.2f}°)"
                )
                total_score -= 15.0

        # 確保分數不小於 0
        total_score = max(0.0, total_score)

        is_pass = len(failures) == 0 and total_score >= 70.0

        return is_pass, total_score, failures

    # ==================== 綜合判斷 ====================

    def judge_overall(
        self,
        spec: ProductSpec,
        measurement: Optional[MeasurementResult] = None,
        defects: Optional[List[DefectResult]] = None,
        assembly_results: Optional[List[AssemblyResult]] = None,
    ) -> JudgmentResult:
        """
        綜合判斷

        Args:
            spec: 產品規格
            measurement: 尺寸量測結果 (可選)
            defects: 缺陷列表 (可選)
            assembly_results: 組裝驗證結果 (可選)

        Returns:
            綜合判斷結果
        """
        all_failures = []
        all_warnings = []

        # 尺寸判斷
        dimension_pass = True
        dimension_score = 100.0
        if measurement is not None:
            dimension_pass, dimension_score, dim_failures = self.judge_dimension(
                measurement, spec
            )
            all_failures.extend(dim_failures)

            # 檢查是否需要警告
            if dimension_score < 100.0 and dimension_pass:
                if dimension_score < 90.0:
                    all_warnings.append(f"尺寸接近容差限制 (分數: {dimension_score:.1f})")

        # 缺陷判斷
        defect_pass = True
        defect_score = 100.0
        if defects is not None and len(defects) > 0:
            defect_pass, defect_score, defect_failures = self.judge_defects(
                defects, spec
            )
            all_failures.extend(defect_failures)

            # 檢查是否需要警告
            if defect_score < 100.0 and defect_pass:
                if defect_score < 85.0:
                    all_warnings.append(f"發現輕微缺陷 (分數: {defect_score:.1f})")

        # 組裝判斷
        assembly_pass = True
        assembly_score = 100.0
        if assembly_results is not None and len(assembly_results) > 0:
            assembly_pass, assembly_score, asm_failures = self.judge_assembly(
                assembly_results, spec
            )
            all_failures.extend(asm_failures)

            # 檢查是否需要警告
            if assembly_score < 100.0 and assembly_pass:
                if assembly_score < 80.0:
                    all_warnings.append(f"組裝接近容差限制 (分數: {assembly_score:.1f})")

        # 計算總分 (加權平均)
        weights = []
        scores = []

        if measurement is not None:
            weights.append(0.4)
            scores.append(dimension_score)
        if defects is not None:
            weights.append(0.3)
            scores.append(defect_score)
        if assembly_results is not None:
            weights.append(0.3)
            scores.append(assembly_score)

        if len(scores) == 0:
            overall_score = 0.0
        else:
            # 正規化權重
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            overall_score = sum(s * w for s, w in zip(scores, weights))

        # 決定品質等級
        if not dimension_pass or not defect_pass or not assembly_pass:
            quality_level = QualityLevel.FAIL
        elif len(all_warnings) > 0 or overall_score < 90.0:
            quality_level = QualityLevel.WARNING
        else:
            quality_level = QualityLevel.PASS

        return JudgmentResult(
            quality_level=quality_level,
            overall_score=overall_score,
            dimension_pass=dimension_pass,
            defect_pass=defect_pass,
            assembly_pass=assembly_pass,
            failure_reasons=all_failures,
            warnings=all_warnings,
            dimension_score=dimension_score,
            defect_score=defect_score,
            assembly_score=assembly_score,
            metadata={
                "weights": dict(
                    zip(
                        ["dimension", "defect", "assembly"][: len(weights)],
                        weights,
                    )
                )
            },
        )


if __name__ == "__main__":
    # 測試品質判斷
    print("品質判斷系統測試\n")

    import numpy as np
    from ..measurement import (
        MeasurementResult,
        DefectResult,
        AssemblyResult,
    )
    from .specification import ProductSpec, ToleranceSpec

    judge = QualityJudge()

    # 建立測試規格
    spec = ProductSpec(
        product_id="TEST-001",
        product_name="測試零件",
        category="測試",
        dimensions={"length": 100.0, "width": 50.0, "height": 30.0},
        required_parts=["screw_m6", "nut_hex"],
        tolerance=ToleranceSpec(
            dimension_tolerance=1.0,
            position_tolerance=2.0,
            angle_tolerance=5.0,
            defect_depth_threshold=0.5,
        ),
        critical_defect_types=["crack", "deep_dent"],
        allowed_defect_types=["minor_scratch"],
    )

    # 測試 1: 合格的尺寸
    print("測試 1: 尺寸判斷 (合格)")
    measurement = MeasurementResult(
        dimensions=np.array([100.2, 50.1, 30.0]),  # 略有誤差但在容差內
        volume=150300.0,
        confidence=0.95,
    )
    dim_pass, dim_score, dim_failures = judge.judge_dimension(measurement, spec)
    print(f"  合格: {dim_pass}")
    print(f"  分數: {dim_score:.1f}")
    print(f"  問題: {dim_failures if dim_failures else '無'}")

    # 測試 2: 不合格的尺寸
    print(f"\n測試 2: 尺寸判斷 (不合格)")
    measurement_fail = MeasurementResult(
        dimensions=np.array([102.5, 51.8, 28.0]),  # 超出容差
        volume=148000.0,
        confidence=0.92,
    )
    dim_pass, dim_score, dim_failures = judge.judge_dimension(measurement_fail, spec)
    print(f"  合格: {dim_pass}")
    print(f"  分數: {dim_score:.1f}")
    print(f"  問題:")
    for f in dim_failures:
        print(f"    - {f}")

    # 測試 3: 缺陷判斷
    print(f"\n測試 3: 缺陷判斷")
    defects = [
        DefectResult(
            defect_type="minor_scratch",
            severity="minor",
            depth=0.2,
            area=20.0,
            location=np.array([10, 20, 5]),
            confidence=0.85,
        ),
        DefectResult(
            defect_type="dent",
            severity="moderate",
            depth=0.8,
            area=150.0,
            location=np.array([50, 60, 10]),
            confidence=0.90,
        ),
    ]
    defect_pass, defect_score, defect_failures = judge.judge_defects(defects, spec)
    print(f"  合格: {defect_pass}")
    print(f"  分數: {defect_score:.1f}")
    print(f"  問題: {defect_failures if defect_failures else '無'}")

    # 測試 4: 組裝判斷
    print(f"\n測試 4: 組裝判斷")
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
        AssemblyResult(
            part_name="nut_hex",
            present=True,
            position_correct=False,
            orientation_correct=True,
            position=np.array([50.0, 63.0, 10.0]),
            expected_position=np.array([50.0, 60.0, 10.0]),
            position_error=3.0,  # 超出容差
            confidence=0.90,
        ),
    ]
    asm_pass, asm_score, asm_failures = judge.judge_assembly(assembly_results, spec)
    print(f"  合格: {asm_pass}")
    print(f"  分數: {asm_score:.1f}")
    print(f"  問題:")
    for f in asm_failures:
        print(f"    - {f}")

    # 測試 5: 綜合判斷
    print(f"\n測試 5: 綜合判斷")
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
        print(f"\n  不合格原因:")
        for reason in result.failure_reasons:
            print(f"    - {reason}")

    if result.warnings:
        print(f"\n  警告:")
        for warning in result.warnings:
            print(f"    - {warning}")

    print(f"\n✓ 品質判斷系統測試完成")
