"""
組裝驗證模組
驗證零件存在性、位置、方向等
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


@dataclass
class AssemblyResult:
    """組裝驗證結果"""

    part_name: str  # 零件名稱
    present: bool  # 是否存在
    position_correct: bool  # 位置是否正確
    orientation_correct: bool  # 方向是否正確
    position: Optional[np.ndarray] = None  # 實際位置 (3,)
    expected_position: Optional[np.ndarray] = None  # 預期位置 (3,)
    position_error: float = 0.0  # 位置誤差 (mm)
    orientation_error: float = 0.0  # 方向誤差 (度)
    confidence: float = 0.0  # 信心度 [0-1]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AssemblyVerifier:
    """
    組裝驗證器
    驗證零件的組裝狀態
    """

    def __init__(
        self, position_tolerance: float = 5.0, angle_tolerance: float = 5.0
    ):
        """
        初始化組裝驗證器

        Args:
            position_tolerance: 位置容差 (mm)
            angle_tolerance: 角度容差 (度)
        """
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance

    # ==================== 零件存在性檢查 ====================

    def check_part_presence(
        self, detection_results: List[Dict[str, Any]], expected_class: str
    ) -> bool:
        """
        檢查零件是否存在

        Args:
            detection_results: 檢測結果列表
            expected_class: 預期類別

        Returns:
            是否存在
        """
        for result in detection_results:
            if result.get("class_name") == expected_class:
                return True

        return False

    # ==================== 位置驗證 ====================

    def verify_position(
        self,
        actual_position: np.ndarray,
        expected_position: np.ndarray,
        tolerance: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        驗證位置是否正確

        Args:
            actual_position: 實際位置 (3,) [x, y, z]
            expected_position: 預期位置 (3,) [x, y, z]
            tolerance: 容差 (mm), None 則使用預設值

        Returns:
            (是否正確, 誤差)
        """
        if tolerance is None:
            tolerance = self.position_tolerance

        # 計算歐氏距離
        error = float(np.linalg.norm(actual_position - expected_position))

        is_correct = error <= tolerance

        return is_correct, error

    # ==================== 方向驗證 ====================

    def verify_orientation(
        self,
        actual_rotation: np.ndarray,
        expected_rotation: np.ndarray,
        tolerance: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        驗證方向是否正確

        Args:
            actual_rotation: 實際旋轉矩陣 (3, 3)
            expected_rotation: 預期旋轉矩陣 (3, 3)
            tolerance: 容差 (度), None 則使用預設值

        Returns:
            (是否正確, 角度誤差)
        """
        if tolerance is None:
            tolerance = self.angle_tolerance

        # 計算旋轉矩陣差異
        R_diff = actual_rotation.T @ expected_rotation

        # 轉換為旋轉角度
        trace = np.trace(R_diff)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        angle_deg = float(np.degrees(angle))

        is_correct = angle_deg <= tolerance

        return is_correct, angle_deg

    # ==================== 綜合組裝驗證 ====================

    def verify_assembly(
        self,
        part_name: str,
        detection_results: List[Dict[str, Any]],
        expected_position: np.ndarray,
        expected_rotation: Optional[np.ndarray] = None,
    ) -> AssemblyResult:
        """
        綜合組裝驗證

        Args:
            part_name: 零件名稱
            detection_results: 檢測結果列表
            expected_position: 預期位置 (3,)
            expected_rotation: 預期旋轉 (3, 3), 可選

        Returns:
            AssemblyResult
        """
        # 1. 檢查存在性
        present = self.check_part_presence(detection_results, part_name)

        if not present:
            return AssemblyResult(
                part_name=part_name,
                present=False,
                position_correct=False,
                orientation_correct=False,
                expected_position=expected_position,
                confidence=1.0,  # 確定不存在
            )

        # 2. 找到對應的檢測結果
        part_result = None
        for result in detection_results:
            if result.get("class_name") == part_name:
                part_result = result
                break

        if part_result is None:
            return AssemblyResult(
                part_name=part_name,
                present=False,
                position_correct=False,
                orientation_correct=False,
            )

        # 3. 驗證位置
        actual_position = part_result.get("position", np.zeros(3))
        position_correct, position_error = self.verify_position(
            actual_position, expected_position
        )

        # 4. 驗證方向 (如果提供)
        orientation_correct = True
        orientation_error = 0.0

        if expected_rotation is not None:
            actual_rotation = part_result.get("rotation", np.eye(3))
            orientation_correct, orientation_error = self.verify_orientation(
                actual_rotation, expected_rotation
            )

        # 5. 建立結果
        confidence = part_result.get("confidence", 0.0)

        return AssemblyResult(
            part_name=part_name,
            present=True,
            position_correct=position_correct,
            orientation_correct=orientation_correct,
            position=actual_position,
            expected_position=expected_position,
            position_error=position_error,
            orientation_error=orientation_error,
            confidence=confidence,
            metadata={"detection_result": part_result},
        )

    # ==================== 批次驗證 ====================

    def verify_multiple_parts(
        self,
        detection_results: List[Dict[str, Any]],
        expected_assembly: Dict[str, Dict[str, Any]],
    ) -> List[AssemblyResult]:
        """
        批次驗證多個零件

        Args:
            detection_results: 檢測結果列表
            expected_assembly: 預期組裝配置
                {
                    "part_name": {
                        "position": [x, y, z],
                        "rotation": [[...], [...], [...]],  # 可選
                    },
                    ...
                }

        Returns:
            驗證結果列表
        """
        results = []

        for part_name, expected in expected_assembly.items():
            expected_pos = np.array(expected["position"])
            expected_rot = (
                np.array(expected["rotation"]) if "rotation" in expected else None
            )

            result = self.verify_assembly(
                part_name, detection_results, expected_pos, expected_rot
            )

            results.append(result)

        return results

    # ==================== 組裝統計 ====================

    def get_assembly_statistics(
        self, results: List[AssemblyResult]
    ) -> Dict[str, Any]:
        """
        取得組裝統計資訊

        Args:
            results: 驗證結果列表

        Returns:
            統計資訊
        """
        total = len(results)
        present = sum(1 for r in results if r.present)
        position_correct = sum(1 for r in results if r.position_correct)
        orientation_correct = sum(1 for r in results if r.orientation_correct)

        avg_position_error = (
            np.mean([r.position_error for r in results if r.present])
            if present > 0
            else 0.0
        )

        avg_orientation_error = (
            np.mean([r.orientation_error for r in results if r.present])
            if present > 0
            else 0.0
        )

        return {
            "total_parts": total,
            "present": present,
            "missing": total - present,
            "position_correct": position_correct,
            "orientation_correct": orientation_correct,
            "avg_position_error": float(avg_position_error),
            "avg_orientation_error": float(avg_orientation_error),
            "pass_rate": position_correct / total if total > 0 else 0.0,
        }


if __name__ == "__main__":
    # 測試組裝驗證
    print("組裝驗證模組測試\n")

    verifier = AssemblyVerifier(position_tolerance=5.0, angle_tolerance=5.0)

    # 模擬檢測結果
    detection_results = [
        {
            "class_name": "screw_m6",
            "position": np.array([10.0, 20.0, 5.0]),
            "rotation": np.eye(3),
            "confidence": 0.95,
        },
        {
            "class_name": "nut_hex",
            "position": np.array([50.2, 60.1, 10.0]),
            "rotation": np.eye(3),
            "confidence": 0.92,
        },
        {
            "class_name": "washer",
            "position": np.array([100.0, 100.0, 15.0]),
            "rotation": np.eye(3),
            "confidence": 0.88,
        },
    ]

    # 預期組裝配置
    expected_assembly = {
        "screw_m6": {"position": [10.0, 20.0, 5.0]},  # 位置正確
        "nut_hex": {"position": [50.0, 60.0, 10.0]},  # 位置略有偏差
        "washer": {"position": [100.0, 100.0, 15.0]},  # 位置正確
        "bracket": {"position": [150.0, 150.0, 20.0]},  # 缺失
    }

    # 測試 1: 單一零件驗證
    print("測試 1: 單一零件驗證")
    result = verifier.verify_assembly(
        "screw_m6", detection_results, np.array([10.0, 20.0, 5.0])
    )

    print(f"  零件: {result.part_name}")
    print(f"  存在: {result.present}")
    print(f"  位置正確: {result.position_correct}")
    print(f"  位置誤差: {result.position_error:.2f} mm")
    print(f"  信心度: {result.confidence:.2f}")

    # 測試 2: 批次驗證
    print(f"\n測試 2: 批次驗證")
    results = verifier.verify_multiple_parts(detection_results, expected_assembly)

    print(f"  驗證 {len(results)} 個零件:")
    for r in results:
        status = "✓" if r.present and r.position_correct else "✗"
        print(f"    {status} {r.part_name}:")
        print(f"       存在: {r.present}")
        if r.present:
            print(f"       位置正確: {r.position_correct} (誤差: {r.position_error:.2f}mm)")

    # 測試 3: 組裝統計
    print(f"\n測試 3: 組裝統計")
    stats = verifier.get_assembly_statistics(results)

    print(f"  總零件數: {stats['total_parts']}")
    print(f"  存在: {stats['present']}")
    print(f"  缺失: {stats['missing']}")
    print(f"  位置正確: {stats['position_correct']}")
    print(f"  平均位置誤差: {stats['avg_position_error']:.2f} mm")
    print(f"  通過率: {stats['pass_rate']*100:.1f}%")

    # 測試 4: 方向驗證
    print(f"\n測試 4: 方向驗證")

    # 建立兩個旋轉矩陣 (繞 Z 軸旋轉 3 度)
    angle = np.radians(3)
    R1 = np.eye(3)
    R2 = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    orientation_correct, angle_error = verifier.verify_orientation(R1, R2)
    print(f"  方向正確: {orientation_correct}")
    print(f"  角度誤差: {angle_error:.2f}°")

    print(f"\n✓ 組裝驗證模組測試完成")
