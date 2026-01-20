"""
量測模組
包含 3D 尺寸量測、缺陷深度分析、組裝驗證等功能
"""

from .dimension import DimensionMeasurement, MeasurementResult
from .defect import DefectAnalyzer, DefectResult
from .assembly import AssemblyVerifier, AssemblyResult

__all__ = [
    # 尺寸量測
    "DimensionMeasurement",
    "MeasurementResult",
    # 缺陷分析
    "DefectAnalyzer",
    "DefectResult",
    # 組裝驗證
    "AssemblyVerifier",
    "AssemblyResult",
]
