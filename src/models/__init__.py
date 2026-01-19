"""
AI 模型模組
包含物體檢測與實例分割模型
"""

from .detector_interface import (
    DetectorInterface,
    DetectionBox,
    SegmentationMask,
    DetectionResult,
    DetectorException,
    ModelLoadError,
    InferenceError,
    PreprocessError,
    PostprocessError,
)

from .yolo_detector import YOLOv11Detector

__all__ = [
    # 介面與資料類別
    "DetectorInterface",
    "DetectionBox",
    "SegmentationMask",
    "DetectionResult",
    # 異常
    "DetectorException",
    "ModelLoadError",
    "InferenceError",
    "PreprocessError",
    "PostprocessError",
    # 檢測器
    "YOLOv11Detector",
]
