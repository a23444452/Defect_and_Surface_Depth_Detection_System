"""
硬體介面模組
包含相機驅動與硬體抽象層
"""

from .camera_interface import (
    CameraInterface,
    CameraInfo,
    Intrinsics,
    RGBDFrame,
    CameraException,
    CameraConnectionError,
    CameraStreamingError,
    CameraTimeoutError,
    CameraConfigurationError,
)
from .mock_camera import MockCamera
from .fast_mock_camera import FastMockCamera

# 嘗試匯入 Gemini2Camera
try:
    from .gemini2_driver import Gemini2Camera

    GEMINI2_AVAILABLE = True
except ImportError:
    GEMINI2_AVAILABLE = False
    Gemini2Camera = None

__all__ = [
    # 介面與資料類別
    "CameraInterface",
    "CameraInfo",
    "Intrinsics",
    "RGBDFrame",
    # 異常
    "CameraException",
    "CameraConnectionError",
    "CameraStreamingError",
    "CameraTimeoutError",
    "CameraConfigurationError",
    # 驅動
    "MockCamera",
    "FastMockCamera",
    "Gemini2Camera",
    "GEMINI2_AVAILABLE",
]
