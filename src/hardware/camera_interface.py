"""
相機抽象介面
定義所有 RGB-D 相機驅動需要實作的標準介面
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class CameraInfo:
    """相機資訊"""

    model: str
    serial_number: str
    firmware_version: str
    rgb_resolution: Tuple[int, int]  # (width, height)
    depth_resolution: Tuple[int, int]  # (width, height)
    fps: int


@dataclass
class Intrinsics:
    """相機內參"""

    fx: float  # 焦距 x
    fy: float  # 焦距 y
    cx: float  # 主點 x
    cy: float  # 主點 y
    width: int  # 影像寬度
    height: int  # 影像高度
    distortion: np.ndarray  # 畸變係數 [k1, k2, p1, p2, k3]


@dataclass
class RGBDFrame:
    """RGB-D 影像幀"""

    rgb: np.ndarray  # RGB 影像 (H, W, 3) - BGR 格式
    depth: np.ndarray  # 深度影像 (H, W) - 單位: mm
    timestamp: float  # 時間戳 (秒)
    frame_number: int  # 幀編號
    is_aligned: bool = False  # 是否已對齊


class CameraInterface(ABC):
    """
    RGB-D 相機抽象介面
    所有相機驅動必須繼承此類並實作所有抽象方法
    """

    def __init__(self):
        """初始化相機介面"""
        self._is_connected = False
        self._is_streaming = False
        self._frame_count = 0

    @abstractmethod
    def connect(self, serial_number: Optional[str] = None) -> bool:
        """
        連接相機

        Args:
            serial_number: 相機序號，None 則使用第一台檢測到的相機

        Returns:
            是否成功連接
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        中斷相機連接

        Returns:
            是否成功中斷
        """
        pass

    @abstractmethod
    def start_streaming(self) -> bool:
        """
        開始串流

        Returns:
            是否成功開始
        """
        pass

    @abstractmethod
    def stop_streaming(self) -> bool:
        """
        停止串流

        Returns:
            是否成功停止
        """
        pass

    @abstractmethod
    def get_frame(self, timeout: int = 5000) -> Optional[RGBDFrame]:
        """
        取得一幀 RGB-D 影像

        Args:
            timeout: 超時時間 (ms)

        Returns:
            RGBDFrame 或 None (如果超時)
        """
        pass

    @abstractmethod
    def get_camera_info(self) -> Optional[CameraInfo]:
        """
        取得相機資訊

        Returns:
            CameraInfo 或 None
        """
        pass

    @abstractmethod
    def get_rgb_intrinsics(self) -> Optional[Intrinsics]:
        """
        取得 RGB 相機內參

        Returns:
            Intrinsics 或 None
        """
        pass

    @abstractmethod
    def get_depth_intrinsics(self) -> Optional[Intrinsics]:
        """
        取得深度相機內參

        Returns:
            Intrinsics 或 None
        """
        pass

    @abstractmethod
    def set_rgb_resolution(self, width: int, height: int) -> bool:
        """
        設定 RGB 解析度

        Args:
            width: 寬度
            height: 高度

        Returns:
            是否成功設定
        """
        pass

    @abstractmethod
    def set_depth_resolution(self, width: int, height: int) -> bool:
        """
        設定深度解析度

        Args:
            width: 寬度
            height: 高度

        Returns:
            是否成功設定
        """
        pass

    @abstractmethod
    def set_fps(self, fps: int) -> bool:
        """
        設定幀率

        Args:
            fps: 幀率

        Returns:
            是否成功設定
        """
        pass

    @abstractmethod
    def enable_alignment(self, enabled: bool = True) -> bool:
        """
        啟用/停用 RGB-D 對齊

        Args:
            enabled: 是否啟用

        Returns:
            是否成功設定
        """
        pass

    @abstractmethod
    def set_depth_range(self, min_depth: int, max_depth: int) -> bool:
        """
        設定深度範圍

        Args:
            min_depth: 最小深度 (mm)
            max_depth: 最大深度 (mm)

        Returns:
            是否成功設定
        """
        pass

    # 便利屬性
    @property
    def is_connected(self) -> bool:
        """是否已連接"""
        return self._is_connected

    @property
    def is_streaming(self) -> bool:
        """是否正在串流"""
        return self._is_streaming

    @property
    def frame_count(self) -> int:
        """已擷取的幀數"""
        return self._frame_count

    # Context Manager 支援
    def __enter__(self):
        """進入 context manager"""
        if not self.is_connected:
            self.connect()
        if not self.is_streaming:
            self.start_streaming()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """離開 context manager"""
        if self.is_streaming:
            self.stop_streaming()
        if self.is_connected:
            self.disconnect()
        return False


class CameraException(Exception):
    """相機相關異常"""

    pass


class CameraConnectionError(CameraException):
    """相機連接錯誤"""

    pass


class CameraStreamingError(CameraException):
    """相機串流錯誤"""

    pass


class CameraTimeoutError(CameraException):
    """相機超時錯誤"""

    pass


class CameraConfigurationError(CameraException):
    """相機配置錯誤"""

    pass


if __name__ == "__main__":
    # 測試程式碼
    print("相機抽象介面定義")
    print("\n資料類別:")
    print(f"  - CameraInfo: 相機資訊")
    print(f"  - Intrinsics: 相機內參")
    print(f"  - RGBDFrame: RGB-D 影像幀")

    print("\n介面方法:")
    methods = [
        method
        for method in dir(CameraInterface)
        if not method.startswith("_") and callable(getattr(CameraInterface, method))
    ]
    for method in methods:
        print(f"  - {method}()")

    print("\n異常類別:")
    print(f"  - CameraException")
    print(f"  - CameraConnectionError")
    print(f"  - CameraStreamingError")
    print(f"  - CameraTimeoutError")
    print(f"  - CameraConfigurationError")

    print("\n✓ 相機抽象介面定義完成")
