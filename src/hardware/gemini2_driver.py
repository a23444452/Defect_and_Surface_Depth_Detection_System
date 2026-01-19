"""
ORBBEC Gemini 2 相機驅動
基於 OrbbecSDK 的 Gemini 2 相機驅動實作
"""

import time
from typing import Optional
import numpy as np
import cv2

from .camera_interface import (
    CameraInterface,
    CameraInfo,
    Intrinsics,
    RGBDFrame,
    CameraConnectionError,
    CameraStreamingError,
    CameraTimeoutError,
    CameraConfigurationError,
)

# 嘗試匯入 OrbbecSDK
try:
    import pyorbbecsdk as ob

    ORBBEC_SDK_AVAILABLE = True
except ImportError:
    ORBBEC_SDK_AVAILABLE = False
    print("警告: OrbbecSDK 未安裝，將使用模擬模式")


class Gemini2Camera(CameraInterface):
    """
    ORBBEC Gemini 2 相機驅動
    實作 CameraInterface 介面
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 Gemini 2 相機

        Args:
            config_path: 配置檔案路徑（可選）
        """
        super().__init__()

        if not ORBBEC_SDK_AVAILABLE:
            raise ImportError("OrbbecSDK 未安裝，請執行: pip install pyorbbecsdk")

        self.config_path = config_path
        self.context = None
        self.device = None
        self.pipeline = None
        self.config = None

        # 預設配置
        self.rgb_width = 1920
        self.rgb_height = 1080
        self.depth_width = 1280
        self.depth_height = 800
        self.fps = 30
        self.alignment_enabled = True

    def connect(self, serial_number: Optional[str] = None) -> bool:
        """
        連接相機

        Args:
            serial_number: 相機序號

        Returns:
            是否成功連接
        """
        try:
            # 建立 context
            self.context = ob.Context()

            # 取得設備列表
            device_list = self.context.query_devices()
            if device_list.get_count() == 0:
                raise CameraConnectionError("未檢測到任何 ORBBEC 相機")

            # 選擇設備
            if serial_number:
                # 根據序號選擇
                device_index = None
                for i in range(device_list.get_count()):
                    device = device_list.get_device(i)
                    info = device.get_device_info()
                    if info.get_serial_number() == serial_number:
                        device_index = i
                        break

                if device_index is None:
                    raise CameraConnectionError(f"找不到序號為 {serial_number} 的相機")

                self.device = device_list.get_device(device_index)
            else:
                # 使用第一台相機
                self.device = device_list.get_device(0)

            # 建立 pipeline
            self.pipeline = ob.Pipeline(self.device)

            self._is_connected = True
            return True

        except Exception as e:
            raise CameraConnectionError(f"連接相機失敗: {e}")

    def disconnect(self) -> bool:
        """
        中斷相機連接

        Returns:
            是否成功中斷
        """
        try:
            if self.is_streaming:
                self.stop_streaming()

            if self.pipeline:
                self.pipeline = None

            if self.device:
                self.device = None

            if self.context:
                self.context = None

            self._is_connected = False
            return True

        except Exception as e:
            print(f"中斷連接時發生錯誤: {e}")
            return False

    def start_streaming(self) -> bool:
        """
        開始串流

        Returns:
            是否成功開始
        """
        if not self.is_connected:
            raise CameraConnectionError("相機未連接")

        try:
            # 建立 config
            self.config = ob.Config()

            # 配置 RGB 串流
            rgb_profile_list = self.pipeline.get_stream_profile_list(ob.SensorType.COLOR_SENSOR)
            rgb_profile = rgb_profile_list.get_video_stream_profile(
                self.rgb_width, self.rgb_height, ob.Format.RGB888, self.fps
            )
            self.config.enable_stream(rgb_profile)

            # 配置深度串流
            depth_profile_list = self.pipeline.get_stream_profile_list(ob.SensorType.DEPTH_SENSOR)
            depth_profile = depth_profile_list.get_video_stream_profile(
                self.depth_width, self.depth_height, ob.Format.Y16, self.fps
            )
            self.config.enable_stream(depth_profile)

            # 啟用對齊
            if self.alignment_enabled:
                self.config.set_align_mode(ob.AlignMode.ALIGN_D2C_HW_MODE)

            # 啟動 pipeline
            self.pipeline.start(self.config)

            self._is_streaming = True
            self._frame_count = 0
            return True

        except Exception as e:
            raise CameraStreamingError(f"啟動串流失敗: {e}")

    def stop_streaming(self) -> bool:
        """
        停止串流

        Returns:
            是否成功停止
        """
        try:
            if self.pipeline and self.is_streaming:
                self.pipeline.stop()

            self._is_streaming = False
            return True

        except Exception as e:
            print(f"停止串流時發生錯誤: {e}")
            return False

    def get_frame(self, timeout: int = 5000) -> Optional[RGBDFrame]:
        """
        取得一幀 RGB-D 影像

        Args:
            timeout: 超時時間 (ms)

        Returns:
            RGBDFrame 或 None
        """
        if not self.is_streaming:
            raise CameraStreamingError("相機未在串流中")

        try:
            # 等待幀集
            frames = self.pipeline.wait_for_frames(timeout)
            if frames is None:
                raise CameraTimeoutError("等待幀超時")

            # 取得 RGB 幀
            rgb_frame = frames.get_color_frame()
            if rgb_frame is None:
                return None

            # 取得深度幀
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                return None

            # 轉換 RGB 影像
            rgb_data = np.frombuffer(rgb_frame.get_data(), dtype=np.uint8)
            rgb_image = rgb_data.reshape((rgb_frame.get_height(), rgb_frame.get_width(), 3))
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # 轉換為 BGR

            # 轉換深度影像
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_image = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))

            # 取得時間戳
            timestamp = rgb_frame.get_timestamp() / 1000000.0  # 轉換為秒

            # 建立 RGBDFrame
            self._frame_count += 1
            frame = RGBDFrame(
                rgb=rgb_image,
                depth=depth_image.astype(np.float32),
                timestamp=timestamp,
                frame_number=self._frame_count,
                is_aligned=self.alignment_enabled,
            )

            return frame

        except ob.OBException as e:
            if "timeout" in str(e).lower():
                raise CameraTimeoutError(f"取得幀超時: {e}")
            else:
                raise CameraStreamingError(f"取得幀失敗: {e}")

    def get_camera_info(self) -> Optional[CameraInfo]:
        """
        取得相機資訊

        Returns:
            CameraInfo 或 None
        """
        if not self.is_connected:
            return None

        try:
            device_info = self.device.get_device_info()

            info = CameraInfo(
                model=device_info.get_name(),
                serial_number=device_info.get_serial_number(),
                firmware_version=device_info.get_firmware_version(),
                rgb_resolution=(self.rgb_width, self.rgb_height),
                depth_resolution=(self.depth_width, self.depth_height),
                fps=self.fps,
            )

            return info

        except Exception as e:
            print(f"取得相機資訊時發生錯誤: {e}")
            return None

    def get_rgb_intrinsics(self) -> Optional[Intrinsics]:
        """
        取得 RGB 相機內參

        Returns:
            Intrinsics 或 None
        """
        if not self.is_streaming:
            return None

        try:
            rgb_profile = self.pipeline.get_stream_profile_list(
                ob.SensorType.COLOR_SENSOR
            ).get_video_stream_profile(
                self.rgb_width, self.rgb_height, ob.Format.RGB888, self.fps
            )

            intrinsic = rgb_profile.get_intrinsic()
            distortion = rgb_profile.get_distortion()

            return Intrinsics(
                fx=intrinsic.fx,
                fy=intrinsic.fy,
                cx=intrinsic.cx,
                cy=intrinsic.cy,
                width=intrinsic.width,
                height=intrinsic.height,
                distortion=np.array([distortion.k1, distortion.k2, distortion.p1, distortion.p2, distortion.k3]),
            )

        except Exception as e:
            print(f"取得 RGB 內參時發生錯誤: {e}")
            return None

    def get_depth_intrinsics(self) -> Optional[Intrinsics]:
        """
        取得深度相機內參

        Returns:
            Intrinsics 或 None
        """
        if not self.is_streaming:
            return None

        try:
            depth_profile = self.pipeline.get_stream_profile_list(
                ob.SensorType.DEPTH_SENSOR
            ).get_video_stream_profile(
                self.depth_width, self.depth_height, ob.Format.Y16, self.fps
            )

            intrinsic = depth_profile.get_intrinsic()
            distortion = depth_profile.get_distortion()

            return Intrinsics(
                fx=intrinsic.fx,
                fy=intrinsic.fy,
                cx=intrinsic.cx,
                cy=intrinsic.cy,
                width=intrinsic.width,
                height=intrinsic.height,
                distortion=np.array([distortion.k1, distortion.k2, distortion.p1, distortion.p2, distortion.k3]),
            )

        except Exception as e:
            print(f"取得深度內參時發生錯誤: {e}")
            return None

    def set_rgb_resolution(self, width: int, height: int) -> bool:
        """設定 RGB 解析度"""
        if self.is_streaming:
            raise CameraConfigurationError("無法在串流中修改解析度")

        self.rgb_width = width
        self.rgb_height = height
        return True

    def set_depth_resolution(self, width: int, height: int) -> bool:
        """設定深度解析度"""
        if self.is_streaming:
            raise CameraConfigurationError("無法在串流中修改解析度")

        self.depth_width = width
        self.depth_height = height
        return True

    def set_fps(self, fps: int) -> bool:
        """設定幀率"""
        if self.is_streaming:
            raise CameraConfigurationError("無法在串流中修改幀率")

        self.fps = fps
        return True

    def enable_alignment(self, enabled: bool = True) -> bool:
        """啟用/停用 RGB-D 對齊"""
        if self.is_streaming:
            raise CameraConfigurationError("無法在串流中修改對齊設定")

        self.alignment_enabled = enabled
        return True

    def set_depth_range(self, min_depth: int, max_depth: int) -> bool:
        """設定深度範圍"""
        # Gemini 2 的深度範圍是硬體固定的
        # 這裡只是儲存，實際濾波會在後處理進行
        self.min_depth = min_depth
        self.max_depth = max_depth
        return True


if __name__ == "__main__":
    # 測試程式碼
    print("ORBBEC Gemini 2 相機驅動")
    print(f"OrbbecSDK 可用: {ORBBEC_SDK_AVAILABLE}")

    if ORBBEC_SDK_AVAILABLE:
        print("\n✓ OrbbecSDK 已安裝")
        print("  可以使用真實相機進行測試")
    else:
        print("\n⚠ OrbbecSDK 未安裝")
        print("  請執行: pip install pyorbbecsdk")
        print("  或參考: https://github.com/orbbec/pyorbbecsdk")
