"""
快速模擬相機
移除幀率延遲,用於效能測試
"""

import time
import numpy as np
import cv2
from typing import Optional
from .camera_interface import CameraInterface, RGBDFrame, Intrinsics


class FastMockCamera(CameraInterface):
    """
    快速模擬相機
    移除所有不必要的延遲,專注於效能測試
    """

    def __init__(self, mode: str = "objects"):
        """
        初始化快速模擬相機

        Args:
            mode: 模擬模式
                - "simple": 簡單固定影像 (最快)
                - "objects": 帶物體的場景
        """
        super().__init__()

        self.mode = mode
        self.rgb_width = 1920
        self.rgb_height = 1080
        self.depth_width = 1280
        self.depth_height = 800

        # 預生成固定影像 (避免重複生成)
        self._rgb_cache = None
        self._depth_cache = None
        self._generate_cache()

        self.start_time = 0
        self._frame_count = 0

    def _generate_cache(self):
        """預生成固定影像快取"""
        if self.mode == "simple":
            # 最簡單的固定影像
            self._rgb_cache = np.full(
                (self.rgb_height, self.rgb_width, 3), 128, dtype=np.uint8
            )
            self._depth_cache = np.full(
                (self.depth_height, self.depth_width), 1000.0, dtype=np.float32
            )

        elif self.mode == "objects":
            # 帶物體的場景
            rgb = np.zeros((self.rgb_height, self.rgb_width, 3), dtype=np.uint8)
            rgb[:] = (220, 230, 220)  # 淡綠色背景

            # 繪製物體
            cv2.rectangle(rgb, (300, 300), (600, 500), (150, 130, 100), -1)
            cv2.circle(rgb, (1000, 400), 150, (180, 150, 120), -1)
            cv2.ellipse(rgb, (450, 700), (120, 60), 0, 0, 360, (80, 80, 80), -1)

            self._rgb_cache = rgb

            # 深度場景
            depth = np.ones((self.depth_height, self.depth_width), dtype=np.float32) * 1500

            # 計算縮放比例
            scale_x = self.depth_width / self.rgb_width
            scale_y = self.depth_height / self.rgb_height

            # 物體深度
            x1, y1, x2, y2 = int(300 * scale_x), int(300 * scale_y), int(600 * scale_x), int(
                500 * scale_y
            )
            depth[y1:y2, x1:x2] = 1000

            cx, cy, r = int(1000 * scale_x), int(400 * scale_y), int(150 * scale_y)
            y_coords, x_coords = np.ogrid[: self.depth_height, : self.depth_width]
            mask = (x_coords - cx) ** 2 + (y_coords - cy) ** 2 <= r**2
            depth[mask] = 1200

            self._depth_cache = depth

    def connect(self, serial_number: Optional[str] = None) -> bool:
        """連接快速模擬相機"""
        self._is_connected = True
        self.start_time = time.time()
        return True

    def disconnect(self) -> bool:
        """中斷快速模擬相機連接"""
        self._is_connected = False
        return True

    def start_stream(self) -> bool:
        """開始串流"""
        if not self._is_connected:
            return False

        self._is_streaming = True
        return True

    def stop_stream(self) -> bool:
        """停止串流"""
        self._is_streaming = False
        return True

    def get_frame(self, timeout: int = 1000) -> RGBDFrame:
        """
        取得一幀 RGB-D 資料 (無延遲)

        Args:
            timeout: 超時時間 (ms) - 忽略

        Returns:
            RGBDFrame
        """
        if not self._is_streaming:
            raise RuntimeError("相機未在串流中")

        # 直接返回快取的影像 (無任何延遲)
        self._frame_count += 1

        frame = RGBDFrame(
            rgb=self._rgb_cache.copy(),
            depth=self._depth_cache.copy(),
            timestamp=time.time() - self.start_time,
            frame_number=self._frame_count,
        )

        return frame

    def get_rgb_intrinsics(self) -> Optional[Intrinsics]:
        """取得 RGB 相機內參"""
        if not self._is_connected:
            return None

        return Intrinsics(
            fx=1081.37,
            fy=1081.37,
            cx=self.rgb_width / 2,
            cy=self.rgb_height / 2,
            width=self.rgb_width,
            height=self.rgb_height,
            distortion=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        )

    def get_depth_intrinsics(self) -> Optional[Intrinsics]:
        """取得深度相機內參"""
        if not self._is_connected:
            return None

        return Intrinsics(
            fx=720.91,
            fy=720.91,
            cx=self.depth_width / 2,
            cy=self.depth_height / 2,
            width=self.depth_width,
            height=self.depth_height,
            distortion=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        )

    def get_camera_info(self):
        """取得相機資訊"""
        from .camera_interface import CameraInfo

        return CameraInfo(
            name="FastMockCamera",
            serial_number="FAST-MOCK-001",
            firmware_version="1.0.0",
            usb_type="USB 3.0",
        )

    def set_rgb_resolution(self, width: int, height: int) -> bool:
        """設定 RGB 解析度"""
        return True

    def set_depth_resolution(self, width: int, height: int) -> bool:
        """設定深度解析度"""
        return True

    def set_fps(self, fps: int) -> bool:
        """設定幀率"""
        return True

    def set_depth_range(self, min_depth: int, max_depth: int) -> bool:
        """設定深度範圍"""
        return True

    def enable_alignment(self, align_to_rgb: bool = True) -> bool:
        """啟用對齊"""
        return True

    def start_streaming(self) -> bool:
        """開始串流 (別名)"""
        return self.start_stream()

    def stop_streaming(self) -> bool:
        """停止串流 (別名)"""
        return self.stop_stream()


if __name__ == "__main__":
    # 測試快速模擬相機
    print("快速模擬相機測試\n")

    camera = FastMockCamera(mode="objects")

    print(f"模式: {camera.mode}")
    print(f"RGB 解析度: {camera.rgb_width} x {camera.rgb_height}")
    print(f"深度解析度: {camera.depth_width} x {camera.depth_height}")

    # 連接並開始串流
    camera.connect()
    camera.start_stream()

    # 效能測試
    num_frames = 100
    start_time = time.time()

    for i in range(num_frames):
        frame = camera.get_frame()

    elapsed = time.time() - start_time
    fps = num_frames / elapsed

    print(f"\n效能測試:")
    print(f"  擷取 {num_frames} 幀")
    print(f"  總耗時: {elapsed:.3f} 秒")
    print(f"  平均幀時間: {elapsed/num_frames*1000:.2f} ms")
    print(f"  平均 FPS: {fps:.1f}")

    camera.stop_stream()
    camera.disconnect()

    print(f"\n✓ 快速模擬相機測試完成")
