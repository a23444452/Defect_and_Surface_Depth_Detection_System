"""
模擬相機驅動
用於無實體硬體的開發與測試
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
)


class MockCamera(CameraInterface):
    """
    模擬相機驅動
    生成模擬的 RGB-D 資料用於開發與測試
    """

    def __init__(self, mode: str = "random"):
        """
        初始化模擬相機

        Args:
            mode: 模擬模式
                - "random": 隨機影像
                - "pattern": 圖樣影像
                - "objects": 帶物體的場景
        """
        super().__init__()

        self.mode = mode
        self.rgb_width = 1920
        self.rgb_height = 1080
        self.depth_width = 1280
        self.depth_height = 800
        self.fps = 30
        self.alignment_enabled = True

        self.start_time = 0
        self.last_frame_time = 0

    def connect(self, serial_number: Optional[str] = None) -> bool:
        """連接模擬相機"""
        self._is_connected = True
        self.start_time = time.time()
        return True

    def disconnect(self) -> bool:
        """中斷模擬相機連接"""
        self._is_connected = False
        return True

    def start_streaming(self) -> bool:
        """開始模擬串流"""
        if not self.is_connected:
            raise CameraConnectionError("相機未連接")

        self._is_streaming = True
        self._frame_count = 0
        self.last_frame_time = time.time()
        return True

    def stop_streaming(self) -> bool:
        """停止模擬串流"""
        self._is_streaming = False
        return True

    def get_frame(self, timeout: int = 5000) -> Optional[RGBDFrame]:
        """
        取得一幀模擬 RGB-D 影像

        Args:
            timeout: 超時時間 (ms)

        Returns:
            RGBDFrame
        """
        if not self.is_streaming:
            raise CameraStreamingError("相機未在串流中")

        # 模擬幀率延遲
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        frame_interval = 1.0 / self.fps

        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

        # 生成 RGB 影像
        rgb_image = self._generate_rgb_image()

        # 生成深度影像
        depth_image = self._generate_depth_image()

        # 建立幀
        self._frame_count += 1
        self.last_frame_time = time.time()

        frame = RGBDFrame(
            rgb=rgb_image,
            depth=depth_image,
            timestamp=time.time() - self.start_time,
            frame_number=self._frame_count,
            is_aligned=self.alignment_enabled,
        )

        return frame

    def _generate_rgb_image(self) -> np.ndarray:
        """生成 RGB 影像"""
        if self.mode == "random":
            # 隨機噪點影像
            image = np.random.randint(0, 256, (self.rgb_height, self.rgb_width, 3), dtype=np.uint8)

        elif self.mode == "pattern":
            # 棋盤格圖樣
            image = np.zeros((self.rgb_height, self.rgb_width, 3), dtype=np.uint8)
            square_size = 100

            for i in range(0, self.rgb_height, square_size):
                for j in range(0, self.rgb_width, square_size):
                    if ((i // square_size) + (j // square_size)) % 2 == 0:
                        image[i : i + square_size, j : j + square_size] = [200, 200, 200]

        elif self.mode == "objects":
            # 帶物體的場景
            image = np.zeros((self.rgb_height, self.rgb_width, 3), dtype=np.uint8)
            image[:] = (220, 230, 220)  # 淡綠色背景

            # 繪製幾個物體
            # 金屬零件 (藍灰色)
            cv2.rectangle(image, (300, 300), (600, 500), (150, 130, 100), -1)
            cv2.rectangle(image, (300, 300), (600, 500), (100, 80, 60), 3)

            # 塑膠零件 (淺藍色)
            cv2.circle(image, (1000, 400), 150, (180, 150, 120), -1)
            cv2.circle(image, (1000, 400), 150, (130, 100, 80), 3)

            # 缺陷 (深色)
            cv2.ellipse(image, (450, 700), (120, 60), 0, 0, 360, (80, 80, 80), -1)

            # 加入一些噪點
            noise = np.random.randint(-15, 15, (self.rgb_height, self.rgb_width, 3), dtype=np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        else:
            image = np.zeros((self.rgb_height, self.rgb_width, 3), dtype=np.uint8)

        return image

    def _generate_depth_image(self) -> np.ndarray:
        """生成深度影像"""
        if self.mode == "random":
            # 隨機深度
            depth = np.random.randint(500, 2000, (self.depth_height, self.depth_width)).astype(np.float32)

        elif self.mode == "pattern":
            # 梯度深度
            depth = np.zeros((self.depth_height, self.depth_width), dtype=np.float32)
            for i in range(self.depth_height):
                depth[i, :] = 500 + (i / self.depth_height) * 1500

        elif self.mode == "objects":
            # 帶物體的深度場景
            depth = np.ones((self.depth_height, self.depth_width), dtype=np.float32) * 1500  # 背景

            # 計算縮放比例（RGB 到深度）
            scale_x = self.depth_width / self.rgb_width
            scale_y = self.depth_height / self.rgb_height

            # 金屬零件（近距離）
            x1, y1 = int(300 * scale_x), int(300 * scale_y)
            x2, y2 = int(600 * scale_x), int(500 * scale_y)
            depth[y1:y2, x1:x2] = 600

            # 塑膠零件（中距離）
            center_x, center_y = int(1000 * scale_x), int(400 * scale_y)
            radius = int(150 * scale_x)
            y, x = np.ogrid[: self.depth_height, : self.depth_width]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
            depth[mask] = 1000

            # 缺陷（凹陷）
            center_x, center_y = int(450 * scale_x), int(700 * scale_y)
            radius_x, radius_y = int(120 * scale_x), int(60 * scale_y)
            y, x = np.ogrid[: self.depth_height, : self.depth_width]
            mask = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
            depth[mask] = 1550  # 稍微凹陷

            # 加入深度噪點
            noise = np.random.randn(self.depth_height, self.depth_width) * 10
            depth = depth + noise

        else:
            depth = np.ones((self.depth_height, self.depth_width), dtype=np.float32) * 1000

        return depth

    def get_camera_info(self) -> Optional[CameraInfo]:
        """取得模擬相機資訊"""
        if not self.is_connected:
            return None

        return CameraInfo(
            model="Mock Gemini 2",
            serial_number="MOCK-12345678",
            firmware_version="1.0.0-mock",
            rgb_resolution=(self.rgb_width, self.rgb_height),
            depth_resolution=(self.depth_width, self.depth_height),
            fps=self.fps,
        )

    def get_rgb_intrinsics(self) -> Optional[Intrinsics]:
        """取得模擬 RGB 相機內參"""
        if not self.is_connected:
            return None

        # 模擬標準相機內參
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
        """取得模擬深度相機內參"""
        if not self.is_connected:
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

    def set_rgb_resolution(self, width: int, height: int) -> bool:
        """設定 RGB 解析度"""
        if self.is_streaming:
            return False

        self.rgb_width = width
        self.rgb_height = height
        return True

    def set_depth_resolution(self, width: int, height: int) -> bool:
        """設定深度解析度"""
        if self.is_streaming:
            return False

        self.depth_width = width
        self.depth_height = height
        return True

    def set_fps(self, fps: int) -> bool:
        """設定幀率"""
        self.fps = fps
        return True

    def enable_alignment(self, enabled: bool = True) -> bool:
        """啟用/停用 RGB-D 對齊"""
        self.alignment_enabled = enabled
        return True

    def set_depth_range(self, min_depth: int, max_depth: int) -> bool:
        """設定深度範圍"""
        self.min_depth = min_depth
        self.max_depth = max_depth
        return True


if __name__ == "__main__":
    # 測試模擬相機
    print("模擬相機驅動測試\n")

    # 測試三種模式
    modes = ["random", "pattern", "objects"]

    for mode in modes:
        print(f"測試模式: {mode}")

        camera = MockCamera(mode=mode)

        # 連接相機
        camera.connect()
        print(f"  ✓ 相機已連接")

        # 取得相機資訊
        info = camera.get_camera_info()
        print(f"  ✓ 相機型號: {info.model}")

        # 開始串流
        camera.start_streaming()
        print(f"  ✓ 開始串流")

        # 取得一幀
        frame = camera.get_frame()
        print(f"  ✓ 取得幀 #{frame.frame_number}")
        print(f"    RGB 大小: {frame.rgb.shape}")
        print(f"    深度大小: {frame.depth.shape}")
        print(f"    深度範圍: {frame.depth.min():.1f} - {frame.depth.max():.1f} mm")

        # 停止與中斷
        camera.stop_streaming()
        camera.disconnect()
        print(f"  ✓ 已停止並中斷\n")

    print("✓ 模擬相機測試完成")
