"""
RGB-D 處理器
整合 RGB 與深度影像的處理流程
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import time

try:
    from .depth_filter import DepthFilter
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.processing.depth_filter import DepthFilter


@dataclass
class ProcessedFrame:
    """處理後的 RGB-D 幀"""

    rgb: np.ndarray  # 處理後 RGB (H, W, 3)
    depth: np.ndarray  # 處理後深度 (H, W)
    rgb_original: np.ndarray  # 原始 RGB
    depth_original: np.ndarray  # 原始深度
    valid_mask: np.ndarray  # 有效像素遮罩 (H, W) bool
    processing_time: float  # 處理時間 (秒)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid_ratio(self) -> float:
        """有效像素比例"""
        return np.sum(self.valid_mask) / self.valid_mask.size


class RGBDProcessor:
    """
    RGB-D 處理器
    整合 RGB 與深度影像的處理流程
    """

    def __init__(
        self,
        enable_depth_filter: bool = True,
        enable_rgb_enhance: bool = False,
        depth_filter_method: str = "bilateral",
        depth_fill_holes: bool = True,
        depth_range: tuple = (300, 3000),
    ):
        """
        初始化 RGB-D 處理器

        Args:
            enable_depth_filter: 是否啟用深度濾波
            enable_rgb_enhance: 是否啟用 RGB 增強
            depth_filter_method: 深度濾波方法
            depth_fill_holes: 是否填補深度洞
            depth_range: 深度範圍 (min, max) mm
        """
        self.enable_depth_filter = enable_depth_filter
        self.enable_rgb_enhance = enable_rgb_enhance
        self.depth_filter_method = depth_filter_method
        self.depth_fill_holes = depth_fill_holes
        self.depth_range = depth_range

        # 建立深度濾波器
        self.depth_filter = DepthFilter()

    def process(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
    ) -> ProcessedFrame:
        """
        處理 RGB-D 幀

        Args:
            rgb: RGB 影像 (H, W, 3) uint8
            depth: 深度影像 (H, W) float32

        Returns:
            ProcessedFrame
        """
        start_time = time.time()

        # 保存原始資料
        rgb_original = rgb.copy()
        depth_original = depth.copy()

        # 處理深度
        processed_depth = self._process_depth(depth)

        # 處理 RGB
        processed_rgb = self._process_rgb(rgb)

        # 建立有效遮罩
        valid_mask = (processed_depth > 0) & (processed_depth < 10000)

        # 計算處理時間
        processing_time = time.time() - start_time

        return ProcessedFrame(
            rgb=processed_rgb,
            depth=processed_depth,
            rgb_original=rgb_original,
            depth_original=depth_original,
            valid_mask=valid_mask,
            processing_time=processing_time,
            metadata={
                "depth_filter_enabled": self.enable_depth_filter,
                "rgb_enhance_enabled": self.enable_rgb_enhance,
                "depth_filter_method": self.depth_filter_method,
                "depth_range": self.depth_range,
            },
        )

    def _process_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        處理深度影像

        Args:
            depth: 原始深度

        Returns:
            處理後的深度
        """
        if not self.enable_depth_filter:
            return depth.copy()

        # 使用深度濾波器
        processed = self.depth_filter.process_depth(
            depth,
            remove_outliers=True,
            clamp_range=True,
            filter_method=self.depth_filter_method,
            fill_holes=self.depth_fill_holes,
            min_depth=self.depth_range[0],
            max_depth=self.depth_range[1],
        )

        return processed

    def _process_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """
        處理 RGB 影像

        Args:
            rgb: 原始 RGB

        Returns:
            處理後的 RGB
        """
        processed = rgb.copy()

        if not self.enable_rgb_enhance:
            return processed

        # RGB 增強
        import cv2

        # 1. 亮度與對比度調整 (CLAHE)
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        processed = cv2.merge([l, a, b])
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)

        return processed

    def process_batch(
        self,
        rgb_list: list,
        depth_list: list,
    ) -> list:
        """
        批次處理多個 RGB-D 幀

        Args:
            rgb_list: RGB 影像列表
            depth_list: 深度影像列表

        Returns:
            ProcessedFrame 列表
        """
        if len(rgb_list) != len(depth_list):
            raise ValueError("RGB 與深度列表長度不符")

        results = []
        for rgb, depth in zip(rgb_list, depth_list):
            result = self.process(rgb, depth)
            results.append(result)

        return results

    def get_statistics(self, processed_frame: ProcessedFrame) -> Dict[str, Any]:
        """
        取得處理統計資訊

        Args:
            processed_frame: 處理後的幀

        Returns:
            統計資訊字典
        """
        depth = processed_frame.depth
        valid_mask = processed_frame.valid_mask

        valid_depth = depth[valid_mask]

        stats = {
            "processing_time": processed_frame.processing_time,
            "valid_ratio": processed_frame.valid_ratio,
            "total_pixels": depth.size,
            "valid_pixels": np.sum(valid_mask),
            "depth_min": float(valid_depth.min()) if len(valid_depth) > 0 else 0,
            "depth_max": float(valid_depth.max()) if len(valid_depth) > 0 else 0,
            "depth_mean": float(valid_depth.mean()) if len(valid_depth) > 0 else 0,
            "depth_std": float(valid_depth.std()) if len(valid_depth) > 0 else 0,
        }

        return stats


if __name__ == "__main__":
    # 測試 RGB-D 處理器
    print("RGB-D 處理器測試\n")

    # 建立測試資料
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.random.randint(800, 1500, (480, 640)).astype(np.float32)

    # 加入噪點與洞
    noise_mask = np.random.rand(480, 640) < 0.1
    depth[noise_mask] = np.random.randint(100, 2500, np.sum(noise_mask))
    hole_mask = np.random.rand(480, 640) < 0.05
    depth[hole_mask] = 0

    print(f"測試資料:")
    print(f"  RGB: {rgb.shape}")
    print(f"  深度: {depth.shape}")
    print(f"  有效深度: {np.sum(depth > 0)} / {depth.size}")

    # 建立處理器
    processor = RGBDProcessor(
        enable_depth_filter=True,
        enable_rgb_enhance=False,
        depth_filter_method="bilateral",
        depth_fill_holes=True,
    )

    print(f"\n處理器配置:")
    print(f"  深度濾波: {processor.enable_depth_filter}")
    print(f"  RGB 增強: {processor.enable_rgb_enhance}")
    print(f"  濾波方法: {processor.depth_filter_method}")

    # 處理
    print(f"\n執行處理...")
    result = processor.process(rgb, depth)

    print(f"\n處理結果:")
    print(f"  處理時間: {result.processing_time:.3f}s")
    print(f"  有效比例: {result.valid_ratio * 100:.1f}%")
    print(f"  有效像素: {np.sum(result.valid_mask)} / {result.valid_mask.size}")

    # 統計資訊
    stats = processor.get_statistics(result)
    print(f"\n深度統計:")
    print(f"  最小: {stats['depth_min']:.1f} mm")
    print(f"  最大: {stats['depth_max']:.1f} mm")
    print(f"  平均: {stats['depth_mean']:.1f} mm")
    print(f"  標準差: {stats['depth_std']:.1f} mm")

    print(f"\n✓ RGB-D 處理器測試完成")
