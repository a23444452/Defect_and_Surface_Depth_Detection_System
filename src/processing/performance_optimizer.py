"""
效能優化模組
提供各種效能優化策略以達到 30 FPS 目標
"""

import time
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import cv2


@dataclass
class PerformanceMetrics:
    """效能指標"""

    frame_time: float  # 幀處理時間 (秒)
    fps: float  # 每秒幀數
    bottleneck: str  # 瓶頸模組
    breakdown: Dict[str, float]  # 各模組耗時


class PerformanceOptimizer:
    """
    效能優化器
    提供快取、批次處理、多執行緒等優化策略
    """

    def __init__(self):
        """初始化優化器"""
        # 快取
        self._depth_filter_cache = None
        self._last_depth = None

        # 統計
        self.timings = {
            "capture": [],
            "process": [],
            "pointcloud": [],
            "total": [],
        }

    # ==================== 深度影像優化 ====================

    def fast_bilateral_filter(
        self, depth: np.ndarray, d: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0
    ) -> np.ndarray:
        """
        快速雙邊濾波 (使用較小的核心)

        Args:
            depth: 深度影像
            d: 核心大小 (預設 5, 原本 9)
            sigma_color: 色彩空間標準差
            sigma_space: 座標空間標準差

        Returns:
            濾波後的深度
        """
        # 使用 cv2.bilateralFilter 的快速版本
        return cv2.bilateralFilter(depth.astype(np.float32), d, sigma_color, sigma_space)

    def fast_hole_filling(self, depth: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        快速孔洞填補 (使用較小的核心)

        Args:
            depth: 深度影像
            kernel_size: 核心大小 (預設 3, 原本 5)

        Returns:
            填補後的深度
        """
        mask = (depth == 0).astype(np.uint8)

        if np.sum(mask) == 0:
            return depth

        # 使用 cv2.inpaint 快速版本
        filled = cv2.inpaint(depth.astype(np.float32), mask, kernel_size, cv2.INPAINT_NS)

        return filled

    def temporal_filter(self, current_depth: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """
        時域濾波 (與前一幀融合)

        Args:
            current_depth: 當前深度影像
            alpha: 當前幀權重 (0.7 = 70% 當前, 30% 前一幀)

        Returns:
            濾波後的深度
        """
        if self._last_depth is None:
            self._last_depth = current_depth.copy()
            return current_depth

        # 只在有效像素上融合
        valid_mask = (current_depth > 0) & (self._last_depth > 0)

        filtered = current_depth.copy()
        filtered[valid_mask] = (
            alpha * current_depth[valid_mask] + (1 - alpha) * self._last_depth[valid_mask]
        )

        # 更新快取
        self._last_depth = filtered.copy()

        return filtered

    # ==================== 點雲優化 ====================

    def fast_pointcloud_generation(
        self,
        depth: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        subsample: int = 4,
    ) -> np.ndarray:
        """
        快速點雲生成 (使用向量化操作)

        Args:
            depth: 深度影像 (H, W)
            fx, fy, cx, cy: 相機內參
            subsample: 降採樣係數 (預設 4, 原本 2)

        Returns:
            3D 點雲 (N, 3)
        """
        h, w = depth.shape

        # 向量化生成像素網格 (降採樣)
        v_coords, u_coords = np.mgrid[0:h:subsample, 0:w:subsample]

        # 展平
        u = u_coords.ravel()
        v = v_coords.ravel()
        d = depth[::subsample, ::subsample].ravel()

        # 過濾無效深度 (向量化)
        valid_mask = d > 0
        u = u[valid_mask]
        v = v[valid_mask]
        d = d[valid_mask]

        # 向量化 3D 轉換
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        z = d

        # 使用 column_stack 一次性建立
        points = np.column_stack([x, y, z])

        return points

    def adaptive_subsample(self, depth: np.ndarray, target_points: int = 50000) -> int:
        """
        自適應降採樣係數

        Args:
            depth: 深度影像
            target_points: 目標點數

        Returns:
            建議的降採樣係數
        """
        valid_pixels = np.sum(depth > 0)

        if valid_pixels == 0:
            return 1

        # 計算需要的降採樣係數
        subsample = int(np.sqrt(valid_pixels / target_points))

        return max(1, min(subsample, 8))  # 限制在 1-8 之間

    # ==================== 批次處理 ====================

    def batch_process_frames(
        self, frames: list, processor_func, batch_size: int = 4
    ) -> list:
        """
        批次處理多個幀

        Args:
            frames: 幀列表
            processor_func: 處理函數
            batch_size: 批次大小

        Returns:
            處理結果列表
        """
        results = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]

            # 批次處理
            batch_results = [processor_func(frame) for frame in batch]
            results.extend(batch_results)

        return results

    # ==================== 效能監控 ====================

    def record_timing(self, stage: str, elapsed_time: float):
        """
        記錄模組耗時

        Args:
            stage: 階段名稱
            elapsed_time: 耗時 (秒)
        """
        if stage not in self.timings:
            self.timings[stage] = []

        self.timings[stage].append(elapsed_time)

        # 只保留最近 100 個記錄
        if len(self.timings[stage]) > 100:
            self.timings[stage] = self.timings[stage][-100:]

    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        取得效能指標

        Returns:
            PerformanceMetrics
        """
        if not self.timings["total"]:
            return PerformanceMetrics(
                frame_time=0.0, fps=0.0, bottleneck="N/A", breakdown={}
            )

        # 計算平均值
        avg_times = {
            stage: np.mean(times) if times else 0.0 for stage, times in self.timings.items()
        }

        total_time = avg_times.get("total", 0.0)
        fps = 1.0 / total_time if total_time > 0 else 0.0

        # 找出瓶頸
        bottleneck = max(
            (k for k in avg_times if k != "total"),
            key=lambda k: avg_times[k],
            default="N/A",
        )

        return PerformanceMetrics(
            frame_time=total_time, fps=fps, bottleneck=bottleneck, breakdown=avg_times
        )

    def print_performance_report(self):
        """列印效能報告"""
        metrics = self.get_performance_metrics()

        print("\n" + "=" * 70)
        print("  效能分析報告")
        print("=" * 70)

        print(f"\n總體效能:")
        print(f"  平均幀時間: {metrics.frame_time*1000:.1f} ms")
        print(f"  平均 FPS: {metrics.fps:.1f}")

        print(f"\n各模組耗時:")
        for stage, time_ms in sorted(
            metrics.breakdown.items(), key=lambda x: x[1], reverse=True
        ):
            if stage != "total" and time_ms > 0:
                percentage = (time_ms / metrics.frame_time * 100) if metrics.frame_time > 0 else 0
                print(f"  {stage:15s}: {time_ms*1000:6.1f} ms ({percentage:5.1f}%)")

        print(f"\n效能瓶頸: {metrics.bottleneck}")

        # 建議
        print(f"\n優化建議:")
        if metrics.breakdown.get("capture", 0) > 0.03:  # > 30ms
            print("  • 考慮關閉相機的幀率模擬")
        if metrics.breakdown.get("process", 0) > 0.015:  # > 15ms
            print("  • 減少深度濾波強度或使用較小核心")
        if metrics.breakdown.get("pointcloud", 0) > 0.005:  # > 5ms
            print("  • 增加點雲降採樣係數")

        print("=" * 70)

    def reset_statistics(self):
        """重置統計資料"""
        for stage in self.timings:
            self.timings[stage] = []

        self._last_depth = None


class Timer:
    """
    計時器上下文管理器

    用法:
        with Timer() as t:
            # 執行操作
            pass
        print(f"耗時: {t.elapsed:.3f}s")
    """

    def __init__(self):
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time


# ==================== 工具函數 ====================


def benchmark_function(func, *args, iterations: int = 100, warmup: int = 10, **kwargs):
    """
    函數效能基準測試

    Args:
        func: 要測試的函數
        *args: 函數參數
        iterations: 測試迭代次數
        warmup: 預熱次數
        **kwargs: 函數關鍵字參數

    Returns:
        平均耗時 (秒)
    """
    # 預熱
    for _ in range(warmup):
        func(*args, **kwargs)

    # 測試
    times = []
    for _ in range(iterations):
        start = time.time()
        func(*args, **kwargs)
        times.append(time.time() - start)

    return np.mean(times), np.std(times)


if __name__ == "__main__":
    # 測試效能優化器
    print("效能優化器測試\n")

    optimizer = PerformanceOptimizer()

    # 測試深度濾波優化
    print("測試快速雙邊濾波...")
    depth = np.random.randint(800, 1500, (800, 1280)).astype(np.float32)

    # 標準版本
    avg_time_std, _ = benchmark_function(
        cv2.bilateralFilter, depth, 9, 75.0, 75.0, iterations=50
    )
    print(f"  標準雙邊濾波 (d=9): {avg_time_std*1000:.1f} ms")

    # 快速版本
    avg_time_fast, _ = benchmark_function(
        optimizer.fast_bilateral_filter, depth, iterations=50
    )
    print(f"  快速雙邊濾波 (d=5): {avg_time_fast*1000:.1f} ms")
    print(f"  加速: {avg_time_std/avg_time_fast:.2f}x")

    # 測試點雲生成優化
    print(f"\n測試快速點雲生成...")

    # 標準降採樣
    avg_time_std, _ = benchmark_function(
        optimizer.fast_pointcloud_generation,
        depth,
        720.91,
        720.91,
        640,
        400,
        subsample=2,
        iterations=50,
    )
    print(f"  降採樣 2x: {avg_time_std*1000:.1f} ms")

    # 快速降採樣
    avg_time_fast, _ = benchmark_function(
        optimizer.fast_pointcloud_generation,
        depth,
        720.91,
        720.91,
        640,
        400,
        subsample=4,
        iterations=50,
    )
    print(f"  降採樣 4x: {avg_time_fast*1000:.1f} ms")
    print(f"  加速: {avg_time_std/avg_time_fast:.2f}x")

    # 測試 Timer
    print(f"\n測試 Timer...")
    with Timer() as t:
        time.sleep(0.1)
    print(f"  測量耗時: {t.elapsed*1000:.1f} ms (預期 100ms)")

    print(f"\n✓ 效能優化器測試完成")
