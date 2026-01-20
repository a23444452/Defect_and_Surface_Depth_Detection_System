#!/usr/bin/env python3
"""
效能優化模組完整測試
測試所有優化功能與工具
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
from src.hardware import FastMockCamera, MockCamera
from src.processing import (
    PerformanceOptimizer,
    PerformanceMetrics,
    Timer,
    PointCloudGenerator,
)
from src.utils import setup_logger


def test_timer():
    """測試 Timer 計時器"""
    print("\n" + "=" * 70)
    print("  測試 1: Timer 計時器")
    print("=" * 70)

    # 測試基本計時
    print("\n測試基本計時:")
    with Timer() as t:
        time.sleep(0.1)
    print(f"  ✓ 計時: {t.elapsed*1000:.1f} ms (預期 ~100ms)")

    # 測試多次計時
    print("\n測試多次計時:")
    timings = []
    for i in range(5):
        with Timer() as t:
            time.sleep(0.01)
        timings.append(t.elapsed)
        print(f"  第 {i+1} 次: {t.elapsed*1000:.2f} ms")

    print(f"  平均: {np.mean(timings)*1000:.2f} ms")
    print(f"  標準差: {np.std(timings)*1000:.2f} ms")


def test_fast_bilateral_filter():
    """測試快速雙邊濾波"""
    print("\n" + "=" * 70)
    print("  測試 2: 快速雙邊濾波")
    print("=" * 70)

    optimizer = PerformanceOptimizer()

    # 建立測試深度影像
    depth = np.random.randint(800, 1500, (800, 1280)).astype(np.float32)
    # 加入噪點
    noise_mask = np.random.rand(800, 1280) < 0.1
    depth[noise_mask] += np.random.randint(-200, 200, np.sum(noise_mask))

    print(f"\n深度影像:")
    print(f"  尺寸: {depth.shape}")
    print(f"  範圍: {depth.min():.1f} - {depth.max():.1f} mm")

    # 測試標準濾波
    print(f"\n測試標準雙邊濾波 (d=9):")
    with Timer() as t:
        import cv2
        filtered_std = cv2.bilateralFilter(depth, 9, 75.0, 75.0)
    print(f"  ✓ 耗時: {t.elapsed*1000:.1f} ms")

    # 測試快速濾波
    print(f"\n測試快速雙邊濾波 (d=5):")
    with Timer() as t:
        filtered_fast = optimizer.fast_bilateral_filter(depth, d=5)
    print(f"  ✓ 耗時: {t.elapsed*1000:.1f} ms")

    # 比較
    speedup = (t.elapsed / t.elapsed) if t.elapsed > 0 else 1.0
    print(f"\n效能對比:")
    print(f"  加速比: 計算中...")

    # 品質比較
    diff = np.abs(filtered_std - filtered_fast)
    print(f"\n品質對比:")
    print(f"  平均差異: {diff.mean():.2f} mm")
    print(f"  最大差異: {diff.max():.2f} mm")


def test_temporal_filter():
    """測試時域濾波"""
    print("\n" + "=" * 70)
    print("  測試 3: 時域濾波")
    print("=" * 70)

    optimizer = PerformanceOptimizer()

    # 模擬連續幀
    print("\n模擬 5 個連續幀:")
    for i in range(5):
        # 生成深度影像 (帶噪點)
        base_depth = 1000.0 + i * 10  # 略微變化
        depth = np.full((800, 1280), base_depth, dtype=np.float32)
        noise = np.random.normal(0, 20, (800, 1280)).astype(np.float32)
        depth += noise

        # 應用時域濾波
        filtered = optimizer.temporal_filter(depth, alpha=0.7)

        # 計算噪聲
        original_noise = np.std(depth - base_depth)
        filtered_noise = np.std(filtered - base_depth)

        print(f"  幀 {i+1}:")
        print(f"    原始噪聲: {original_noise:.2f} mm")
        print(f"    濾波後: {filtered_noise:.2f} mm")
        print(f"    降噪: {(1 - filtered_noise/original_noise)*100:.1f}%")


def test_fast_pointcloud_generation():
    """測試快速點雲生成"""
    print("\n" + "=" * 70)
    print("  測試 4: 快速點雲生成")
    print("=" * 70)

    optimizer = PerformanceOptimizer()

    # 建立測試深度影像
    depth = np.random.randint(800, 1500, (800, 1280)).astype(np.float32)

    print(f"\n深度影像: {depth.shape}")

    # 測試不同降採樣係數
    for subsample in [1, 2, 4, 8]:
        with Timer() as t:
            points = optimizer.fast_pointcloud_generation(
                depth, fx=720.91, fy=720.91, cx=640, cy=400, subsample=subsample
            )

        print(f"\n降採樣 {subsample}x:")
        print(f"  點數量: {len(points):,}")
        print(f"  生成時間: {t.elapsed*1000:.2f} ms")
        print(f"  速率: {len(points)/t.elapsed/1000:.1f}K 點/秒")


def test_adaptive_subsample():
    """測試自適應降採樣"""
    print("\n" + "=" * 70)
    print("  測試 5: 自適應降採樣")
    print("=" * 70)

    optimizer = PerformanceOptimizer()

    # 測試不同的深度影像
    test_cases = [
        ("高密度", np.ones((800, 1280), dtype=np.float32) * 1000),  # 全部有效
        ("中密度", np.random.choice([1000.0, 0.0], size=(800, 1280), p=[0.5, 0.5])),
        ("低密度", np.random.choice([1000.0, 0.0], size=(800, 1280), p=[0.2, 0.8])),
    ]

    print("\n測試不同深度影像密度:")
    for name, depth in test_cases:
        valid_pixels = np.sum(depth > 0)
        subsample = optimizer.adaptive_subsample(depth, target_points=50000)

        print(f"\n{name}:")
        print(f"  有效像素: {valid_pixels:,}")
        print(f"  建議降採樣: {subsample}x")
        print(f"  預期點數: ~{valid_pixels/(subsample**2):,.0f}")


def test_performance_metrics():
    """測試效能指標"""
    print("\n" + "=" * 70)
    print("  測試 6: 效能指標與監控")
    print("=" * 70)

    optimizer = PerformanceOptimizer()

    # 模擬處理流程並記錄時間
    print("\n模擬 10 次處理流程:")
    for i in range(10):
        # 模擬各模組耗時
        optimizer.record_timing("capture", np.random.uniform(0.0005, 0.001))
        optimizer.record_timing("process", np.random.uniform(0.002, 0.004))
        optimizer.record_timing("pointcloud", np.random.uniform(0.0005, 0.001))
        optimizer.record_timing(
            "total",
            optimizer.timings["capture"][-1]
            + optimizer.timings["process"][-1]
            + optimizer.timings["pointcloud"][-1],
        )

    # 取得效能指標
    metrics = optimizer.get_performance_metrics()

    print(f"\n效能指標:")
    print(f"  平均幀時間: {metrics.frame_time*1000:.2f} ms")
    print(f"  平均 FPS: {metrics.fps:.1f}")
    print(f"  效能瓶頸: {metrics.bottleneck}")

    print(f"\n各模組耗時:")
    for stage, time_val in sorted(
        metrics.breakdown.items(), key=lambda x: x[1], reverse=True
    ):
        if stage != "total" and time_val > 0:
            print(f"  {stage:12s}: {time_val*1000:6.2f} ms")

    # 列印報告
    print()
    optimizer.print_performance_report()


def test_camera_comparison():
    """測試相機效能對比"""
    print("\n" + "=" * 70)
    print("  測試 7: MockCamera vs FastMockCamera")
    print("=" * 70)

    # MockCamera 測試
    print("\nMockCamera (帶延遲模擬):")
    camera = MockCamera(mode="objects")

    with camera:
        times = []
        for i in range(10):
            with Timer() as t:
                frame = camera.get_frame()
            times.append(t.elapsed)

        avg_time = np.mean(times)
        print(f"  平均擷取時間: {avg_time*1000:.2f} ms")
        print(f"  平均 FPS: {1.0/avg_time:.1f}")

    # FastMockCamera 測試
    print("\nFastMockCamera (無延遲):")
    camera = FastMockCamera(mode="objects")

    with camera:
        times = []
        for i in range(10):
            with Timer() as t:
                frame = camera.get_frame()
            times.append(t.elapsed)

        avg_time = np.mean(times)
        print(f"  平均擷取時間: {avg_time*1000:.2f} ms")
        print(f"  平均 FPS: {1.0/avg_time:.1f}")

    print(f"\n加速比: {np.mean(times) / avg_time:.1f}x")


def test_end_to_end_optimized():
    """測試端到端優化流程"""
    print("\n" + "=" * 70)
    print("  測試 8: 端到端優化流程")
    print("=" * 70)

    logger = setup_logger(name="TestE2E", log_dir="outputs/logs")
    optimizer = PerformanceOptimizer()
    camera = FastMockCamera(mode="objects")

    print("\n執行 20 次完整優化流程:")

    with camera:
        for i in range(20):
            t_total_start = time.time()

            # 1. 擷取
            with Timer() as t:
                frame = camera.get_frame()
            optimizer.record_timing("capture", t.elapsed)

            # 2. 優化處理
            with Timer() as t:
                depth_filtered = optimizer.fast_bilateral_filter(frame.depth, d=5)
                depth_filtered = optimizer.temporal_filter(depth_filtered, alpha=0.7)
            optimizer.record_timing("process", t.elapsed)

            # 3. 快速點雲
            with Timer() as t:
                points = optimizer.fast_pointcloud_generation(
                    depth_filtered,
                    fx=720.91,
                    fy=720.91,
                    cx=640,
                    cy=400,
                    subsample=4,
                )
            optimizer.record_timing("pointcloud", t.elapsed)

            optimizer.record_timing("total", time.time() - t_total_start)

            if (i + 1) % 5 == 0:
                logger.info(f"完成 {i+1}/20 次")

    # 取得結果
    metrics = optimizer.get_performance_metrics()

    print(f"\n優化流程效能:")
    print(f"  平均幀時間: {metrics.frame_time*1000:.2f} ms")
    print(f"  平均 FPS: {metrics.fps:.1f}")
    print(f"  效能瓶頸: {metrics.bottleneck}")

    print(f"\n各模組耗時:")
    for stage in ["capture", "process", "pointcloud"]:
        time_val = metrics.breakdown.get(stage, 0)
        percentage = (
            (time_val / metrics.frame_time * 100) if metrics.frame_time > 0 else 0
        )
        print(f"  {stage:12s}: {time_val*1000:6.2f} ms ({percentage:5.1f}%)")


def main():
    """主函數"""
    print("\n" + "=" * 70)
    print("  效能優化模組完整測試")
    print("=" * 70)

    # 執行所有測試
    test_timer()
    test_fast_bilateral_filter()
    test_temporal_filter()
    test_fast_pointcloud_generation()
    test_adaptive_subsample()
    test_performance_metrics()
    test_camera_comparison()
    test_end_to_end_optimized()

    print("\n" + "=" * 70)
    print("  ✓ 所有測試完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
