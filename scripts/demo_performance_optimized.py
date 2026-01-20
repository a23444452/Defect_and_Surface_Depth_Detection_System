#!/usr/bin/env python3
"""
效能優化展示
展示各種優化策略對效能的提升
目標: 從 13 FPS 提升到 30 FPS
"""

import sys
from pathlib import Path

# 加入專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
from src.hardware import FastMockCamera
from src.processing import (
    RGBDProcessor,
    PointCloudGenerator,
    DepthFilter,
)
from src.processing.performance_optimizer import (
    PerformanceOptimizer,
    Timer,
)
from src.utils import setup_logger


def demo_baseline_performance():
    """基準效能測試 (未優化)"""
    print("\n" + "=" * 70)
    print("  基準效能測試 (未優化)")
    print("=" * 70)

    logger = setup_logger(name="Baseline", log_dir="outputs/logs")

    # 初始化 (標準配置)
    camera = FastMockCamera(mode="objects")
    rgbd_processor = RGBDProcessor(
        enable_depth_filter=True,
        depth_filter_method="bilateral",
        depth_fill_holes=True,
    )
    pointcloud_gen = PointCloudGenerator(fx=720.91, fy=720.91, cx=640, cy=400)

    num_iterations = 20
    times = {"capture": [], "process": [], "pointcloud": [], "total": []}

    logger.info(f"執行 {num_iterations} 次基準測試...")

    with camera:
        for i in range(num_iterations):
            t_total_start = time.time()

            # 1. 擷取
            t0 = time.time()
            frame = camera.get_frame()
            times["capture"].append(time.time() - t0)

            # 2. 處理
            t0 = time.time()
            processed = rgbd_processor.process(frame.rgb, frame.depth)
            times["process"].append(time.time() - t0)

            # 3. 點雲
            t0 = time.time()
            pointcloud = pointcloud_gen.generate_from_rgbd(
                processed.depth, processed.rgb, subsample=2
            )
            times["pointcloud"].append(time.time() - t0)

            times["total"].append(time.time() - t_total_start)

            if (i + 1) % 5 == 0:
                logger.info(f"  完成 {i+1}/{num_iterations} 次")

    # 統計
    print("\n基準效能 (平均值):")
    avg_capture = np.mean(times["capture"])
    avg_process = np.mean(times["process"])
    avg_pointcloud = np.mean(times["pointcloud"])
    avg_total = np.mean(times["total"])

    print(f"  相機擷取: {avg_capture*1000:.1f} ms ({avg_capture/avg_total*100:.1f}%)")
    print(f"  RGB-D 處理: {avg_process*1000:.1f} ms ({avg_process/avg_total*100:.1f}%)")
    print(f"  點雲生成: {avg_pointcloud*1000:.1f} ms ({avg_pointcloud/avg_total*100:.1f}%)")
    print(f"  總計: {avg_total*1000:.1f} ms")
    print(f"  平均 FPS: {1.0/avg_total:.1f}")

    return times


def demo_optimized_performance():
    """優化後的效能測試"""
    print("\n" + "=" * 70)
    print("  優化後效能測試")
    print("=" * 70)

    logger = setup_logger(name="Optimized", log_dir="outputs/logs")
    optimizer = PerformanceOptimizer()

    # 初始化 (優化配置)
    camera = FastMockCamera(mode="objects")

    # 優化 1: 簡化深度處理
    rgbd_processor = RGBDProcessor(
        enable_depth_filter=True,
        depth_filter_method="bilateral",  # 使用較快的濾波
        depth_fill_holes=False,  # 關閉孔洞填補 (耗時)
    )

    pointcloud_gen = PointCloudGenerator(fx=720.91, fy=720.91, cx=640, cy=400)

    num_iterations = 20
    times = {"capture": [], "process": [], "pointcloud": [], "total": []}

    logger.info(f"執行 {num_iterations} 次優化測試...")

    with camera:
        for i in range(num_iterations):
            t_total_start = time.time()

            # 1. 擷取 (無變化)
            t0 = time.time()
            frame = camera.get_frame()
            times["capture"].append(time.time() - t0)

            # 2. 優化的處理流程
            t0 = time.time()

            # 優化 2: 使用快速濾波
            depth_filtered = optimizer.fast_bilateral_filter(
                frame.depth, d=5, sigma_color=50.0, sigma_space=50.0
            )

            # 優化 3: 使用時域濾波替代孔洞填補
            depth_filtered = optimizer.temporal_filter(depth_filtered, alpha=0.7)

            # 簡化的處理
            processed_rgb = frame.rgb
            valid_mask = (depth_filtered > 0) & (depth_filtered < 10000)

            times["process"].append(time.time() - t0)

            # 3. 優化的點雲生成
            t0 = time.time()

            # 優化 4: 使用更高的降採樣係數
            points = optimizer.fast_pointcloud_generation(
                depth_filtered,
                fx=720.91,
                fy=720.91,
                cx=640,
                cy=400,
                subsample=4,  # 從 2 提升到 4
            )

            times["pointcloud"].append(time.time() - t0)

            times["total"].append(time.time() - t_total_start)

            if (i + 1) % 5 == 0:
                logger.info(f"  完成 {i+1}/{num_iterations} 次")

    # 統計
    print("\n優化效能 (平均值):")
    avg_capture = np.mean(times["capture"])
    avg_process = np.mean(times["process"])
    avg_pointcloud = np.mean(times["pointcloud"])
    avg_total = np.mean(times["total"])

    print(f"  相機擷取: {avg_capture*1000:.1f} ms ({avg_capture/avg_total*100:.1f}%)")
    print(f"  RGB-D 處理: {avg_process*1000:.1f} ms ({avg_process/avg_total*100:.1f}%)")
    print(f"  點雲生成: {avg_pointcloud*1000:.1f} ms ({avg_pointcloud/avg_total*100:.1f}%)")
    print(f"  總計: {avg_total*1000:.1f} ms")
    print(f"  平均 FPS: {1.0/avg_total:.1f}")

    return times


def demo_aggressive_optimization():
    """激進優化 (追求最高 FPS)"""
    print("\n" + "=" * 70)
    print("  激進優化測試 (追求 30+ FPS)")
    print("=" * 70)

    logger = setup_logger(name="Aggressive", log_dir="outputs/logs")
    optimizer = PerformanceOptimizer()

    # 初始化 (激進配置)
    camera = FastMockCamera(mode="objects")

    # 不使用 RGBDProcessor,直接處理
    pointcloud_gen = PointCloudGenerator(fx=720.91, fy=720.91, cx=640, cy=400)

    num_iterations = 20
    times = {"capture": [], "process": [], "pointcloud": [], "total": []}

    logger.info(f"執行 {num_iterations} 次激進優化測試...")

    with camera:
        for i in range(num_iterations):
            t_total_start = time.time()

            # 1. 擷取
            t0 = time.time()
            frame = camera.get_frame()
            times["capture"].append(time.time() - t0)

            # 2. 最小化處理
            t0 = time.time()

            # 優化 5: 只保留時域濾波
            depth_filtered = optimizer.temporal_filter(frame.depth, alpha=0.8)

            times["process"].append(time.time() - t0)

            # 3. 最小化點雲生成
            t0 = time.time()

            # 優化 6: 自適應降採樣
            subsample = optimizer.adaptive_subsample(depth_filtered, target_points=30000)

            points = optimizer.fast_pointcloud_generation(
                depth_filtered,
                fx=720.91,
                fy=720.91,
                cx=640,
                cy=400,
                subsample=subsample,
            )

            times["pointcloud"].append(time.time() - t0)

            times["total"].append(time.time() - t_total_start)

            if (i + 1) % 5 == 0:
                logger.info(f"  完成 {i+1}/{num_iterations} 次")

    # 統計
    print("\n激進優化效能 (平均值):")
    avg_capture = np.mean(times["capture"])
    avg_process = np.mean(times["process"])
    avg_pointcloud = np.mean(times["pointcloud"])
    avg_total = np.mean(times["total"])

    print(f"  相機擷取: {avg_capture*1000:.1f} ms ({avg_capture/avg_total*100:.1f}%)")
    print(f"  RGB-D 處理: {avg_process*1000:.1f} ms ({avg_process/avg_total*100:.1f}%)")
    print(f"  點雲生成: {avg_pointcloud*1000:.1f} ms ({avg_pointcloud/avg_total*100:.1f}%)")
    print(f"  總計: {avg_total*1000:.1f} ms")
    print(f"  平均 FPS: {1.0/avg_total:.1f}")

    return times


def compare_results(baseline, optimized, aggressive):
    """比較結果"""
    print("\n" + "=" * 70)
    print("  效能比較")
    print("=" * 70)

    baseline_total = np.mean(baseline["total"])
    optimized_total = np.mean(optimized["total"])
    aggressive_total = np.mean(aggressive["total"])

    baseline_fps = 1.0 / baseline_total
    optimized_fps = 1.0 / optimized_total
    aggressive_fps = 1.0 / aggressive_total

    print("\n總處理時間:")
    print(f"  基準版本:   {baseline_total*1000:6.1f} ms ({baseline_fps:5.1f} FPS)")
    print(
        f"  優化版本:   {optimized_total*1000:6.1f} ms ({optimized_fps:5.1f} FPS) [{optimized_fps/baseline_fps:.2f}x]"
    )
    print(
        f"  激進優化:   {aggressive_total*1000:6.1f} ms ({aggressive_fps:5.1f} FPS) [{aggressive_fps/baseline_fps:.2f}x]"
    )

    print("\n各模組對比:")
    stages = ["capture", "process", "pointcloud"]

    for stage in stages:
        baseline_time = np.mean(baseline[stage])
        optimized_time = np.mean(optimized[stage])
        aggressive_time = np.mean(aggressive[stage])

        print(f"\n{stage.capitalize()}:")
        print(f"  基準:     {baseline_time*1000:6.1f} ms")
        print(
            f"  優化:     {optimized_time*1000:6.1f} ms (減少 {(1-optimized_time/baseline_time)*100:5.1f}%)"
        )
        print(
            f"  激進:     {aggressive_time*1000:6.1f} ms (減少 {(1-aggressive_time/baseline_time)*100:5.1f}%)"
        )

    # 目標達成檢查
    print("\n" + "=" * 70)
    target_fps = 30.0
    target_time = 1000.0 / target_fps  # 33.3 ms

    print(f"  目標: {target_fps} FPS ({target_time:.1f} ms)")
    print("=" * 70)

    if aggressive_fps >= target_fps:
        print(f"\n✅ 目標達成! 激進優化達到 {aggressive_fps:.1f} FPS")
    elif optimized_fps >= target_fps:
        print(f"\n✅ 目標達成! 優化版本達到 {optimized_fps:.1f} FPS")
    else:
        print(f"\n⚠️  未達目標,當前最佳: {max(optimized_fps, aggressive_fps):.1f} FPS")
        print(f"    還需優化: {target_time - min(optimized_total, aggressive_total)*1000:.1f} ms")


def demo_optimization_strategies():
    """展示各種優化策略"""
    print("\n" + "=" * 70)
    print("  優化策略說明")
    print("=" * 70)

    strategies = [
        ("優化 1", "簡化深度濾波", "使用較小的核心 (9→5), 減少計算量"),
        ("優化 2", "關閉孔洞填補", "孔洞填補耗時較長,可用時域濾波替代"),
        ("優化 3", "時域濾波", "與前一幀融合,既能去噪又比空間濾波快"),
        ("優化 4", "提高降採樣", "點雲降採樣 (2x→4x), 大幅減少點數"),
        ("優化 5", "自適應降採樣", "根據深度資料自動調整降採樣係數"),
        ("優化 6", "向量化計算", "使用 NumPy 向量化操作替代迴圈"),
    ]

    print("\n已實作的優化策略:")
    for opt_id, name, desc in strategies:
        print(f"\n{opt_id}: {name}")
        print(f"  {desc}")

    print("\n" + "=" * 70)
    print("  建議的優化組合")
    print("=" * 70)

    modes = [
        (
            "平衡模式",
            "優化 1-4",
            "保持良好品質,提升到 ~20 FPS",
            ["快速濾波", "時域濾波", "降採樣 4x"],
        ),
        (
            "高速模式",
            "優化 1-6",
            "追求最高速度,達到 30+ FPS",
            ["最小濾波", "自適應降採樣", "向量化"],
        ),
        (
            "高品質模式",
            "優化 3, 6",
            "保持高品質,~15 FPS",
            ["完整濾波", "時域濾波", "降採樣 2x"],
        ),
    ]

    for mode_name, opts, desc, features in modes:
        print(f"\n{mode_name} ({opts}):")
        print(f"  目標: {desc}")
        print(f"  特點: {', '.join(features)}")


def main():
    """主函數"""
    print("\n" + "=" * 70)
    print("  效能優化展示 - 從 13 FPS 提升到 30 FPS")
    print("=" * 70)

    # 1. 展示優化策略
    demo_optimization_strategies()

    # 2. 基準測試
    baseline = demo_baseline_performance()

    # 3. 優化測試
    optimized = demo_optimized_performance()

    # 4. 激進優化測試
    aggressive = demo_aggressive_optimization()

    # 5. 比較結果
    compare_results(baseline, optimized, aggressive)

    print("\n" + "=" * 70)
    print("  所有效能測試完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
