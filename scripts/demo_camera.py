#!/usr/bin/env python3
"""
相機模組示範腳本
展示如何使用硬體介面模組擷取 RGB-D 影像
"""

import sys
from pathlib import Path

# 加入專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
import cv2
from src.hardware import MockCamera, GEMINI2_AVAILABLE
from src.utils import setup_logger, get_visualizer


def demo_basic_capture():
    """基本擷取範例"""
    print("\n" + "=" * 70)
    print("  範例 1: 基本影像擷取")
    print("=" * 70)

    logger = setup_logger(name="CameraDemo", log_dir="outputs/logs")
    visualizer = get_visualizer()

    # 使用模擬相機
    logger.info("使用模擬相機 (objects 模式)")

    with MockCamera(mode="objects") as camera:
        logger.info("相機已連接並開始串流")

        # 取得一幀
        frame = camera.get_frame()
        logger.info(f"取得幀 #{frame.frame_number}")
        logger.info(f"  時間戳: {frame.timestamp:.3f}s")
        logger.info(f"  RGB 大小: {frame.rgb.shape}")
        logger.info(f"  深度大小: {frame.depth.shape}")

        # 視覺化並儲存
        depth_colored = visualizer.draw_depth_map(frame.depth)

        comparison = visualizer.create_comparison_view(
            rgb_image=frame.rgb, depth_image=depth_colored
        )

        save_path = "outputs/camera_demo_basic.png"
        visualizer.save_image(comparison, save_path)
        logger.success(f"結果已儲存: {save_path}")

    print(f"\n  ✓ 完成")
    print(f"  結果: {save_path}")


def demo_continuous_capture():
    """連續擷取範例"""
    print("\n" + "=" * 70)
    print("  範例 2: 連續影像擷取")
    print("=" * 70)

    logger = setup_logger(name="CameraDemo", log_dir="outputs/logs")
    visualizer = get_visualizer()

    num_frames = 10
    logger.info(f"連續擷取 {num_frames} 幀")

    with MockCamera(mode="objects") as camera:
        start_time = time.time()

        for i in range(num_frames):
            frame = camera.get_frame()

            if i % 3 == 0:  # 每 3 幀顯示一次
                logger.info(f"幀 #{frame.frame_number}: {frame.timestamp:.3f}s")

        elapsed = time.time() - start_time
        actual_fps = num_frames / elapsed

        logger.success(f"完成 {num_frames} 幀擷取")
        logger.info(f"  耗時: {elapsed:.3f}s")
        logger.info(f"  平均 FPS: {actual_fps:.2f}")

    print(f"\n  ✓ 完成")
    print(f"  擷取 {num_frames} 幀，平均 FPS: {actual_fps:.2f}")


def demo_depth_analysis():
    """深度分析範例"""
    print("\n" + "=" * 70)
    print("  範例 3: 深度資料分析")
    print("=" * 70)

    logger = setup_logger(name="CameraDemo", log_dir="outputs/logs")

    with MockCamera(mode="objects") as camera:
        frame = camera.get_frame()

        # 分析深度資料
        depth = frame.depth
        valid_mask = (depth > 0) & (depth < 10000)
        valid_depth = depth[valid_mask]

        logger.info("深度資料統計:")
        logger.info(f"  總像素數: {depth.size}")
        logger.info(f"  有效像素數: {len(valid_depth)} ({len(valid_depth)/depth.size*100:.1f}%)")
        logger.info(f"  最小深度: {valid_depth.min():.1f} mm")
        logger.info(f"  最大深度: {valid_depth.max():.1f} mm")
        logger.info(f"  平均深度: {valid_depth.mean():.1f} mm")
        logger.info(f"  深度標準差: {valid_depth.std():.1f} mm")

        # 深度直方圖
        hist, bins = np.histogram(valid_depth, bins=10)
        logger.info("\n  深度分布:")
        for i in range(len(hist)):
            logger.info(f"    {bins[i]:.0f}-{bins[i+1]:.0f} mm: {hist[i]} 像素")

    print(f"\n  ✓ 完成")


def demo_camera_info():
    """相機資訊範例"""
    print("\n" + "=" * 70)
    print("  範例 4: 相機資訊查詢")
    print("=" * 70)

    logger = setup_logger(name="CameraDemo", log_dir="outputs/logs")

    with MockCamera(mode="objects") as camera:
        # 取得相機資訊
        info = camera.get_camera_info()
        logger.info("相機資訊:")
        logger.info(f"  型號: {info.model}")
        logger.info(f"  序號: {info.serial_number}")
        logger.info(f"  韌體版本: {info.firmware_version}")
        logger.info(f"  RGB 解析度: {info.rgb_resolution}")
        logger.info(f"  深度解析度: {info.depth_resolution}")
        logger.info(f"  FPS: {info.fps}")

        # 取得內參
        rgb_intrinsics = camera.get_rgb_intrinsics()
        logger.info("\nRGB 相機內參:")
        logger.info(f"  fx: {rgb_intrinsics.fx:.2f}")
        logger.info(f"  fy: {rgb_intrinsics.fy:.2f}")
        logger.info(f"  cx: {rgb_intrinsics.cx:.2f}")
        logger.info(f"  cy: {rgb_intrinsics.cy:.2f}")

        depth_intrinsics = camera.get_depth_intrinsics()
        logger.info("\n深度相機內參:")
        logger.info(f"  fx: {depth_intrinsics.fx:.2f}")
        logger.info(f"  fy: {depth_intrinsics.fy:.2f}")
        logger.info(f"  cx: {depth_intrinsics.cx:.2f}")
        logger.info(f"  cy: {depth_intrinsics.cy:.2f}")

    print(f"\n  ✓ 完成")


def demo_multi_mode_comparison():
    """多模式比較範例"""
    print("\n" + "=" * 70)
    print("  範例 5: 多種模式比較")
    print("=" * 70)

    logger = setup_logger(name="CameraDemo", log_dir="outputs/logs")
    visualizer = get_visualizer()

    modes = ["random", "pattern", "objects"]
    results = []

    for mode in modes:
        logger.info(f"測試模式: {mode}")

        with MockCamera(mode=mode) as camera:
            frame = camera.get_frame()

            # 轉換深度圖
            depth_colored = visualizer.draw_depth_map(frame.depth)

            # 建立比較視圖
            comparison = visualizer.create_comparison_view(
                rgb_image=frame.rgb, depth_image=depth_colored
            )

            results.append(comparison)

    # 垂直堆疊三種模式
    import cv2

    final_result = np.vstack(results)
    save_path = "outputs/camera_demo_modes_comparison.png"
    visualizer.save_image(final_result, save_path)

    logger.success(f"多模式比較圖已儲存: {save_path}")

    print(f"\n  ✓ 完成")
    print(f"  結果: {save_path}")


def main():
    """主函數"""
    print("\n" + "=" * 70)
    print("  相機模組使用示範")
    print("=" * 70)

    print(f"\nOrbbecSDK 可用: {GEMINI2_AVAILABLE}")
    if not GEMINI2_AVAILABLE:
        print("  使用模擬相機進行示範")

    # 執行所有範例
    demo_basic_capture()
    demo_continuous_capture()
    demo_depth_analysis()
    demo_camera_info()
    demo_multi_mode_comparison()

    print("\n" + "=" * 70)
    print("  所有示範完成！")
    print("=" * 70)

    print("\n生成的檔案:")
    print("  - outputs/camera_demo_basic.png")
    print("  - outputs/camera_demo_modes_comparison.png")
    print("  - outputs/logs/CameraDemo.log")


if __name__ == "__main__":
    main()
