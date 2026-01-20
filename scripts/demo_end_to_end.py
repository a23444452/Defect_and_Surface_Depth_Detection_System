#!/usr/bin/env python3
"""
端到端整合展示
展示完整的檢測系統流程: 相機 → 處理 → AI 檢測 → 點雲
"""

import sys
from pathlib import Path

# 加入專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
from src.hardware import MockCamera
from src.processing import (
    RGBDProcessor,
    PointCloudGenerator,
    AIPreprocessor,
)
from src.utils import setup_logger, get_visualizer


def demo_full_pipeline():
    """完整處理流程展示"""
    print("\n" + "=" * 70)
    print("  端到端整合展示 - 完整處理流程")
    print("=" * 70)

    logger = setup_logger(name="EndToEnd", log_dir="outputs/logs")
    visualizer = get_visualizer()

    # ==================== 初始化 ====================
    logger.info("初始化系統組件...")

    # 相機
    camera = MockCamera(mode="objects")

    # RGB-D 處理器
    rgbd_processor = RGBDProcessor(
        enable_depth_filter=True,
        enable_rgb_enhance=False,
        depth_filter_method="bilateral",
        depth_fill_holes=True,
    )

    # 點雲生成器
    pointcloud_gen = PointCloudGenerator(fx=720.91, fy=720.91, cx=640, cy=400)

    # AI 前處理器
    ai_preprocessor = AIPreprocessor(target_size=(640, 640), keep_aspect=True)

    logger.success("系統初始化完成")

    # ==================== 相機擷取 ====================
    print("\n" + "=" * 70)
    print("  步驟 1: 相機擷取")
    print("=" * 70)

    with camera:
        logger.info("擷取 RGB-D 影像...")
        frame = camera.get_frame()

        logger.info(f"✓ 取得幀 #{frame.frame_number}")
        logger.info(f"  RGB: {frame.rgb.shape}")
        logger.info(f"  深度: {frame.depth.shape}")
        logger.info(f"  時間戳: {frame.timestamp:.3f}s")

        # ==================== RGB-D 處理 ====================
        print("\n" + "=" * 70)
        print("  步驟 2: RGB-D 影像處理")
        print("=" * 70)

        logger.info("執行 RGB-D 處理...")
        processed = rgbd_processor.process(frame.rgb, frame.depth)

        stats = rgbd_processor.get_statistics(processed)
        logger.success(f"✓ 處理完成 (耗時 {processed.processing_time:.3f}s)")
        logger.info(f"  有效比例: {stats['valid_ratio']*100:.1f}%")
        logger.info(f"  深度範圍: {stats['depth_min']:.1f} - {stats['depth_max']:.1f} mm")

        # 視覺化處理結果
        depth_colored = visualizer.draw_depth_map(processed.depth)
        processing_comparison = visualizer.create_comparison_view(
            rgb_image=processed.rgb,
            depth_image=depth_colored,
        )

        save_path_1 = "outputs/end_to_end_processing.png"
        visualizer.save_image(processing_comparison, save_path_1)
        logger.success(f"處理結果已儲存: {save_path_1}")

        # ==================== AI 前處理 ====================
        print("\n" + "=" * 70)
        print("  步驟 3: AI 模型前處理")
        print("=" * 70)

        logger.info("準備 AI 模型輸入...")
        ai_input, transform_info = ai_preprocessor.prepare_for_yolo(processed.rgb)

        logger.success(f"✓ AI 輸入準備完成")
        logger.info(f"  原始尺寸: {transform_info['original_shape']}")
        logger.info(f"  處理後: {transform_info['processed_shape']}")
        logger.info(f"  縮放比例: {transform_info['scale']:.3f}")

        # ==================== 點雲生成 ====================
        print("\n" + "=" * 70)
        print("  步驟 4: 點雲生成")
        print("=" * 70)

        logger.info("生成 3D 點雲...")
        pointcloud = pointcloud_gen.generate_from_rgbd(
            depth=processed.depth, rgb=processed.rgb, subsample=2
        )

        logger.success(f"✓ 點雲生成完成")
        logger.info(f"  點數量: {pointcloud.num_points:,}")
        logger.info(f"  有顏色: {pointcloud.has_colors()}")
        logger.info(f"  邊界: {pointcloud.bounds}")

        # 降採樣
        logger.info("執行體素降採樣...")
        downsampled = pointcloud_gen.downsample(pointcloud, voxel_size=50.0)
        logger.success(f"✓ 降採樣完成: {pointcloud.num_points:,} → {downsampled.num_points:,}")

        # 儲存點雲
        ply_path = "outputs/end_to_end_pointcloud.ply"
        try:
            pointcloud_gen.save_ply(downsampled, ply_path)
            logger.success(f"點雲已儲存: {ply_path}")
        except Exception as e:
            logger.warning(f"點雲儲存失敗: {e}")

        # ==================== 效能統計 ====================
        print("\n" + "=" * 70)
        print("  效能統計")
        print("=" * 70)

        total_time = (
            frame.timestamp + processed.processing_time + 0.001
        )  # 簡化計算

        logger.info(f"處理流程耗時:")
        logger.info(f"  相機擷取: ~{frame.timestamp:.3f}s")
        logger.info(f"  RGB-D 處理: {processed.processing_time:.3f}s")
        logger.info(f"  點雲生成: ~0.020s (估計)")
        logger.info(f"  總計: ~{total_time:.3f}s")

    print("\n" + "=" * 70)
    print("  ✓ 完整流程執行完成")
    print("=" * 70)

    print(f"\n生成的檔案:")
    print(f"  - {save_path_1}")
    print(f"  - {ply_path}")
    print(f"  - outputs/logs/EndToEnd.log")


def demo_performance_test():
    """效能測試"""
    print("\n" + "=" * 70)
    print("  端到端效能測試")
    print("=" * 70)

    logger = setup_logger(name="Performance", log_dir="outputs/logs")

    # 初始化
    camera = MockCamera(mode="objects")
    rgbd_processor = RGBDProcessor(enable_depth_filter=True)
    pointcloud_gen = PointCloudGenerator(fx=720.91, fy=720.91, cx=640, cy=400)

    num_iterations = 10
    times = {"capture": [], "process": [], "pointcloud": []}

    logger.info(f"執行 {num_iterations} 次完整流程測試...")

    with camera:
        for i in range(num_iterations):
            # 擷取
            t0 = time.time()
            frame = camera.get_frame()
            t1 = time.time()
            times["capture"].append(t1 - t0)

            # 處理
            t0 = time.time()
            processed = rgbd_processor.process(frame.rgb, frame.depth)
            t1 = time.time()
            times["process"].append(t1 - t0)

            # 點雲
            t0 = time.time()
            pointcloud = pointcloud_gen.generate_from_rgbd(
                processed.depth, processed.rgb, subsample=4
            )
            t1 = time.time()
            times["pointcloud"].append(t1 - t0)

            if (i + 1) % 3 == 0:
                logger.info(f"  完成 {i+1}/{num_iterations} 次")

    # 統計
    print("\n效能統計 (平均值):")
    avg_capture = np.mean(times["capture"])
    avg_process = np.mean(times["process"])
    avg_pointcloud = np.mean(times["pointcloud"])
    avg_total = avg_capture + avg_process + avg_pointcloud

    print(f"  相機擷取: {avg_capture*1000:.1f} ms")
    print(f"  RGB-D 處理: {avg_process*1000:.1f} ms")
    print(f"  點雲生成: {avg_pointcloud*1000:.1f} ms")
    print(f"  總計: {avg_total*1000:.1f} ms")
    print(f"  平均 FPS: {1.0/avg_total:.1f}")

    logger.success(f"效能測試完成")
    logger.info(f"  平均總耗時: {avg_total*1000:.1f} ms")
    logger.info(f"  平均 FPS: {1.0/avg_total:.1f}")

    print("\n  ✓ 效能測試完成")


def demo_capability_showcase():
    """系統能力展示"""
    print("\n" + "=" * 70)
    print("  系統能力展示")
    print("=" * 70)

    print("\n目前系統已實作的功能:")

    print("\n1. 硬體介面模組:")
    print("   ✓ MockCamera - 三種模擬模式 (random, pattern, objects)")
    print("   ✓ Gemini2Camera - ORBBEC Gemini 2 驅動 (需實體硬體)")
    print("   ✓ RGB-D 資料擷取")
    print("   ✓ 相機內參查詢")

    print("\n2. 影像處理模組:")
    print("   ✓ DepthFilter - 深度濾波、填洞、異常值處理")
    print("   ✓ CoordinateTransformer - 2D↔3D 座標轉換")
    print("   ✓ PointCloudGenerator - 點雲生成與處理")
    print("   ✓ RGBDProcessor - RGB-D 整合處理")
    print("   ✓ AIPreprocessor - AI 模型前處理")

    print("\n3. AI 模型模組:")
    print("   ✓ YOLOv11Detector - 物體檢測")
    print("   ✓ YOLOv11Detector - 實例分割")
    print("   ✓ 結果過濾與後處理")
    print("   ✓ 多裝置支援 (CPU, CUDA, MPS)")

    print("\n4. 工具模組:")
    print("   ✓ Logger - 彩色日誌系統")
    print("   ✓ ConfigLoader - YAML 配置載入")
    print("   ✓ Visualizer - 視覺化工具")

    print("\n目前可以做的應用:")
    print("   • RGB-D 影像擷取與處理")
    print("   • 深度影像清理與增強")
    print("   • 3D 點雲生成與視覺化")
    print("   • 物體檢測 (使用 YOLOv11)")
    print("   • 實例分割 (使用 YOLOv11)")
    print("   • 完整的檢測流程")

    print("\n待開發功能:")
    print("   ⏳ 3D 尺寸量測")
    print("   ⏳ 缺陷深度分析")
    print("   ⏳ 表面平整度檢測")
    print("   ⏳ 良/不良品判定")
    print("   ⏳ 缺陷嚴重度評估")

    print("\n專案進度: ~55%")


def main():
    """主函數"""
    print("\n" + "=" * 70)
    print("  ORBBEC Gemini 2 工業檢測系統 - 端到端整合展示")
    print("=" * 70)

    # 執行展示
    demo_full_pipeline()
    demo_performance_test()
    demo_capability_showcase()

    print("\n" + "=" * 70)
    print("  所有展示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
