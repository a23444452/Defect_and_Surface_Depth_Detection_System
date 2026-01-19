#!/usr/bin/env python3
"""
自動展示腳本
自動執行所有範例並生成結果
"""

import sys
from pathlib import Path

# 加入專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from src.utils import setup_logger, get_config_loader, get_visualizer


def print_section(title):
    """列印區段標題"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_all():
    """執行所有示範"""
    print_section("🚀 ORBBEC Gemini 2 工具模組自動展示")

    # ========================================
    # 1. Logger 示範
    # ========================================
    print_section("📝 1/7 - Logger 日誌系統")

    logger = setup_logger(
        name="AutoDemo",
        log_dir="outputs/logs",
        log_level="INFO"
    )

    logger.info("🔧 系統初始化中...")
    logger.info("✓ Logger 模組已載入")

    # ========================================
    # 2. Config Loader 示範
    # ========================================
    print_section("⚙️  2/7 - Config Loader 配置載入")

    config_loader = get_config_loader(config_dir="config")

    # 載入相機配置
    camera_config = config_loader.load_camera_config()
    camera_model = config_loader.get_camera_setting("camera", "model")
    rgb_width = config_loader.get_camera_setting("camera", "rgb", "width")
    rgb_height = config_loader.get_camera_setting("camera", "rgb", "height")

    print(f"  📷 相機型號: {camera_model}")
    print(f"  📐 RGB 解析度: {rgb_width} x {rgb_height}")
    logger.info(f"相機配置已載入 - {camera_model}")

    # 載入模型配置
    model_config = config_loader.load_model_config()
    model_type = config_loader.get_model_setting("model", "type")
    model_variant = config_loader.get_model_setting("model", "variant")
    conf_threshold = config_loader.get_model_setting("inference", "conf")

    print(f"  🤖 模型類型: {model_type}")
    print(f"  🎯 模型變體: {model_variant}")
    print(f"  🎚️  信心閾值: {conf_threshold}")
    logger.info(f"模型配置已載入 - {model_type}-{model_variant}")

    # ========================================
    # 3. Visualizer 基本繪製
    # ========================================
    print_section("🎨 3/7 - Visualizer 基本視覺化")

    visualizer = get_visualizer()

    # 建立測試影像
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (235, 235, 235)

    # 模擬檢測結果
    boxes = np.array([
        [80, 80, 280, 220],
        [320, 120, 540, 340],
        [120, 280, 320, 440],
    ])

    labels = ["metal_part", "plastic_part", "defect_scratch"]
    scores = np.array([0.96, 0.89, 0.78])
    class_ids = np.array([0, 1, 2])

    result = visualizer.draw_detection_results(
        image.copy(),
        boxes=boxes,
        labels=labels,
        scores=scores,
        class_ids=class_ids
    )

    save_path = "outputs/auto_demo_detection.png"
    visualizer.save_image(result, save_path)
    print(f"  ✓ 檢測結果已儲存: {save_path}")
    logger.success(f"檢測結果視覺化完成")

    # ========================================
    # 4. 深度圖視覺化
    # ========================================
    print_section("🌈 4/7 - 深度圖視覺化")

    # 建立模擬深度資料
    depth = np.zeros((480, 640))

    # 建立不同深度的區域（模擬真實場景）
    depth[120:240, 120:240] = 600   # 近距離物體 (0.6m)
    depth[140:320, 340:520] = 1200  # 中距離物體 (1.2m)
    depth[300:420, 140:340] = 1800  # 遠距離物體 (1.8m)

    # 加入噪點模擬真實感測器
    depth += np.random.randn(480, 640) * 30

    # 使用不同色彩映射
    colormaps = [
        (cv2.COLORMAP_JET, "jet"),
        (cv2.COLORMAP_TURBO, "turbo"),
        (cv2.COLORMAP_VIRIDIS, "viridis"),
    ]

    print("  生成不同色彩映射的深度圖:")
    for colormap, name in colormaps:
        depth_colored = visualizer.draw_depth_map(depth, colormap=colormap)
        save_path = f"outputs/auto_demo_depth_{name}.png"
        visualizer.save_image(depth_colored, save_path)
        print(f"    ✓ {name.upper():8s} -> {save_path}")

    logger.info("深度圖視覺化完成（3 種色彩映射）")

    # ========================================
    # 5. 比較視圖
    # ========================================
    print_section("🖼️  5/7 - RGB-D 比較視圖")

    # 建立 RGB 影像（添加一些視覺元素）
    rgb_image = image.copy()
    cv2.circle(rgb_image, (180, 150), 60, (150, 180, 200), -1)
    cv2.circle(rgb_image, (430, 230), 80, (180, 200, 150), -1)
    cv2.rectangle(rgb_image, (220, 340), (260, 390), (200, 150, 180), -1)

    # 使用 JET 色彩映射的深度圖
    depth_colored = visualizer.draw_depth_map(depth, colormap=cv2.COLORMAP_JET)

    # 建立比較視圖
    comparison = visualizer.create_comparison_view(
        rgb_image=rgb_image,
        depth_image=depth_colored,
        detection_image=result
    )

    save_path = "outputs/auto_demo_comparison.png"
    visualizer.save_image(comparison, save_path)
    print(f"  ✓ 比較視圖已儲存: {save_path}")
    logger.success("RGB-D 比較視圖已生成")

    # ========================================
    # 6. 指標圖表
    # ========================================
    print_section("📊 6/7 - 模型效能指標")

    # 模擬訓練指標
    metrics = {
        "mAP@0.5": 0.856,
        "mAP@0.75": 0.724,
        "Precision": 0.918,
        "Recall": 0.883,
        "F1-Score": 0.900,
        "IoU": 0.782,
    }

    print("  模型效能指標:")
    for metric, value in metrics.items():
        print(f"    {metric:12s}: {value:.3f}")

    save_path = "outputs/auto_demo_metrics.png"
    visualizer.plot_metrics(
        metrics,
        title="YOLOv11 模型效能指標 - 工業檢測系統",
        save_path=save_path
    )
    print(f"  ✓ 指標圖表已儲存: {save_path}")
    logger.info("效能指標圖表已生成")

    # ========================================
    # 7. 完整工作流程
    # ========================================
    print_section("🔄 7/7 - 完整檢測工作流程模擬")

    print("  [步驟 1/5] 載入系統配置...")
    depth_min = config_loader.get_camera_setting(
        "depth_processing", "depth_range", "min"
    )
    depth_max = config_loader.get_camera_setting(
        "depth_processing", "depth_range", "max"
    )
    print(f"    ✓ 深度範圍: {depth_min}mm - {depth_max}mm")

    print("  [步驟 2/5] 擷取 RGB-D 影像...")
    print(f"    ✓ RGB 影像: {rgb_image.shape}")
    print(f"    ✓ 深度影像: {depth.shape}")
    logger.info("影像擷取完成")

    print("  [步驟 3/5] 執行 AI 推理...")
    print(f"    ✓ 檢測到 {len(boxes)} 個物體")
    for i, (label, score) in enumerate(zip(labels, scores)):
        print(f"      - 物體 {i+1}: {label:18s} (信心度: {score:.2f})")
    logger.info(f"AI 推理完成 - 檢測到 {len(boxes)} 個物體")

    print("  [步驟 4/5] 3D 量測分析...")
    # 模擬量測結果
    measurements = {
        "metal_part": {"width": 185.3, "height": 132.7, "depth_avg": 608.2},
        "plastic_part": {"width": 204.6, "height": 198.3, "depth_avg": 1195.7},
        "defect_scratch": {"width": 178.9, "height": 144.2, "depth_avg": 1782.4},
    }

    for label, measure in measurements.items():
        print(f"    ✓ {label}:")
        print(f"      尺寸: {measure['width']:.1f}mm x {measure['height']:.1f}mm")
        print(f"      平均深度: {measure['depth_avg']:.1f}mm")

    logger.info("3D 量測分析完成")

    print("  [步驟 5/5] 生成檢測報告...")

    # 建立最終結果（包含量測資訊）
    final_result = result.copy()

    # 加入標題
    final_result = visualizer.draw_text(
        final_result,
        text="ORBBEC Gemini 2 - Industrial Inspection System",
        position=(10, 25),
        color=(0, 120, 255),
        background=True
    )

    # 加入檢測統計
    stats_text = f"Detected: {len(boxes)} objects | Avg Conf: {scores.mean():.2f}"
    final_result = visualizer.draw_text(
        final_result,
        text=stats_text,
        position=(10, 460),
        color=(0, 255, 0),
        background=True
    )

    save_path = "outputs/auto_demo_final_result.png"
    visualizer.save_image(final_result, save_path)
    print(f"    ✓ 最終結果已儲存: {save_path}")
    logger.success("完整工作流程執行完成")

    # ========================================
    # 總結
    # ========================================
    print_section("✨ 展示完成 - 結果總覽")

    print("\n  📁 生成的檔案:")
    print("    日誌檔案:")
    print("      - outputs/logs/AutoDemo.log")
    print("      - outputs/logs/AutoDemo_error.log")
    print("\n    視覺化結果:")
    print("      - outputs/auto_demo_detection.png       (檢測結果)")
    print("      - outputs/auto_demo_depth_jet.png       (深度圖 - JET)")
    print("      - outputs/auto_demo_depth_turbo.png     (深度圖 - TURBO)")
    print("      - outputs/auto_demo_depth_viridis.png   (深度圖 - VIRIDIS)")
    print("      - outputs/auto_demo_comparison.png      (RGB-D 比較視圖)")
    print("      - outputs/auto_demo_metrics.png         (效能指標)")
    print("      - outputs/auto_demo_final_result.png    (最終結果)")

    print("\n  📊 系統統計:")
    print(f"    ✓ 處理影像: {rgb_image.shape[1]}x{rgb_image.shape[0]}")
    print(f"    ✓ 檢測物體: {len(boxes)} 個")
    print(f"    ✓ 平均信心度: {scores.mean():.3f}")
    print(f"    ✓ 模型 mAP: {metrics['mAP@0.5']:.3f}")

    print("\n  🎯 下一步建議:")
    print("    1. 查看生成的視覺化結果圖片")
    print("    2. 閱讀 docs/UTILS_USAGE.md 了解詳細用法")
    print("    3. 執行 python scripts/interactive_demo.py 互動式體驗")

    logger.success("所有展示項目執行完成！")

    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    demo_all()
