#!/usr/bin/env python3
"""
互動式工具模組展示
提供多個範例讓使用者選擇執行
"""

import sys
from pathlib import Path

# 加入專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from src.utils import setup_logger, get_config_loader, get_visualizer


def demo_logger():
    """範例 1: 日誌系統示範"""
    print("\n" + "=" * 60)
    print("範例 1: 日誌系統")
    print("=" * 60)

    # 建立 logger
    logger = setup_logger(
        name="DemoLogger",
        log_dir="outputs/logs",
        log_level="DEBUG"
    )

    # 展示不同等級的日誌
    logger.debug("這是 DEBUG 等級訊息 - 用於除錯資訊")
    logger.info("這是 INFO 等級訊息 - 用於一般資訊")
    logger.warning("這是 WARNING 等級訊息 - 用於警告")
    logger.error("這是 ERROR 等級訊息 - 用於錯誤")
    logger.success("這是 SUCCESS 等級訊息 - 用於成功操作")

    # 展示例外處理
    try:
        result = 10 / 0
    except ZeroDivisionError:
        logger.exception("展示例外處理 - 會自動記錄堆疊追蹤")

    print("\n✓ 日誌已記錄到: outputs/logs/DemoLogger.log")
    input("\n按 Enter 繼續...")


def demo_config_loader():
    """範例 2: 配置載入示範"""
    print("\n" + "=" * 60)
    print("範例 2: 配置載入")
    print("=" * 60)

    loader = get_config_loader(config_dir="config")

    # 載入相機配置
    print("\n--- 相機配置 ---")
    camera_config = loader.load_camera_config()

    camera_model = loader.get_camera_setting("camera", "model")
    rgb_width = loader.get_camera_setting("camera", "rgb", "width")
    rgb_height = loader.get_camera_setting("camera", "rgb", "height")
    rgb_fps = loader.get_camera_setting("camera", "rgb", "fps")

    print(f"相機型號: {camera_model}")
    print(f"RGB 解析度: {rgb_width} x {rgb_height}")
    print(f"RGB FPS: {rgb_fps}")

    # 載入模型配置
    print("\n--- 模型配置 ---")
    model_config = loader.load_model_config()

    model_type = loader.get_model_setting("model", "type")
    model_variant = loader.get_model_setting("model", "variant")
    conf_threshold = loader.get_model_setting("inference", "conf")
    iou_threshold = loader.get_model_setting("inference", "iou")

    print(f"模型類型: {model_type}")
    print(f"模型變體: {model_variant}")
    print(f"信心閾值: {conf_threshold}")
    print(f"IoU 閾值: {iou_threshold}")

    # 展示巢狀配置讀取
    print("\n--- 深度處理配置 ---")
    depth_min = loader.get_camera_setting(
        "depth_processing", "depth_range", "min"
    )
    depth_max = loader.get_camera_setting(
        "depth_processing", "depth_range", "max"
    )

    print(f"深度範圍: {depth_min}mm - {depth_max}mm")

    input("\n按 Enter 繼續...")


def demo_visualizer_basic():
    """範例 3: 基本視覺化"""
    print("\n" + "=" * 60)
    print("範例 3: 基本視覺化")
    print("=" * 60)

    visualizer = get_visualizer()

    # 建立測試影像
    print("\n建立測試影像...")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (240, 240, 240)  # 淺灰色背景

    # 繪製邊界框
    print("繪製邊界框...")
    boxes = np.array([
        [50, 50, 200, 200],
        [250, 100, 400, 300],
        [100, 320, 280, 450],
    ])

    labels = ["metal_part", "plastic_part", "defect"]
    scores = np.array([0.95, 0.87, 0.76])
    class_ids = np.array([0, 1, 2])

    result = visualizer.draw_detection_results(
        image.copy(),
        boxes=boxes,
        labels=labels,
        scores=scores,
        class_ids=class_ids
    )

    # 儲存結果
    save_path = "outputs/demo_basic_detection.png"
    visualizer.save_image(result, save_path)
    print(f"✓ 檢測結果已儲存至: {save_path}")

    # 加入文字標註
    result_with_text = visualizer.draw_text(
        result,
        text="檢測結果示範",
        position=(10, 30),
        color=(0, 255, 0)
    )

    save_path_text = "outputs/demo_with_text.png"
    visualizer.save_image(result_with_text, save_path_text)
    print(f"✓ 文字標註結果已儲存至: {save_path_text}")

    input("\n按 Enter 繼續...")


def demo_visualizer_depth():
    """範例 4: 深度視覺化"""
    print("\n" + "=" * 60)
    print("範例 4: 深度視覺化")
    print("=" * 60)

    visualizer = get_visualizer()

    # 模擬深度資料
    print("\n生成模擬深度資料...")
    depth = np.zeros((480, 640))

    # 建立幾個不同深度的區域
    depth[100:200, 100:200] = 500   # 近距離物體
    depth[150:300, 300:450] = 1000  # 中距離物體
    depth[300:400, 100:300] = 1500  # 遠距離物體

    # 加入隨機噪點
    depth += np.random.randn(480, 640) * 50

    # 繪製深度圖（使用不同色彩映射）
    print("繪製深度圖...")

    colormaps = [
        (cv2.COLORMAP_JET, "jet"),
        (cv2.COLORMAP_VIRIDIS, "viridis"),
        (cv2.COLORMAP_TURBO, "turbo"),
    ]

    for colormap, name in colormaps:
        depth_colored = visualizer.draw_depth_map(depth, colormap=colormap)
        save_path = f"outputs/demo_depth_{name}.png"
        visualizer.save_image(depth_colored, save_path)
        print(f"✓ 深度圖已儲存至: {save_path} (色彩映射: {name})")

    input("\n按 Enter 繼續...")


def demo_visualizer_comparison():
    """範例 5: 比較視圖"""
    print("\n" + "=" * 60)
    print("範例 5: 比較視圖")
    print("=" * 60)

    visualizer = get_visualizer()

    # 建立 RGB 影像
    print("\n建立 RGB 影像...")
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb_image[:] = (200, 220, 200)  # 淡綠色背景
    cv2.circle(rgb_image, (320, 240), 80, (100, 150, 200), -1)

    # 建立深度圖
    print("建立深度圖...")
    depth = np.random.rand(480, 640) * 1500
    depth_colored = visualizer.draw_depth_map(depth)

    # 建立檢測結果
    print("建立檢測結果...")
    detection_result = rgb_image.copy()
    boxes = np.array([[240, 160, 400, 320]])
    labels = ["detected_object"]
    scores = np.array([0.88])
    class_ids = np.array([0])

    detection_result = visualizer.draw_detection_results(
        detection_result,
        boxes=boxes,
        labels=labels,
        scores=scores,
        class_ids=class_ids
    )

    # 建立比較視圖
    print("建立比較視圖...")
    comparison = visualizer.create_comparison_view(
        rgb_image=rgb_image,
        depth_image=depth_colored,
        detection_image=detection_result
    )

    save_path = "outputs/demo_comparison.png"
    visualizer.save_image(comparison, save_path)
    print(f"✓ 比較視圖已儲存至: {save_path}")

    input("\n按 Enter 繼續...")


def demo_visualizer_metrics():
    """範例 6: 指標圖表"""
    print("\n" + "=" * 60)
    print("範例 6: 指標圖表")
    print("=" * 60)

    visualizer = get_visualizer()

    # 模擬訓練指標
    print("\n繪製模型效能指標...")
    metrics = {
        "mAP@0.5": 0.85,
        "mAP@0.75": 0.72,
        "Precision": 0.92,
        "Recall": 0.88,
        "F1-Score": 0.90,
        "IoU": 0.78,
    }

    save_path = "outputs/demo_metrics.png"
    visualizer.plot_metrics(
        metrics,
        title="YOLOv11 模型效能指標",
        save_path=save_path
    )
    print(f"✓ 指標圖表已儲存至: {save_path}")

    input("\n按 Enter 繼續...")


def demo_complete_workflow():
    """範例 7: 完整工作流程"""
    print("\n" + "=" * 60)
    print("範例 7: 完整工作流程（整合所有工具）")
    print("=" * 60)

    # 1. 初始化所有工具
    print("\n[1/6] 初始化工具...")
    logger = setup_logger(
        name="CompleteDemo",
        log_dir="outputs/logs",
        log_level="INFO"
    )
    config_loader = get_config_loader()
    visualizer = get_visualizer()

    logger.info("所有工具已初始化")

    # 2. 載入配置
    print("[2/6] 載入配置...")
    camera_config = config_loader.load_camera_config()
    model_config = config_loader.load_model_config()

    rgb_res = f"{camera_config['camera']['rgb']['width']}x{camera_config['camera']['rgb']['height']}"
    logger.info(f"相機配置已載入 - RGB 解析度: {rgb_res}")

    model_info = f"{model_config['model']['type']}-{model_config['model']['variant']}"
    logger.info(f"模型配置已載入 - 模型: {model_info}")

    # 3. 建立測試資料
    print("[3/6] 建立測試資料...")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (230, 230, 230)
    depth = np.random.rand(480, 640) * 2000

    logger.info(f"測試資料已建立 - 影像大小: {image.shape}")

    # 4. 模擬檢測
    print("[4/6] 模擬檢測...")
    boxes = np.array([
        [100, 100, 250, 220],
        [300, 150, 500, 350],
    ])
    labels = ["metal_part", "defect"]
    scores = np.array([0.94, 0.81])
    class_ids = np.array([0, 1])

    logger.info(f"檢測完成 - 發現 {len(boxes)} 個物體")
    for i, (label, score) in enumerate(zip(labels, scores)):
        logger.info(f"  物體 {i+1}: {label} (信心度: {score:.2f})")

    # 5. 視覺化結果
    print("[5/6] 視覺化結果...")
    detection_result = visualizer.draw_detection_results(
        image.copy(),
        boxes=boxes,
        labels=labels,
        scores=scores,
        class_ids=class_ids
    )

    depth_colored = visualizer.draw_depth_map(depth)

    comparison = visualizer.create_comparison_view(
        rgb_image=image,
        depth_image=depth_colored,
        detection_image=detection_result
    )

    # 6. 儲存結果
    print("[6/6] 儲存結果...")
    save_path = "outputs/demo_complete_workflow.png"
    visualizer.save_image(comparison, save_path)

    logger.success(f"工作流程完成 - 結果已儲存至: {save_path}")

    print("\n✓ 完整工作流程執行完成！")
    input("\n按 Enter 繼續...")


def main():
    """主選單"""
    while True:
        print("\n" + "=" * 60)
        print("工具模組互動式展示")
        print("=" * 60)
        print("\n請選擇要執行的範例：")
        print("  1. 日誌系統示範")
        print("  2. 配置載入示範")
        print("  3. 基本視覺化示範")
        print("  4. 深度視覺化示範")
        print("  5. 比較視圖示範")
        print("  6. 指標圖表示範")
        print("  7. 完整工作流程示範")
        print("  0. 退出")

        choice = input("\n請輸入選項 (0-7): ").strip()

        if choice == "1":
            demo_logger()
        elif choice == "2":
            demo_config_loader()
        elif choice == "3":
            demo_visualizer_basic()
        elif choice == "4":
            demo_visualizer_depth()
        elif choice == "5":
            demo_visualizer_comparison()
        elif choice == "6":
            demo_visualizer_metrics()
        elif choice == "7":
            demo_complete_workflow()
        elif choice == "0":
            print("\n感謝使用！再見！\n")
            break
        else:
            print("\n無效的選項，請重新選擇。")


if __name__ == "__main__":
    main()
