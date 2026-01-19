#!/usr/bin/env python3
"""
工具模組示範腳本
展示如何使用 logger, config_loader, visualization 模組
"""

import sys
from pathlib import Path

# 加入專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.utils import (
    setup_logger,
    get_config_loader,
    get_visualizer,
)


def main():
    """主函數"""
    print("\n" + "=" * 60)
    print("工具模組使用示範")
    print("=" * 60)

    # 1. 設定日誌系統
    print("\n1. 設定日誌系統...")
    logger = setup_logger(
        name="DemoSystem",
        log_dir="outputs/logs",
        log_level="INFO",
    )
    logger.info("日誌系統已初始化")
    logger.success("準備開始示範")

    # 2. 載入配置檔案
    print("\n2. 載入配置檔案...")
    config_loader = get_config_loader(config_dir="config")

    # 載入相機配置
    camera_config = config_loader.load_camera_config()
    logger.info(f"相機型號: {camera_config.get('camera', {}).get('model')}")
    logger.info(
        f"RGB 解析度: {camera_config.get('camera', {}).get('rgb', {}).get('width')}x"
        f"{camera_config.get('camera', {}).get('rgb', {}).get('height')}"
    )

    # 載入模型配置
    model_config = config_loader.load_model_config()
    logger.info(f"模型類型: {model_config.get('model', {}).get('type')}")
    logger.info(f"模型變體: {model_config.get('model', {}).get('variant')}")

    # 3. 使用視覺化工具
    print("\n3. 使用視覺化工具...")
    visualizer = get_visualizer()

    # 建立示範影像
    demo_image = np.zeros((480, 640, 3), dtype=np.uint8)
    demo_image[:] = (230, 230, 230)

    # 模擬檢測結果
    boxes = np.array([
        [100, 100, 300, 250],
        [350, 150, 500, 350],
    ])
    labels = ["metal_part", "plastic_part"]
    scores = np.array([0.92, 0.85])
    class_ids = np.array([0, 1])

    # 繪製檢測結果
    result_image = visualizer.draw_detection_results(
        demo_image,
        boxes=boxes,
        labels=labels,
        scores=scores,
        class_ids=class_ids,
    )

    # 儲存結果
    save_path = "outputs/demo_result.png"
    visualizer.save_image(result_image, save_path)
    logger.success(f"檢測結果已儲存至: {save_path}")

    # 4. 使用配置載入器的便利方法
    print("\n4. 使用配置載入器便利方法...")
    depth_min = config_loader.get_camera_setting(
        "depth_processing", "depth_range", "min", default=150
    )
    depth_max = config_loader.get_camera_setting(
        "depth_processing", "depth_range", "max", default=3000
    )
    logger.info(f"深度範圍: {depth_min}mm - {depth_max}mm")

    conf_threshold = config_loader.get_model_setting(
        "inference", "conf", default=0.25
    )
    iou_threshold = config_loader.get_model_setting(
        "inference", "iou", default=0.7
    )
    logger.info(f"信心閾值: {conf_threshold}")
    logger.info(f"IoU 閾值: {iou_threshold}")

    # 5. 完成
    print("\n" + "=" * 60)
    logger.success("示範完成！")
    print("=" * 60)
    print("\n生成的檔案：")
    print(f"  - {save_path}")
    print("  - outputs/logs/DemoSystem.log")
    print("  - outputs/logs/DemoSystem_error.log")


if __name__ == "__main__":
    main()
