"""
工具模組測試
測試 logger, config_loader, visualization 模組
"""

import sys
from pathlib import Path

# 加入專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.utils.logger import setup_logger, get_logger
from src.utils.config_loader import ConfigLoader, get_config_loader
from src.utils.visualization import Visualizer, get_visualizer


def test_logger():
    """測試日誌模組"""
    print("\n" + "=" * 60)
    print("測試日誌模組")
    print("=" * 60)

    # 設定 logger
    logger = setup_logger(
        name="TestLogger",
        log_dir="outputs/logs",
        log_level="DEBUG",
    )

    # 測試不同等級的日誌
    logger.debug("這是除錯訊息")
    logger.info("這是一般訊息")
    logger.warning("這是警告訊息")
    logger.success("這是成功訊息")

    # 測試例外處理
    try:
        result = 10 / 0
    except ZeroDivisionError:
        logger.exception("捕獲到除零錯誤")

    # 測試全域 logger
    global_logger = get_logger()
    global_logger.info("使用全域 logger")

    print("✓ 日誌模組測試完成")
    print(f"  日誌檔案位置: outputs/logs/")


def test_config_loader():
    """測試配置載入模組"""
    print("\n" + "=" * 60)
    print("測試配置載入模組")
    print("=" * 60)

    # 建立配置載入器
    loader = ConfigLoader(config_dir="config")

    # 測試載入相機配置
    try:
        camera_config = loader.load_camera_config()
        print("✓ 成功載入相機配置")

        # 顯示部分配置
        camera_model = camera_config.get("camera", {}).get("model")
        rgb_width = camera_config.get("camera", {}).get("rgb", {}).get("width")
        rgb_height = camera_config.get("camera", {}).get("rgb", {}).get("height")

        print(f"  相機型號: {camera_model}")
        print(f"  RGB 解析度: {rgb_width}x{rgb_height}")

    except Exception as e:
        print(f"✗ 載入相機配置失敗: {e}")

    # 測試載入模型配置
    try:
        model_config = loader.load_model_config()
        print("✓ 成功載入模型配置")

        # 顯示部分配置
        model_type = model_config.get("model", {}).get("type")
        model_variant = model_config.get("model", {}).get("variant")
        model_task = model_config.get("model", {}).get("task")

        print(f"  模型類型: {model_type}")
        print(f"  模型變體: {model_variant}")
        print(f"  任務類型: {model_task}")

    except Exception as e:
        print(f"✗ 載入模型配置失敗: {e}")

    # 測試便利方法
    rgb_fps = loader.get_camera_setting("camera", "rgb", "fps", default=30)
    print(f"  RGB FPS: {rgb_fps}")

    conf_threshold = loader.get_model_setting("inference", "conf", default=0.25)
    print(f"  信心閾值: {conf_threshold}")

    # 測試全域配置載入器
    global_loader = get_config_loader()
    print("✓ 配置載入模組測試完成")


def test_visualizer():
    """測試視覺化模組"""
    print("\n" + "=" * 60)
    print("測試視覺化模組")
    print("=" * 60)

    # 建立視覺化工具
    visualizer = Visualizer()

    # 建立測試影像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (220, 220, 220)  # 淺灰色背景

    # 測試繪製邊界框
    boxes = np.array([
        [50, 50, 200, 200],
        [250, 100, 400, 300],
        [100, 300, 250, 450],
    ])

    labels = ["metal_part", "plastic_part", "defect"]
    scores = np.array([0.95, 0.87, 0.76])
    class_ids = np.array([0, 1, 2])

    result_image = visualizer.draw_detection_results(
        test_image.copy(),
        boxes=boxes,
        labels=labels,
        scores=scores,
        class_ids=class_ids,
    )

    # 儲存結果
    visualizer.save_image(result_image, "outputs/test_detection.png")
    print("✓ 成功繪製檢測結果")
    print(f"  結果圖片: outputs/test_detection.png")

    # 測試深度圖
    depth = np.random.rand(480, 640) * 2000  # 模擬深度值（mm）
    depth_colored = visualizer.draw_depth_map(depth)
    visualizer.save_image(depth_colored, "outputs/test_depth.png")
    print("✓ 成功繪製深度圖")
    print(f"  深度圖片: outputs/test_depth.png")

    # 測試比較視圖
    rgb_image = test_image.copy()
    comparison = visualizer.create_comparison_view(
        rgb_image, depth_colored, result_image
    )
    visualizer.save_image(comparison, "outputs/test_comparison.png")
    print("✓ 成功建立比較視圖")
    print(f"  比較圖片: outputs/test_comparison.png")

    # 測試指標繪製
    metrics = {
        "mAP": 0.85,
        "Precision": 0.92,
        "Recall": 0.88,
        "F1-Score": 0.90,
    }
    visualizer.plot_metrics(metrics, title="測試指標", save_path="outputs/test_metrics.png")
    print("✓ 成功繪製指標圖表")
    print(f"  指標圖表: outputs/test_metrics.png")

    # 測試全域視覺化工具
    global_visualizer = get_visualizer()
    print("✓ 視覺化模組測試完成")


def main():
    """主測試函數"""
    print("\n" + "=" * 60)
    print("工具模組測試開始")
    print("=" * 60)

    # 測試各個模組
    test_logger()
    test_config_loader()
    test_visualizer()

    print("\n" + "=" * 60)
    print("所有測試完成！")
    print("=" * 60)
    print("\n生成的檔案：")
    print("  - outputs/logs/TestLogger.log")
    print("  - outputs/logs/TestLogger_error.log")
    print("  - outputs/test_detection.png")
    print("  - outputs/test_depth.png")
    print("  - outputs/test_comparison.png")
    print("  - outputs/test_metrics.png")


if __name__ == "__main__":
    main()
