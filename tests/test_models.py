"""
AI 模型模組測試
測試檢測器介面與 YOLO 檢測器
"""

import sys
from pathlib import Path

# 加入專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2
from src.models import (
    DetectionBox,
    SegmentationMask,
    DetectionResult,
    YOLOv11Detector,
)
from src.utils import setup_logger, get_visualizer


def test_detection_box():
    """測試 DetectionBox 資料類別"""
    print("\n" + "=" * 60)
    print("測試 DetectionBox")
    print("=" * 60)

    # 建立測試邊界框
    box = DetectionBox(
        x1=100,
        y1=150,
        x2=300,
        y2=400,
        confidence=0.95,
        class_id=0,
        class_name="metal_part",
    )

    print(f"  ✓ 建立邊界框")
    print(f"    位置: ({box.x1}, {box.y1}) -> ({box.x2}, {box.y2})")
    print(f"    寬高: {box.width} × {box.height}")
    print(f"    中心: {box.center}")
    print(f"    面積: {box.area}")
    print(f"    類別: {box.class_name} (ID: {box.class_id})")
    print(f"    信心度: {box.confidence:.2f}")

    # 測試格式轉換
    print(f"\n  格式轉換:")
    print(f"    xyxy: {box.to_xyxy()}")
    print(f"    xywh: {box.to_xywh()}")
    print(f"    cxcywh: {box.to_cxcywh()}")


def test_segmentation_mask():
    """測試 SegmentationMask 資料類別"""
    print("\n" + "=" * 60)
    print("測試 SegmentationMask")
    print("=" * 60)

    # 建立測試遮罩
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 1  # 建立一個方形遮罩

    seg_mask = SegmentationMask(
        mask=mask,
        confidence=0.92,
        class_id=1,
        class_name="defect",
    )

    print(f"  ✓ 建立分割遮罩")
    print(f"    尺寸: {seg_mask.shape}")
    print(f"    面積: {seg_mask.area} 像素")
    print(f"    中心: {seg_mask.get_center()}")
    print(f"    類別: {seg_mask.class_name} (ID: {seg_mask.class_id})")
    print(f"    信心度: {seg_mask.confidence:.2f}")

    # 測試輪廓提取
    contours = seg_mask.get_contours()
    print(f"    輪廓數量: {len(contours)}")


def test_detection_result():
    """測試 DetectionResult 資料類別"""
    print("\n" + "=" * 60)
    print("測試 DetectionResult")
    print("=" * 60)

    # 建立測試結果
    boxes = [
        DetectionBox(100, 100, 200, 200, 0.95, 0, "metal"),
        DetectionBox(300, 300, 400, 450, 0.88, 1, "plastic"),
        DetectionBox(500, 200, 600, 300, 0.72, 2, "defect"),
    ]

    result = DetectionResult(
        boxes=boxes,
        masks=[],
        inference_time=0.05,
        image_shape=(640, 640),
    )

    print(f"  ✓ 建立檢測結果")
    print(f"    檢測數量: {result.num_detections}")
    print(f"    推論時間: {result.inference_time:.3f}s")
    print(f"    影像尺寸: {result.image_shape}")

    # 測試類別統計
    class_counts = result.get_class_counts()
    print(f"\n  類別統計:")
    for class_name, count in class_counts.items():
        print(f"    {class_name}: {count}")

    # 測試過濾
    filtered = result.filter_by_confidence(0.8)
    print(f"\n  信心度過濾 (>= 0.8):")
    print(f"    原始: {result.num_detections} 個")
    print(f"    過濾後: {filtered.num_detections} 個")

    filtered_by_class = result.filter_by_class([0, 1])
    print(f"\n  類別過濾 (ID: 0, 1):")
    print(f"    原始: {result.num_detections} 個")
    print(f"    過濾後: {filtered_by_class.num_detections} 個")


def test_yolo_detector_init():
    """測試 YOLOv11Detector 初始化"""
    print("\n" + "=" * 60)
    print("測試 YOLOv11Detector - 初始化")
    print("=" * 60)

    # 建立檢測器
    detector = YOLOv11Detector(task="detect")

    print(f"  ✓ 建立檢測器")
    print(f"    YOLO 可用: {detector.yolo_available}")
    print(f"    任務類型: {detector.task}")
    print(f"    已載入: {detector.is_loaded}")

    # 取得模型資訊
    info = detector.get_model_info()
    print(f"\n  模型資訊:")
    for key, value in info.items():
        print(f"    {key}: {value}")

    return detector.yolo_available


def test_yolo_detector_with_pretrained():
    """測試 YOLOv11Detector 使用預訓練模型"""
    print("\n" + "=" * 60)
    print("測試 YOLOv11Detector - 預訓練模型")
    print("=" * 60)

    logger = setup_logger(name="ModelTest", log_dir="outputs/logs")
    visualizer = get_visualizer()

    try:
        # 建立檢測器
        detector = YOLOv11Detector(task="detect")

        # 載入預訓練模型 (nano 版本)
        logger.info("載入 YOLOv11 nano 模型...")
        success = detector.load_model("yolo11n.pt", device="cpu", verbose=True)

        if not success:
            logger.error("模型載入失敗")
            return

        logger.success("模型載入成功")

        # 建立測試影像
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # 在影像上繪製一些簡單的物體形狀
        cv2.rectangle(test_image, (100, 100), (300, 300), (0, 255, 0), -1)
        cv2.circle(test_image, (450, 450), 80, (255, 0, 0), -1)

        # 執行檢測
        logger.info("執行檢測...")
        result = detector.detect(test_image, conf_threshold=0.25)

        logger.success(f"檢測完成 (耗時 {result.inference_time:.3f}s)")
        logger.info(f"檢測到 {result.num_detections} 個物體")

        # 顯示結果
        if result.num_detections > 0:
            class_counts = result.get_class_counts()
            logger.info("檢測結果:")
            for class_name, count in class_counts.items():
                logger.info(f"  {class_name}: {count}")

            # 視覺化結果
            result_image = visualizer.draw_detection_results(
                image=test_image.copy(),
                boxes=[b.to_xyxy() for b in result.boxes],
                masks=None,
                labels=[b.class_name for b in result.boxes],
                scores=[b.confidence for b in result.boxes],
                class_ids=[b.class_id for b in result.boxes],
            )

            # 儲存結果
            save_path = "outputs/yolo_test_detection.png"
            visualizer.save_image(result_image, save_path)
            logger.success(f"結果已儲存: {save_path}")
        else:
            logger.warning("未檢測到任何物體")

        print(f"\n  ✓ 完成")

    except Exception as e:
        logger.error(f"測試失敗: {e}")
        print(f"\n  ✗ 測試失敗: {e}")


def main():
    """主測試函數"""
    print("\n" + "=" * 60)
    print("AI 模型模組測試開始")
    print("=" * 60)

    # 執行基本測試
    test_detection_box()
    test_segmentation_mask()
    test_detection_result()
    yolo_available = test_yolo_detector_init()

    # 如果 YOLO 可用,執行額外測試
    if yolo_available:
        print("\n" + "=" * 60)
        print("✓ Ultralytics YOLO 可用")
        print("=" * 60)
        print("\n若要測試預訓練模型,請執行:")
        print("  python tests/test_models.py --with-model")
        print("\n注意: 首次執行會下載 yolo11n.pt (~6MB)")
    else:
        print("\n" + "=" * 60)
        print("⚠ Ultralytics YOLO 未安裝")
        print("=" * 60)
        print("\n若要使用 YOLOv11 檢測器,請執行:")
        print("  pip install ultralytics")

    print("\n" + "=" * 60)
    print("測試完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
