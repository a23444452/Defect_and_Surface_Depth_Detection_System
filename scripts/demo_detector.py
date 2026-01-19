#!/usr/bin/env python3
"""
AI 檢測器示範腳本
展示 YOLOv11 檢測器的使用方式
"""

import sys
from pathlib import Path

# 加入專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2
from src.models import YOLOv11Detector
from src.hardware import MockCamera
from src.utils import setup_logger, get_visualizer


def demo_detector_basic():
    """基本檢測範例"""
    print("\n" + "=" * 70)
    print("  範例 1: 基本物體檢測")
    print("=" * 70)

    logger = setup_logger(name="DetectorDemo", log_dir="outputs/logs")
    visualizer = get_visualizer()

    try:
        # 建立檢測器
        detector = YOLOv11Detector(task="detect")

        # 載入預訓練模型
        logger.info("載入 YOLOv11n 模型...")
        detector.load_model("yolo11n.pt", device="cpu")

        # 建立測試影像
        test_image = np.ones((640, 640, 3), dtype=np.uint8) * 200

        # 繪製一些簡單物體
        cv2.rectangle(test_image, (100, 100), (300, 300), (100, 150, 200), -1)
        cv2.circle(test_image, (450, 200), 80, (200, 100, 150), -1)
        cv2.ellipse(test_image, (200, 500), (100, 50), 0, 0, 360, (150, 200, 100), -1)

        # 執行檢測
        logger.info("執行檢測...")
        result = detector.detect(test_image, conf_threshold=0.25)

        logger.success(f"檢測完成 (耗時 {result.inference_time:.3f}s)")
        logger.info(f"檢測到 {result.num_detections} 個物體")

        # 視覺化結果
        if result.num_detections > 0:
            result_image = visualizer.draw_detection_results(
                image=test_image.copy(),
                boxes=[b.to_xyxy() for b in result.boxes],
                masks=None,
                labels=[b.class_name for b in result.boxes],
                scores=[b.confidence for b in result.boxes],
                class_ids=[b.class_id for b in result.boxes],
            )

            # 儲存結果
            save_path = "outputs/detector_demo_basic.png"
            visualizer.save_image(result_image, save_path)
            logger.success(f"結果已儲存: {save_path}")

            print(f"\n  ✓ 完成")
            print(f"  結果: {save_path}")
        else:
            logger.warning("未檢測到任何物體")
            print(f"\n  ⚠ 未檢測到物體")

    except Exception as e:
        logger.error(f"範例執行失敗: {e}")
        print(f"\n  ✗ 失敗: {e}")


def demo_detector_with_camera():
    """使用相機影像進行檢測"""
    print("\n" + "=" * 70)
    print("  範例 2: 使用相機影像進行檢測")
    print("=" * 70)

    logger = setup_logger(name="DetectorDemo", log_dir="outputs/logs")
    visualizer = get_visualizer()

    try:
        # 建立檢測器
        detector = YOLOv11Detector(task="detect")
        logger.info("載入 YOLOv11n 模型...")
        detector.load_model("yolo11n.pt", device="cpu")

        # 使用模擬相機
        logger.info("使用模擬相機 (objects 模式)")
        with MockCamera(mode="objects") as camera:
            # 取得一幀
            frame = camera.get_frame()
            logger.info(f"取得幀 #{frame.frame_number}")

            # 執行檢測
            logger.info("執行檢測...")
            result = detector.detect(frame.rgb, conf_threshold=0.25)

            logger.success(f"檢測完成 (耗時 {result.inference_time:.3f}s)")
            logger.info(f"檢測到 {result.num_detections} 個物體")

            # 視覺化結果
            result_image = visualizer.draw_detection_results(
                image=frame.rgb.copy(),
                boxes=[b.to_xyxy() for b in result.boxes],
                masks=None,
                labels=[b.class_name for b in result.boxes],
                scores=[b.confidence for b in result.boxes],
                class_ids=[b.class_id for b in result.boxes],
            )

            # 建立比較視圖
            depth_colored = visualizer.draw_depth_map(frame.depth)
            comparison = visualizer.create_comparison_view(
                rgb_image=frame.rgb,
                depth_image=depth_colored,
                detection_image=result_image,
            )

            # 儲存結果
            save_path = "outputs/detector_demo_with_camera.png"
            visualizer.save_image(comparison, save_path)
            logger.success(f"結果已儲存: {save_path}")

        print(f"\n  ✓ 完成")
        print(f"  結果: {save_path}")

    except Exception as e:
        logger.error(f"範例執行失敗: {e}")
        print(f"\n  ✗ 失敗: {e}")


def demo_detector_filtering():
    """結果過濾範例"""
    print("\n" + "=" * 70)
    print("  範例 3: 檢測結果過濾")
    print("=" * 70)

    logger = setup_logger(name="DetectorDemo", log_dir="outputs/logs")
    visualizer = get_visualizer()

    try:
        # 建立檢測器
        detector = YOLOv11Detector(task="detect")
        logger.info("載入 YOLOv11n 模型...")
        detector.load_model("yolo11n.pt", device="cpu")

        # 使用模擬相機
        with MockCamera(mode="objects") as camera:
            frame = camera.get_frame()

            # 執行檢測
            result = detector.detect(frame.rgb, conf_threshold=0.1)  # 降低閾值
            logger.info(f"原始檢測: {result.num_detections} 個物體")

            # 過濾低信心度結果
            filtered_result = result.filter_by_confidence(0.5)
            logger.info(f"過濾後 (conf >= 0.5): {filtered_result.num_detections} 個物體")

            # 視覺化比較
            original_image = visualizer.draw_detection_results(
                image=frame.rgb.copy(),
                boxes=[b.to_xyxy() for b in result.boxes],
                masks=None,
                labels=[f"{b.class_name} {b.confidence:.2f}" for b in result.boxes],
                scores=[b.confidence for b in result.boxes],
                class_ids=[b.class_id for b in result.boxes],
            )

            filtered_image = visualizer.draw_detection_results(
                image=frame.rgb.copy(),
                boxes=[b.to_xyxy() for b in filtered_result.boxes],
                masks=None,
                labels=[f"{b.class_name} {b.confidence:.2f}" for b in filtered_result.boxes],
                scores=[b.confidence for b in filtered_result.boxes],
                class_ids=[b.class_id for b in filtered_result.boxes],
            )

            # 水平拼接
            comparison = np.hstack([original_image, filtered_image])

            # 儲存結果
            save_path = "outputs/detector_demo_filtering.png"
            visualizer.save_image(comparison, save_path)
            logger.success(f"結果已儲存: {save_path}")

        print(f"\n  ✓ 完成")
        print(f"  結果: {save_path}")

    except Exception as e:
        logger.error(f"範例執行失敗: {e}")
        print(f"\n  ✗ 失敗: {e}")


def demo_detector_performance():
    """效能測試範例"""
    print("\n" + "=" * 70)
    print("  範例 4: 檢測效能測試")
    print("=" * 70)

    logger = setup_logger(name="DetectorDemo", log_dir="outputs/logs")

    try:
        # 建立檢測器
        detector = YOLOv11Detector(task="detect")
        logger.info("載入 YOLOv11n 模型...")
        detector.load_model("yolo11n.pt", device="cpu")

        # 使用模擬相機連續檢測
        with MockCamera(mode="objects") as camera:
            num_frames = 10
            total_time = 0.0
            total_detections = 0

            logger.info(f"執行 {num_frames} 次檢測...")

            for i in range(num_frames):
                frame = camera.get_frame()
                result = detector.detect(frame.rgb, conf_threshold=0.25)

                total_time += result.inference_time
                total_detections += result.num_detections

                if i % 3 == 0:
                    logger.info(f"  幀 {i+1}: {result.num_detections} 個物體, {result.inference_time:.3f}s")

            avg_time = total_time / num_frames
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            avg_detections = total_detections / num_frames

            logger.success(f"效能測試完成")
            logger.info(f"  總幀數: {num_frames}")
            logger.info(f"  平均推論時間: {avg_time:.3f}s")
            logger.info(f"  平均 FPS: {avg_fps:.2f}")
            logger.info(f"  平均檢測數: {avg_detections:.1f}")

        print(f"\n  ✓ 完成")
        print(f"  平均 FPS: {avg_fps:.2f}")

    except Exception as e:
        logger.error(f"範例執行失敗: {e}")
        print(f"\n  ✗ 失敗: {e}")


def main():
    """主函數"""
    print("\n" + "=" * 70)
    print("  AI 檢測器使用示範")
    print("=" * 70)

    # 檢查 YOLO 是否可用
    try:
        from ultralytics import YOLO

        yolo_available = True
    except ImportError:
        yolo_available = False

    if not yolo_available:
        print("\n✗ Ultralytics YOLO 未安裝")
        print("\n若要執行此示範,請先安裝:")
        print("  pip install ultralytics")
        return

    print("\n✓ Ultralytics YOLO 已安裝")

    # 執行所有範例
    demo_detector_basic()
    demo_detector_with_camera()
    demo_detector_filtering()
    demo_detector_performance()

    print("\n" + "=" * 70)
    print("  所有示範完成！")
    print("=" * 70)

    print("\n生成的檔案:")
    print("  - outputs/detector_demo_basic.png")
    print("  - outputs/detector_demo_with_camera.png")
    print("  - outputs/detector_demo_filtering.png")
    print("  - outputs/logs/DetectorDemo.log")


if __name__ == "__main__":
    main()
