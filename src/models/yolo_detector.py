"""
YOLOv11 檢測器實作
基於 Ultralytics YOLO 框架
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

from .detector_interface import (
    DetectorInterface,
    DetectionBox,
    SegmentationMask,
    DetectionResult,
    ModelLoadError,
    InferenceError,
)


class YOLOv11Detector(DetectorInterface):
    """
    YOLOv11 檢測器
    支援物體檢測與實例分割
    """

    def __init__(self, task: str = "detect"):
        """
        初始化 YOLOv11 檢測器

        Args:
            task: 任務類型
                - "detect": 物體檢測
                - "segment": 實例分割
        """
        super().__init__()

        self.task = task
        self.model = None
        self.device = "cpu"
        self.model_path = None

        # 檢查 Ultralytics 是否可用
        try:
            from ultralytics import YOLO

            self.YOLO = YOLO
            self.yolo_available = True
        except ImportError:
            self.YOLO = None
            self.yolo_available = False

    def load_model(
        self,
        model_path: str,
        device: str = "cpu",
        verbose: bool = True,
        **kwargs,
    ) -> bool:
        """
        載入 YOLOv11 模型

        Args:
            model_path: 模型路徑或模型名稱
                - 檔案路徑: "models/yolo11n.pt"
                - 預訓練模型: "yolo11n.pt", "yolo11s.pt", etc.
            device: 執行裝置 ("cpu", "cuda", "mps")
            verbose: 是否顯示詳細資訊
            **kwargs: 其他參數

        Returns:
            是否成功載入
        """
        if not self.yolo_available:
            raise ModelLoadError("Ultralytics YOLO 未安裝。請執行: pip install ultralytics")

        try:
            # 載入模型
            self.model = self.YOLO(model_path)
            self.model_path = model_path
            self.device = device

            # 設定裝置
            if device != "cpu":
                self.model.to(device)

            # 取得類別資訊
            if hasattr(self.model, "names"):
                self.class_names = list(self.model.names.values())
                self.num_classes = len(self.class_names)

            self._is_loaded = True

            if verbose:
                print(f"✓ YOLOv11 模型已載入: {model_path}")
                print(f"  裝置: {device}")
                print(f"  類別數: {self.num_classes}")
                print(f"  任務: {self.task}")

            return True

        except Exception as e:
            raise ModelLoadError(f"模型載入失敗: {e}")

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        verbose: bool = False,
        **kwargs,
    ) -> DetectionResult:
        """
        執行物體檢測

        Args:
            image: 輸入影像 (H, W, 3) BGR 格式
            conf_threshold: 信心度閾值
            iou_threshold: NMS IoU 閾值
            max_det: 最大檢測數量
            verbose: 是否顯示詳細資訊
            **kwargs: 其他參數

        Returns:
            檢測結果
        """
        if not self.is_loaded:
            raise InferenceError("模型尚未載入")

        try:
            # 記錄開始時間
            start_time = time.time()

            # 執行推論
            results = self.model.predict(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                verbose=verbose,
                device=self.device,
                **kwargs,
            )

            # 計算推論時間
            inference_time = time.time() - start_time

            # 解析結果
            detection_result = self._parse_detection_results(
                results=results,
                image_shape=image.shape[:2],
                inference_time=inference_time,
            )

            return detection_result

        except Exception as e:
            raise InferenceError(f"檢測推論失敗: {e}")

    def segment(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        verbose: bool = False,
        **kwargs,
    ) -> DetectionResult:
        """
        執行實例分割

        Args:
            image: 輸入影像 (H, W, 3) BGR 格式
            conf_threshold: 信心度閾值
            iou_threshold: NMS IoU 閾值
            max_det: 最大檢測數量
            verbose: 是否顯示詳細資訊
            **kwargs: 其他參數

        Returns:
            分割結果 (包含遮罩)
        """
        if not self.is_loaded:
            raise InferenceError("模型尚未載入")

        try:
            # 記錄開始時間
            start_time = time.time()

            # 執行推論
            results = self.model.predict(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                verbose=verbose,
                device=self.device,
                **kwargs,
            )

            # 計算推論時間
            inference_time = time.time() - start_time

            # 解析結果
            detection_result = self._parse_segmentation_results(
                results=results,
                image_shape=image.shape[:2],
                inference_time=inference_time,
            )

            return detection_result

        except Exception as e:
            raise InferenceError(f"分割推論失敗: {e}")

    def _parse_detection_results(
        self,
        results: Any,
        image_shape: tuple,
        inference_time: float,
    ) -> DetectionResult:
        """
        解析檢測結果

        Args:
            results: YOLO 原始結果
            image_shape: 影像尺寸
            inference_time: 推論時間

        Returns:
            DetectionResult
        """
        boxes = []

        # YOLO 結果通常是列表,取第一個
        result = results[0] if isinstance(results, list) else results

        # 解析邊界框
        if hasattr(result, "boxes") and result.boxes is not None:
            for box in result.boxes:
                # 取得座標 (xyxy 格式)
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                # 建立 DetectionBox
                detection_box = DetectionBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                )

                boxes.append(detection_box)

        return DetectionResult(
            boxes=boxes,
            masks=[],
            inference_time=inference_time,
            image_shape=image_shape,
            metadata={"model": self.model_path, "task": "detect"},
        )

    def _parse_segmentation_results(
        self,
        results: Any,
        image_shape: tuple,
        inference_time: float,
    ) -> DetectionResult:
        """
        解析分割結果

        Args:
            results: YOLO 原始結果
            image_shape: 影像尺寸
            inference_time: 推論時間

        Returns:
            DetectionResult
        """
        boxes = []
        masks = []

        # YOLO 結果通常是列表,取第一個
        result = results[0] if isinstance(results, list) else results

        # 解析邊界框
        if hasattr(result, "boxes") and result.boxes is not None:
            for i, box in enumerate(result.boxes):
                # 取得座標 (xyxy 格式)
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                # 建立 DetectionBox
                detection_box = DetectionBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                )

                boxes.append(detection_box)

                # 解析遮罩
                if hasattr(result, "masks") and result.masks is not None:
                    mask_data = result.masks.data[i].cpu().numpy()

                    # 調整遮罩大小至原始影像尺寸
                    import cv2

                    mask_resized = cv2.resize(
                        mask_data,
                        (image_shape[1], image_shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    # 二值化
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)

                    # 建立 SegmentationMask
                    seg_mask = SegmentationMask(
                        mask=mask_binary,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                        bbox=detection_box,
                    )

                    masks.append(seg_mask)

        return DetectionResult(
            boxes=boxes,
            masks=masks,
            inference_time=inference_time,
            image_shape=image_shape,
            metadata={"model": self.model_path, "task": "segment"},
        )

    def get_model_info(self) -> Dict[str, Any]:
        """取得模型資訊"""
        info = super().get_model_info()
        info.update(
            {
                "model_path": self.model_path,
                "task": self.task,
                "device": self.device,
                "yolo_available": self.yolo_available,
            }
        )
        return info


if __name__ == "__main__":
    # 測試 YOLOv11 檢測器
    print("YOLOv11 檢測器測試\n")

    # 建立檢測器
    detector = YOLOv11Detector(task="detect")

    print(f"YOLO 可用: {detector.yolo_available}")

    if detector.yolo_available:
        print("\n✓ Ultralytics YOLO 已安裝")
        print("  可以使用 load_model() 載入模型進行測試")
    else:
        print("\n✗ Ultralytics YOLO 未安裝")
        print("  請執行: pip install ultralytics")
