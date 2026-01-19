"""
檢測器抽象介面
定義標準的物體檢測與分割介面
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from pathlib import Path


@dataclass
class DetectionBox:
    """檢測框資料類別"""

    x1: float  # 左上角 x 座標
    y1: float  # 左上角 y 座標
    x2: float  # 右下角 x 座標
    y2: float  # 右下角 y 座標
    confidence: float  # 信心度 (0-1)
    class_id: int  # 類別 ID
    class_name: str  # 類別名稱

    @property
    def width(self) -> float:
        """邊界框寬度"""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """邊界框高度"""
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        """邊界框中心點"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        """邊界框面積"""
        return self.width * self.height

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """轉換為 (x1, y1, x2, y2) 格式"""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """轉換為 (x, y, w, h) 格式 (COCO 格式)"""
        return (self.x1, self.y1, self.width, self.height)

    def to_cxcywh(self) -> Tuple[float, float, float, float]:
        """轉換為 (center_x, center_y, w, h) 格式 (YOLO 格式)"""
        cx, cy = self.center
        return (cx, cy, self.width, self.height)


@dataclass
class SegmentationMask:
    """分割遮罩資料類別"""

    mask: np.ndarray  # 二值化遮罩 (H, W) bool 或 (H, W) uint8
    confidence: float  # 信心度 (0-1)
    class_id: int  # 類別 ID
    class_name: str  # 類別名稱
    bbox: Optional[DetectionBox] = None  # 對應的邊界框

    @property
    def area(self) -> int:
        """遮罩面積 (像素數)"""
        return int(np.sum(self.mask > 0))

    @property
    def shape(self) -> Tuple[int, int]:
        """遮罩尺寸"""
        return self.mask.shape

    def get_contours(self) -> List[np.ndarray]:
        """取得輪廓"""
        import cv2

        # 確保是 uint8 格式
        mask_uint8 = self.mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def get_center(self) -> Tuple[float, float]:
        """取得遮罩中心點"""
        import cv2

        moments = cv2.moments(self.mask.astype(np.uint8))
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            return (cx, cy)
        return (0.0, 0.0)


@dataclass
class DetectionResult:
    """檢測結果資料類別"""

    boxes: List[DetectionBox] = field(default_factory=list)  # 檢測框列表
    masks: List[SegmentationMask] = field(default_factory=list)  # 分割遮罩列表
    inference_time: float = 0.0  # 推論時間 (秒)
    image_shape: Tuple[int, int] = (0, 0)  # 影像尺寸 (H, W)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 額外的元資料

    @property
    def num_detections(self) -> int:
        """檢測數量"""
        return len(self.boxes)

    @property
    def num_masks(self) -> int:
        """分割遮罩數量"""
        return len(self.masks)

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        """依信心度過濾結果"""
        filtered_boxes = [box for box in self.boxes if box.confidence >= threshold]
        filtered_masks = [mask for mask in self.masks if mask.confidence >= threshold]

        return DetectionResult(
            boxes=filtered_boxes,
            masks=filtered_masks,
            inference_time=self.inference_time,
            image_shape=self.image_shape,
            metadata=self.metadata,
        )

    def filter_by_class(self, class_ids: List[int]) -> "DetectionResult":
        """依類別過濾結果"""
        filtered_boxes = [box for box in self.boxes if box.class_id in class_ids]
        filtered_masks = [mask for mask in self.masks if mask.class_id in class_ids]

        return DetectionResult(
            boxes=filtered_boxes,
            masks=filtered_masks,
            inference_time=self.inference_time,
            image_shape=self.image_shape,
            metadata=self.metadata,
        )

    def get_classes(self) -> List[int]:
        """取得所有出現的類別 ID"""
        return list(set([box.class_id for box in self.boxes]))

    def get_class_counts(self) -> Dict[str, int]:
        """取得各類別的數量統計"""
        counts = {}
        for box in self.boxes:
            counts[box.class_name] = counts.get(box.class_name, 0) + 1
        return counts


class DetectorInterface(ABC):
    """
    檢測器抽象介面
    定義物體檢測與分割的標準介面
    """

    def __init__(self):
        self._is_loaded = False
        self.class_names: List[str] = []
        self.num_classes: int = 0

    @property
    def is_loaded(self) -> bool:
        """模型是否已載入"""
        return self._is_loaded

    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> bool:
        """
        載入模型

        Args:
            model_path: 模型路徑
            **kwargs: 其他參數

        Returns:
            是否成功載入
        """
        pass

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs,
    ) -> DetectionResult:
        """
        執行物體檢測

        Args:
            image: 輸入影像 (H, W, 3) BGR 格式
            conf_threshold: 信心度閾值
            iou_threshold: NMS IoU 閾值
            **kwargs: 其他參數

        Returns:
            檢測結果
        """
        pass

    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs,
    ) -> DetectionResult:
        """
        執行實例分割

        Args:
            image: 輸入影像 (H, W, 3) BGR 格式
            conf_threshold: 信心度閾值
            iou_threshold: NMS IoU 閾值
            **kwargs: 其他參數

        Returns:
            分割結果 (包含遮罩)
        """
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        影像前處理 (子類別可覆寫)

        Args:
            image: 輸入影像

        Returns:
            前處理後的影像
        """
        return image

    def postprocess(self, result: DetectionResult) -> DetectionResult:
        """
        結果後處理 (子類別可覆寫)

        Args:
            result: 原始結果

        Returns:
            後處理後的結果
        """
        return result

    def get_model_info(self) -> Dict[str, Any]:
        """
        取得模型資訊

        Returns:
            模型資訊字典
        """
        return {
            "is_loaded": self.is_loaded,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }


# 異常類別
class DetectorException(Exception):
    """檢測器基礎異常"""

    pass


class ModelLoadError(DetectorException):
    """模型載入錯誤"""

    pass


class InferenceError(DetectorException):
    """推論錯誤"""

    pass


class PreprocessError(DetectorException):
    """前處理錯誤"""

    pass


class PostprocessError(DetectorException):
    """後處理錯誤"""

    pass
