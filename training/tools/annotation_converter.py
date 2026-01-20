#!/usr/bin/env python3
"""
標註格式轉換工具
支援多種標註格式之間的轉換
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class AnnotationConverter:
    """
    標註格式轉換器
    支援 COCO, YOLO, Pascal VOC 等格式
    """

    def __init__(self):
        """初始化轉換器"""
        self.supported_formats = ["coco", "yolo", "voc", "custom"]

    # ==================== COCO 格式 ====================

    def to_coco(
        self,
        annotations: List[Dict[str, Any]],
        images: List[Dict[str, Any]],
        categories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        轉換為 COCO 格式

        Args:
            annotations: 標註列表
            images: 影像列表
            categories: 類別列表

        Returns:
            COCO 格式字典
        """
        coco_format = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        return coco_format

    def from_coco(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        從 COCO 格式轉換

        Args:
            coco_data: COCO 格式資料

        Returns:
            解析後的資料
        """
        return {
            "images": coco_data.get("images", []),
            "annotations": coco_data.get("annotations", []),
            "categories": coco_data.get("categories", []),
        }

    # ==================== YOLO 格式 ====================

    def to_yolo(
        self,
        annotation: Dict[str, Any],
        image_width: int,
        image_height: int,
        class_names: List[str],
    ) -> List[str]:
        """
        轉換為 YOLO 格式

        Args:
            annotation: 標註資料
            image_width: 影像寬度
            image_height: 影像高度
            class_names: 類別名稱列表

        Returns:
            YOLO 格式行列表
        """
        yolo_lines = []

        for obj in annotation.get("objects", []):
            # 取得 bbox [x, y, w, h]
            bbox = obj.get("bbox", [])
            if len(bbox) != 4:
                continue

            x, y, w, h = bbox

            # 轉換為 YOLO 格式 (中心點 + 正規化)
            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            width = w / image_width
            height = h / image_height

            # 類別 ID
            class_name = obj.get("class", "")
            if class_name in class_names:
                class_id = class_names.index(class_name)
            else:
                logger.warning(f"未知類別: {class_name}")
                continue

            # YOLO 格式: class_id x_center y_center width height
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)

        return yolo_lines

    def from_yolo(
        self,
        yolo_lines: List[str],
        image_width: int,
        image_height: int,
        class_names: List[str],
    ) -> Dict[str, Any]:
        """
        從 YOLO 格式轉換

        Args:
            yolo_lines: YOLO 格式行列表
            image_width: 影像寬度
            image_height: 影像高度
            class_names: 類別名稱列表

        Returns:
            標註字典
        """
        objects = []

        for line in yolo_lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 轉換為絕對座標
            x = (x_center - width / 2) * image_width
            y = (y_center - height / 2) * image_height
            w = width * image_width
            h = height * image_height

            # 類別名稱
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

            objects.append(
                {
                    "class": class_name,
                    "class_id": class_id,
                    "bbox": [x, y, w, h],
                    "confidence": 1.0,
                }
            )

        return {"objects": objects}

    # ==================== Pascal VOC 格式 ====================

    def to_voc_xml(
        self,
        annotation: Dict[str, Any],
        image_path: str,
        image_width: int,
        image_height: int,
    ) -> str:
        """
        轉換為 Pascal VOC XML 格式

        Args:
            annotation: 標註資料
            image_path: 影像路徑
            image_width: 影像寬度
            image_height: 影像高度

        Returns:
            XML 字串
        """
        from xml.etree import ElementTree as ET

        # 建立根節點
        root = ET.Element("annotation")

        # 檔案資訊
        ET.SubElement(root, "folder").text = Path(image_path).parent.name
        ET.SubElement(root, "filename").text = Path(image_path).name

        # 影像尺寸
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(image_width)
        ET.SubElement(size, "height").text = str(image_height)
        ET.SubElement(size, "depth").text = "3"

        # 物件
        for obj in annotation.get("objects", []):
            object_node = ET.SubElement(root, "object")
            ET.SubElement(object_node, "name").text = obj.get("class", "unknown")
            ET.SubElement(object_node, "difficult").text = "0"

            # Bounding box
            bbox = obj.get("bbox", [0, 0, 0, 0])
            bndbox = ET.SubElement(object_node, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(bbox[0]))
            ET.SubElement(bndbox, "ymin").text = str(int(bbox[1]))
            ET.SubElement(bndbox, "xmax").text = str(int(bbox[0] + bbox[2]))
            ET.SubElement(bndbox, "ymax").text = str(int(bbox[1] + bbox[3]))

        # 轉換為字串
        tree = ET.ElementTree(root)
        import io

        f = io.BytesIO()
        tree.write(f, encoding="utf-8", xml_declaration=True)
        return f.getvalue().decode("utf-8")

    # ==================== 自訂格式 ====================

    def to_custom_format(
        self, annotation: Dict[str, Any], include_depth: bool = True
    ) -> Dict[str, Any]:
        """
        轉換為自訂格式 (工業檢測專用)

        Args:
            annotation: 標註資料
            include_depth: 是否包含深度資訊

        Returns:
            自訂格式字典
        """
        custom = {
            "version": "1.0",
            "objects": [],
            "defects": [],
            "measurements": {},
        }

        # 物件
        for obj in annotation.get("objects", []):
            custom["objects"].append(
                {
                    "class": obj.get("class"),
                    "bbox": obj.get("bbox"),
                    "confidence": obj.get("confidence", 1.0),
                    "position_3d": obj.get("position_3d"),  # [x, y, z]
                    "rotation": obj.get("rotation"),  # 旋轉矩陣或四元數
                }
            )

        # 缺陷
        for defect in annotation.get("defects", []):
            custom["defects"].append(
                {
                    "type": defect.get("type"),  # dent, bump, crack, etc.
                    "severity": defect.get("severity"),  # minor, moderate, critical
                    "location": defect.get("location"),  # [x, y, z]
                    "depth": defect.get("depth"),  # mm
                    "area": defect.get("area"),  # mm²
                    "bbox": defect.get("bbox"),  # 2D bounding box
                }
            )

        # 量測資訊
        if "measurements" in annotation:
            custom["measurements"] = annotation["measurements"]

        return custom

    # ==================== 批次轉換 ====================

    def convert_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        input_format: str,
        output_format: str,
        class_names: Optional[List[str]] = None,
    ):
        """
        批次轉換資料集

        Args:
            input_dir: 輸入目錄
            output_dir: 輸出目錄
            input_format: 輸入格式
            output_format: 輸出格式
            class_names: 類別名稱列表 (YOLO 需要)
        """
        logger.info(f"轉換資料集: {input_format} -> {output_format}")
        logger.info(f"  輸入: {input_dir}")
        logger.info(f"  輸出: {output_dir}")

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 根據格式處理
        if input_format == "yolo" and output_format == "coco":
            self._convert_yolo_to_coco(input_dir, output_dir, class_names)
        elif input_format == "coco" and output_format == "yolo":
            self._convert_coco_to_yolo(input_dir, output_dir, class_names)
        else:
            logger.warning(f"不支援的轉換: {input_format} -> {output_format}")

    def _convert_yolo_to_coco(
        self, input_dir: Path, output_dir: Path, class_names: List[str]
    ):
        """YOLO -> COCO"""
        # TODO: 實作完整轉換邏輯
        logger.info("YOLO -> COCO 轉換 (待實作)")

    def _convert_coco_to_yolo(
        self, input_dir: Path, output_dir: Path, class_names: List[str]
    ):
        """COCO -> YOLO"""
        # TODO: 實作完整轉換邏輯
        logger.info("COCO -> YOLO 轉換 (待實作)")

    # ==================== 工具方法 ====================

    def visualize_annotation(
        self, image: np.ndarray, annotation: Dict[str, Any], class_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        視覺化標註

        Args:
            image: 影像 (RGB)
            annotation: 標註資料
            class_names: 類別名稱列表

        Returns:
            標註後的影像
        """
        vis_image = image.copy()

        # 繪製物件
        for obj in annotation.get("objects", []):
            bbox = obj.get("bbox", [])
            if len(bbox) != 4:
                continue

            x, y, w, h = map(int, bbox)
            class_name = obj.get("class", "unknown")
            confidence = obj.get("confidence", 1.0)

            # 繪製矩形
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 繪製標籤
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                vis_image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # 繪製缺陷
        for defect in annotation.get("defects", []):
            bbox = defect.get("bbox", [])
            if len(bbox) == 4:
                x, y, w, h = map(int, bbox)
                defect_type = defect.get("type", "unknown")
                severity = defect.get("severity", "unknown")

                # 根據嚴重程度選擇顏色
                color = (0, 0, 255) if severity == "critical" else (0, 165, 255)

                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

                label = f"{defect_type} ({severity})"
                cv2.putText(
                    vis_image,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        return vis_image


def main():
    """示範使用"""
    logger.info("標註格式轉換工具示範")

    converter = AnnotationConverter()

    # 測試標註資料
    annotation = {
        "objects": [
            {
                "class": "screw",
                "bbox": [100, 100, 50, 50],
                "confidence": 0.95,
            },
            {
                "class": "nut",
                "bbox": [200, 200, 40, 40],
                "confidence": 0.92,
            },
        ],
        "defects": [
            {
                "type": "dent",
                "severity": "moderate",
                "bbox": [150, 150, 30, 30],
                "depth": 0.8,
                "area": 120.0,
            }
        ],
    }

    # 轉換為 YOLO 格式
    logger.info("\n轉換為 YOLO 格式:")
    yolo_lines = converter.to_yolo(annotation, 640, 480, ["screw", "nut"])
    for line in yolo_lines:
        print(f"  {line}")

    # 轉換為 Pascal VOC 格式
    logger.info("\n轉換為 Pascal VOC 格式:")
    voc_xml = converter.to_voc_xml(annotation, "image.jpg", 640, 480)
    print(voc_xml[:200] + "...")

    # 轉換為自訂格式
    logger.info("\n轉換為自訂格式:")
    custom = converter.to_custom_format(annotation)
    print(json.dumps(custom, indent=2))

    # 視覺化
    logger.info("\n視覺化標註:")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    vis_image = converter.visualize_annotation(test_image, annotation)
    logger.info(f"  視覺化影像形狀: {vis_image.shape}")

    logger.info("\n✓ 標註格式轉換示範完成")


if __name__ == "__main__":
    from src.utils import setup_logger

    setup_logger(name=__name__, log_dir="outputs/logs")
    main()
