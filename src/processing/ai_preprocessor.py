"""
AI 模型前處理器
為 YOLO 等模型準備輸入影像
"""

from typing import Tuple, Optional
import numpy as np
import cv2


class AIPreprocessor:
    """
    AI 模型前處理器
    處理影像以符合模型輸入要求
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        normalize: bool = True,
        keep_aspect: bool = True,
    ):
        """
        初始化前處理器

        Args:
            target_size: 目標尺寸 (width, height)
            normalize: 是否正規化到 0-1
            keep_aspect: 是否保持長寬比
        """
        self.target_size = target_size
        self.normalize = normalize
        self.keep_aspect = keep_aspect

    def preprocess(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        前處理影像

        Args:
            image: 輸入影像 (H, W, 3) BGR

        Returns:
            (處理後影像, 變換資訊)
        """
        original_shape = image.shape[:2]

        # Resize
        if self.keep_aspect:
            processed, transform_info = self._resize_keep_aspect(image)
        else:
            processed, transform_info = self._resize_direct(image)

        # 正規化
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0

        transform_info["original_shape"] = original_shape
        transform_info["processed_shape"] = processed.shape[:2]
        transform_info["normalized"] = self.normalize

        return processed, transform_info

    def _resize_keep_aspect(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        保持長寬比的 resize + padding

        Args:
            image: 輸入影像

        Returns:
            (處理後影像, 變換資訊)
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size

        # 計算縮放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),  # 灰色 padding
        )

        transform_info = {
            "scale": scale,
            "new_size": (new_w, new_h),
            "padding": (pad_left, pad_top, pad_right, pad_bottom),
        }

        return padded, transform_info

    def _resize_direct(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        直接 resize (可能變形)

        Args:
            image: 輸入影像

        Returns:
            (處理後影像, 變換資訊)
        """
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

        transform_info = {
            "scale_x": self.target_size[0] / image.shape[1],
            "scale_y": self.target_size[1] / image.shape[0],
        }

        return resized, transform_info

    def postprocess_boxes(
        self,
        boxes: np.ndarray,
        transform_info: dict,
    ) -> np.ndarray:
        """
        將檢測框座標轉回原始影像尺寸

        Args:
            boxes: 檢測框 (N, 4) [x1, y1, x2, y2]
            transform_info: 變換資訊

        Returns:
            原始尺寸的檢測框
        """
        if len(boxes) == 0:
            return boxes

        boxes = boxes.copy()

        if self.keep_aspect:
            # 移除 padding
            pad_left, pad_top, _, _ = transform_info["padding"]
            boxes[:, [0, 2]] -= pad_left
            boxes[:, [1, 3]] -= pad_top

            # 反向縮放
            scale = transform_info["scale"]
            boxes = boxes / scale

        else:
            # 反向縮放
            scale_x = transform_info["scale_x"]
            scale_y = transform_info["scale_y"]
            boxes[:, [0, 2]] /= scale_x
            boxes[:, [1, 3]] /= scale_y

        return boxes

    def postprocess_mask(
        self,
        mask: np.ndarray,
        transform_info: dict,
    ) -> np.ndarray:
        """
        將分割遮罩轉回原始影像尺寸

        Args:
            mask: 分割遮罩 (H, W)
            transform_info: 變換資訊

        Returns:
            原始尺寸的遮罩
        """
        original_shape = transform_info["original_shape"]

        if self.keep_aspect:
            # 移除 padding
            pad_left, pad_top, pad_right, pad_bottom = transform_info["padding"]
            h, w = mask.shape
            mask = mask[pad_top : h - pad_bottom, pad_left : w - pad_right]

        # Resize 回原始尺寸
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        return mask_resized

    def prepare_for_yolo(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        準備給 YOLO 的輸入

        Args:
            image: 輸入影像 (H, W, 3) BGR

        Returns:
            (處理後影像, 變換資訊)
        """
        # YOLO 需要 RGB 格式
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 前處理
        processed, transform_info = self.preprocess(rgb)

        return processed, transform_info

    def denormalize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        反正規化

        Args:
            image: 正規化影像 (0-1)

        Returns:
            反正規化影像 (0-255)
        """
        if image.max() <= 1.0:
            return (image * 255).astype(np.uint8)
        return image.astype(np.uint8)


if __name__ == "__main__":
    # 測試 AI 前處理器
    print("AI 前處理器測試\n")

    # 建立測試影像
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    print(f"原始影像: {test_image.shape}")

    # 建立前處理器
    preprocessor = AIPreprocessor(
        target_size=(640, 640),
        normalize=True,
        keep_aspect=True,
    )

    print(f"\n前處理器配置:")
    print(f"  目標尺寸: {preprocessor.target_size}")
    print(f"  保持長寬比: {preprocessor.keep_aspect}")
    print(f"  正規化: {preprocessor.normalize}")

    # 前處理
    print(f"\n執行前處理...")
    processed, transform_info = preprocessor.preprocess(test_image)

    print(f"\n處理結果:")
    print(f"  處理後尺寸: {processed.shape}")
    print(f"  資料類型: {processed.dtype}")
    print(f"  數值範圍: {processed.min():.3f} - {processed.max():.3f}")

    print(f"\n變換資訊:")
    for key, value in transform_info.items():
        print(f"  {key}: {value}")

    # 測試後處理
    print(f"\n測試後處理...")
    # 模擬檢測框
    test_boxes = np.array([[100, 100, 300, 300], [400, 200, 500, 400]])
    original_boxes = preprocessor.postprocess_boxes(test_boxes, transform_info)

    print(f"  處理後座標: {test_boxes[0]}")
    print(f"  原始座標: {original_boxes[0]}")

    # 測試 YOLO 準備
    print(f"\n測試 YOLO 準備...")
    yolo_input, yolo_info = preprocessor.prepare_for_yolo(test_image)
    print(f"  YOLO 輸入: {yolo_input.shape}")
    print(f"  格式: {'RGB' if yolo_info else 'BGR'}")

    print(f"\n✓ AI 前處理器測試完成")
