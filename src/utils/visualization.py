"""
可視化模組
提供檢測結果的視覺化功能
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Visualizer:
    """
    視覺化工具
    提供檢測結果的繪製與儲存功能
    """

    def __init__(
        self,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        line_width: int = 2,
        alpha: float = 0.4,
    ):
        """
        初始化視覺化工具

        Args:
            font_scale: 字體大小
            font_thickness: 字體粗細
            line_width: 線條寬度
            alpha: 遮罩透明度
        """
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.line_width = line_width
        self.alpha = alpha
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # 預設顏色
        self.colors = self._generate_colors(100)

    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """
        生成不同的顏色

        Args:
            num_colors: 顏色數量

        Returns:
            BGR 顏色列表
        """
        np.random.seed(42)
        colors = []
        for _ in range(num_colors):
            colors.append(
                (
                    int(np.random.randint(0, 255)),
                    int(np.random.randint(0, 255)),
                    int(np.random.randint(0, 255)),
                )
            )
        return colors

    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str = "",
        conf: float = 0.0,
        color: Optional[Tuple[int, int, int]] = None,
        class_id: int = 0,
    ) -> np.ndarray:
        """
        繪製邊界框

        Args:
            image: 輸入影像 (BGR)
            bbox: 邊界框 (x1, y1, x2, y2)
            label: 類別標籤
            conf: 信心分數
            color: 顏色 (BGR)，None 則自動選擇
            class_id: 類別 ID

        Returns:
            繪製後的影像
        """
        x1, y1, x2, y2 = map(int, bbox)

        # 選擇顏色
        if color is None:
            color = self.colors[class_id % len(self.colors)]

        # 繪製邊界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_width)

        # 準備標籤文字
        if label and conf > 0:
            text = f"{label}: {conf:.2f}"
        elif label:
            text = label
        else:
            text = f"{conf:.2f}"

        # 計算文字大小
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )

        # 繪製文字背景
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1,
        )

        # 繪製文字
        cv2.putText(
            image,
            text,
            (x1, y1 - baseline - 2),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.font_thickness,
        )

        return image

    def draw_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Optional[Tuple[int, int, int]] = None,
        class_id: int = 0,
    ) -> np.ndarray:
        """
        繪製分割遮罩

        Args:
            image: 輸入影像 (BGR)
            mask: 二值遮罩 (H, W)
            color: 顏色 (BGR)，None 則自動選擇
            class_id: 類別 ID

        Returns:
            繪製後的影像
        """
        # 選擇顏色
        if color is None:
            color = self.colors[class_id % len(self.colors)]

        # 建立彩色遮罩
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color

        # 混合影像與遮罩
        image = cv2.addWeighted(image, 1.0, colored_mask, self.alpha, 0)

        return image

    def draw_keypoints(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 3,
    ) -> np.ndarray:
        """
        繪製關鍵點

        Args:
            image: 輸入影像 (BGR)
            keypoints: 關鍵點座標 (N, 2) 或 (N, 3)
            color: 顏色 (BGR)
            radius: 點的半徑

        Returns:
            繪製後的影像
        """
        for kpt in keypoints:
            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(image, (x, y), radius, color, -1)

        return image

    def draw_text(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 255, 255),
        background: bool = True,
    ) -> np.ndarray:
        """
        繪製文字

        Args:
            image: 輸入影像 (BGR)
            text: 文字內容
            position: 文字位置 (x, y)
            color: 文字顏色 (BGR)
            background: 是否繪製背景

        Returns:
            繪製後的影像
        """
        x, y = position

        # 計算文字大小
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )

        # 繪製背景
        if background:
            cv2.rectangle(
                image,
                (x, y - text_height - baseline - 5),
                (x + text_width, y),
                (0, 0, 0),
                -1,
            )

        # 繪製文字
        cv2.putText(
            image,
            text,
            (x, y - baseline - 2),
            self.font,
            self.font_scale,
            color,
            self.font_thickness,
        )

        return image

    def draw_detection_results(
        self,
        image: np.ndarray,
        boxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        scores: Optional[np.ndarray] = None,
        class_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        繪製完整的檢測結果

        Args:
            image: 輸入影像 (BGR)
            boxes: 邊界框 (N, 4) - (x1, y1, x2, y2)
            masks: 分割遮罩 (N, H, W)
            labels: 類別標籤列表
            scores: 信心分數 (N,)
            class_ids: 類別 ID (N,)

        Returns:
            繪製後的影像
        """
        result = image.copy()

        # 繪製遮罩
        if masks is not None:
            for i, mask in enumerate(masks):
                cid = class_ids[i] if class_ids is not None else 0
                result = self.draw_mask(result, mask, class_id=cid)

        # 繪製邊界框
        if boxes is not None:
            for i, box in enumerate(boxes):
                label = labels[i] if labels is not None else ""
                score = scores[i] if scores is not None else 0.0
                cid = class_ids[i] if class_ids is not None else 0
                result = self.draw_bbox(result, box, label, score, class_id=cid)

        return result

    def draw_depth_map(
        self, depth: np.ndarray, colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        繪製深度圖

        Args:
            depth: 深度圖 (H, W)
            colormap: OpenCV 色彩映射

        Returns:
            彩色深度圖 (BGR)
        """
        # 正規化到 0-255
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)

        # 套用色彩映射
        depth_colored = cv2.applyColorMap(depth_normalized, colormap)

        return depth_colored

    def create_comparison_view(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        detection_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        建立比較視圖（並排顯示）

        Args:
            rgb_image: RGB 影像
            depth_image: 深度圖
            detection_image: 檢測結果影像

        Returns:
            合併後的影像
        """
        images = [rgb_image, depth_image]
        if detection_image is not None:
            images.append(detection_image)

        # 確保所有影像高度相同
        height = max(img.shape[0] for img in images)
        resized_images = []

        for img in images:
            if img.shape[0] != height:
                scale = height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_width, height))
            resized_images.append(img)

        # 水平拼接
        combined = np.hstack(resized_images)

        return combined

    def save_image(
        self, image: np.ndarray, save_path: Union[str, Path], create_dir: bool = True
    ):
        """
        儲存影像

        Args:
            image: 影像資料
            save_path: 儲存路徑
            create_dir: 是否自動建立目錄
        """
        save_path = Path(save_path)

        if create_dir:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(save_path), image)

    def plot_metrics(
        self,
        metrics: Dict[str, float],
        title: str = "Metrics",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        繪製指標圖表

        Args:
            metrics: 指標字典
            title: 圖表標題
            save_path: 儲存路徑（可選）

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(metrics.keys())
        values = list(metrics.values())

        ax.bar(names, values)
        ax.set_title(title)
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


# 全域視覺化工具實例
_global_visualizer: Optional[Visualizer] = None


def get_visualizer(**kwargs) -> Visualizer:
    """
    取得全域視覺化工具實例

    Args:
        **kwargs: Visualizer 參數

    Returns:
        Visualizer 實例
    """
    global _global_visualizer

    if _global_visualizer is None:
        _global_visualizer = Visualizer(**kwargs)

    return _global_visualizer


if __name__ == "__main__":
    # 測試程式碼
    visualizer = Visualizer()

    # 建立測試影像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (200, 200, 200)

    # 測試繪製邊界框
    bbox = (50, 50, 200, 200)
    test_image = visualizer.draw_bbox(
        test_image, bbox, label="test_object", conf=0.95, class_id=0
    )

    # 測試繪製文字
    test_image = visualizer.draw_text(
        test_image, "Test Visualization", (250, 100)
    )

    # 測試深度圖
    depth = np.random.rand(480, 640) * 1000
    depth_colored = visualizer.draw_depth_map(depth)

    # 儲存測試結果
    visualizer.save_image(test_image, "outputs/test_bbox.png")
    visualizer.save_image(depth_colored, "outputs/test_depth.png")

    print("視覺化測試完成！")
    print(f"測試影像已儲存至 outputs/")
