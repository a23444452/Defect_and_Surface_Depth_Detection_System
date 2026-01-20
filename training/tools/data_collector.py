#!/usr/bin/env python3
"""
資料收集工具
用於收集與管理訓練資料
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import json
import shutil

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2

from src.hardware import CameraInterface
from src.utils import setup_logger

# 設置日誌
logger = logging.getLogger(__name__)


class DataCollector:
    """
    資料收集器
    收集相機影像與深度資料用於訓練
    """

    def __init__(
        self,
        output_dir: Path,
        dataset_name: str = "industrial_inspection",
        create_time: Optional[datetime] = None,
    ):
        """
        初始化資料收集器

        Args:
            output_dir: 輸出目錄
            dataset_name: 資料集名稱
            create_time: 建立時間 (None 則使用當前時間)
        """
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name

        # 建立時間戳記
        if create_time is None:
            create_time = datetime.now()
        self.create_time = create_time
        timestamp = create_time.strftime("%Y%m%d_%H%M%S")

        # 建立資料集目錄
        self.dataset_dir = self.output_dir / f"{dataset_name}_{timestamp}"
        self.images_dir = self.dataset_dir / "images"
        self.depths_dir = self.dataset_dir / "depths"
        self.labels_dir = self.dataset_dir / "labels"

        # 建立目錄
        for d in [self.images_dir, self.depths_dir, self.labels_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 統計資訊
        self.sample_count = 0

        # 元資料
        self.metadata = {
            "dataset_name": dataset_name,
            "create_time": create_time.isoformat(),
            "samples": [],
        }

        logger.info(f"資料收集器已初始化")
        logger.info(f"  輸出目錄: {self.dataset_dir}")

    # ==================== 資料收集 ====================

    def collect_sample(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        label: Optional[Dict[str, Any]] = None,
        sample_id: Optional[str] = None,
    ) -> str:
        """
        收集單個樣本

        Args:
            rgb: RGB 影像 (H, W, 3)
            depth: 深度影像 (H, W)
            label: 標籤資訊 (可選)
            sample_id: 樣本 ID (None 則自動生成)

        Returns:
            樣本 ID
        """
        # 生成樣本 ID
        if sample_id is None:
            sample_id = f"{self.sample_count:06d}"

        # 儲存影像
        rgb_path = self.images_dir / f"{sample_id}.jpg"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # 儲存深度 (NPY 格式保持精度)
        depth_path = self.depths_dir / f"{sample_id}.npy"
        np.save(depth_path, depth)

        # 儲存深度視覺化 (方便檢視)
        depth_vis_path = self.depths_dir / f"{sample_id}_vis.png"
        depth_vis = self._visualize_depth(depth)
        cv2.imwrite(str(depth_vis_path), depth_vis)

        # 儲存標籤
        if label is not None:
            label_path = self.labels_dir / f"{sample_id}.json"
            with open(label_path, "w") as f:
                json.dump(label, f, indent=2)

        # 更新元資料
        sample_metadata = {
            "sample_id": sample_id,
            "timestamp": datetime.now().isoformat(),
            "rgb_path": str(rgb_path.relative_to(self.dataset_dir)),
            "depth_path": str(depth_path.relative_to(self.dataset_dir)),
            "label_path": (
                str((self.labels_dir / f"{sample_id}.json").relative_to(self.dataset_dir))
                if label
                else None
            ),
            "rgb_shape": rgb.shape,
            "depth_shape": depth.shape,
            "has_label": label is not None,
        }

        self.metadata["samples"].append(sample_metadata)
        self.sample_count += 1

        logger.info(f"✓ 收集樣本 {sample_id}")

        return sample_id

    def _visualize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        視覺化深度影像

        Args:
            depth: 深度影像

        Returns:
            視覺化影像 (BGR)
        """
        # 正規化到 0-255
        valid_mask = depth > 0
        if not valid_mask.any():
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        depth_normalized = np.zeros_like(depth)
        depth_normalized[valid_mask] = depth[valid_mask]

        # 使用 JET colormap
        depth_normalized = cv2.normalize(
            depth_normalized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # 將無效區域設為黑色
        depth_colored[~valid_mask] = 0

        return depth_colored

    # ==================== 批次收集 ====================

    def collect_from_camera(
        self,
        camera: CameraInterface,
        num_samples: int = 100,
        interval: float = 1.0,
        auto_label: bool = False,
    ):
        """
        從相機批次收集資料

        Args:
            camera: 相機介面
            num_samples: 收集樣本數
            interval: 採樣間隔 (秒)
            auto_label: 是否自動標註 (需要 AI 模型)
        """
        import time

        logger.info(f"開始從相機收集 {num_samples} 個樣本 (間隔: {interval}s)")

        with camera:
            for i in range(num_samples):
                try:
                    # 擷取影像
                    frame = camera.get_frame()

                    # 自動標註 (如果啟用)
                    label = None
                    if auto_label:
                        label = self._auto_label(frame)

                    # 收集樣本
                    sample_id = self.collect_sample(frame.rgb, frame.depth, label)

                    # 顯示進度
                    if (i + 1) % 10 == 0:
                        logger.info(f"  進度: {i+1}/{num_samples}")

                    # 等待間隔
                    if i < num_samples - 1:
                        time.sleep(interval)

                except KeyboardInterrupt:
                    logger.info("\n收集已中斷")
                    break

                except Exception as e:
                    logger.error(f"收集樣本 {i} 失敗: {e}")
                    continue

        logger.info(f"✓ 收集完成: {self.sample_count} 個樣本")

    def _auto_label(self, frame) -> Dict[str, Any]:
        """
        自動標註 (需要實作 AI 模型推理)

        Args:
            frame: RGBDFrame

        Returns:
            標籤字典
        """
        # TODO: 實作自動標註邏輯
        # 這裡可以使用預訓練模型進行初步標註
        return {
            "auto_labeled": True,
            "needs_verification": True,
            "objects": [],
            "defects": [],
        }

    # ==================== 資料統計 ====================

    def get_statistics(self) -> Dict[str, Any]:
        """
        取得資料集統計

        Returns:
            統計字典
        """
        stats = {
            "total_samples": self.sample_count,
            "labeled_samples": sum(
                1 for s in self.metadata["samples"] if s["has_label"]
            ),
            "unlabeled_samples": sum(
                1 for s in self.metadata["samples"] if not s["has_label"]
            ),
            "dataset_dir": str(self.dataset_dir),
            "create_time": self.metadata["create_time"],
        }

        # 計算資料大小
        total_size = 0
        for d in [self.images_dir, self.depths_dir, self.labels_dir]:
            for f in d.glob("**/*"):
                if f.is_file():
                    total_size += f.stat().st_size

        stats["total_size_mb"] = total_size / (1024**2)

        return stats

    def print_statistics(self):
        """列印統計資訊"""
        stats = self.get_statistics()

        print("\n資料集統計")
        print("=" * 60)
        print(f"資料集名稱: {self.dataset_name}")
        print(f"建立時間: {stats['create_time']}")
        print(f"資料集目錄: {stats['dataset_dir']}")
        print(f"總樣本數: {stats['total_samples']}")
        print(f"  已標註: {stats['labeled_samples']}")
        print(f"  未標註: {stats['unlabeled_samples']}")
        print(f"資料大小: {stats['total_size_mb']:.2f} MB")

    # ==================== 儲存與載入 ====================

    def save_metadata(self):
        """儲存元資料"""
        metadata_path = self.dataset_dir / "metadata.json"

        # 更新統計
        self.metadata["statistics"] = self.get_statistics()

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"✓ 元資料已儲存: {metadata_path}")

    @staticmethod
    def load_dataset(dataset_dir: Path) -> Dict[str, Any]:
        """
        載入資料集

        Args:
            dataset_dir: 資料集目錄

        Returns:
            元資料字典
        """
        metadata_path = Path(dataset_dir) / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"找不到元資料: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        logger.info(f"✓ 載入資料集: {dataset_dir}")
        logger.info(f"  樣本數: {len(metadata['samples'])}")

        return metadata

    # ==================== 資料分割 ====================

    def split_dataset(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1
    ):
        """
        分割資料集為訓練/驗證/測試集

        Args:
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例
            test_ratio: 測試集比例
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例總和必須為 1"

        samples = self.metadata["samples"]
        n = len(samples)

        # 隨機打亂
        import random

        random.shuffle(samples)

        # 分割
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            "train": samples[:train_end],
            "val": samples[train_end:val_end],
            "test": samples[val_end:],
        }

        # 儲存分割資訊
        split_path = self.dataset_dir / "splits.json"
        with open(split_path, "w") as f:
            json.dump(splits, f, indent=2)

        logger.info(f"✓ 資料集已分割:")
        logger.info(f"  訓練集: {len(splits['train'])} ({train_ratio*100:.1f}%)")
        logger.info(f"  驗證集: {len(splits['val'])} ({val_ratio*100:.1f}%)")
        logger.info(f"  測試集: {len(splits['test'])} ({test_ratio*100:.1f}%)")


def main():
    """示範使用"""
    logger.info("資料收集工具示範")

    # 建立資料收集器
    collector = DataCollector(
        output_dir=Path("outputs/datasets"), dataset_name="demo_dataset"
    )

    # 收集模擬資料
    logger.info("\n收集模擬資料...")
    for i in range(10):
        # 生成模擬影像
        rgb = np.random.randint(0, 255, (800, 1280, 3), dtype=np.uint8)
        depth = np.random.randint(800, 1500, (800, 1280), dtype=np.uint16).astype(
            np.float32
        )

        # 模擬標籤
        label = {
            "objects": [
                {
                    "class": "screw",
                    "bbox": [100, 100, 200, 200],
                    "confidence": 0.95,
                }
            ],
            "defects": [],
        }

        # 收集樣本
        collector.collect_sample(rgb, depth, label)

    # 顯示統計
    collector.print_statistics()

    # 儲存元資料
    collector.save_metadata()

    # 分割資料集
    collector.split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    logger.info("\n✓ 資料收集示範完成")


if __name__ == "__main__":
    setup_logger(name=__name__, log_dir="outputs/logs")
    main()
