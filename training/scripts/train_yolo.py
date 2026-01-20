#!/usr/bin/env python3
"""
YOLO 模型訓練腳本
支援 YOLOv11 訓練與微調
"""

import sys
from pathlib import Path
from typing import Optional
import logging
import yaml
import argparse

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import setup_logger

logger = logging.getLogger(__name__)


def train_yolo(config_path: Path, resume: bool = False, weights: Optional[Path] = None):
    """
    訓練 YOLO 模型

    Args:
        config_path: 訓練配置檔案路徑
        resume: 是否從檢查點恢復
        weights: 預訓練權重路徑
    """
    logger.info("=" * 70)
    logger.info("  YOLO 模型訓練")
    logger.info("=" * 70)

    # 載入配置
    logger.info(f"\n載入配置: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 顯示配置
    logger.info("\n訓練配置:")
    logger.info(f"  資料集: {config['dataset']['name']}")
    logger.info(f"  模型: {config['model']['name']}")
    logger.info(f"  類別數: {config['model']['num_classes']}")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  Batch Size: {config['training']['batch_size']}")
    logger.info(f"  裝置: {config['training']['device']}")

    try:
        # 嘗試導入 ultralytics
        from ultralytics import YOLO

        logger.info("\n✓ ultralytics 已安裝")

        # 建立或載入模型
        if resume:
            logger.info("\n從檢查點恢復訓練...")
            model = YOLO("last.pt")
        elif weights:
            logger.info(f"\n從預訓練權重載入: {weights}")
            model = YOLO(str(weights))
        else:
            logger.info(f"\n建立新模型: {config['model']['name']}")
            model_name = config['model']['name']
            model = YOLO(f"{model_name}.yaml")

        # 訓練參數
        train_args = {
            # 資料
            "data": str(Path(config['dataset']['path']) / "data.yaml"),
            "imgsz": config['dataset']['image_size'][0],

            # 訓練
            "epochs": config['training']['epochs'],
            "batch": config['training']['batch_size'],
            "workers": config['training']['workers'],
            "device": config['training']['device'],

            # 優化器
            "optimizer": config['training']['optimizer'],
            "lr0": config['training']['lr0'],
            "lrf": config['training']['lrf'],
            "momentum": config['training']['momentum'],
            "weight_decay": config['training']['weight_decay'],

            # 增強
            "hsv_h": config['augmentation']['hsv_h'],
            "hsv_s": config['augmentation']['hsv_s'],
            "hsv_v": config['augmentation']['hsv_v'],
            "degrees": config['augmentation']['degrees'],
            "translate": config['augmentation']['translate'],
            "scale": config['augmentation']['scale'],
            "fliplr": config['augmentation']['fliplr'],
            "mosaic": config['augmentation']['mosaic'],
            "mixup": config['augmentation']['mixup'],

            # 輸出
            "project": config['output']['project'],
            "name": config['output']['name'],
            "exist_ok": config['output']['exist_ok'],

            # 進階
            "amp": config['advanced']['amp'],
            "patience": config['training']['patience'],
            "save_period": config['training']['save_period'],

            # 驗證
            "val": True,
            "plots": config['output']['save_plots'],
        }

        # 開始訓練
        logger.info("\n開始訓練...")
        results = model.train(**train_args)

        logger.info("\n訓練完成!")
        logger.info(f"  最佳模型: {results.save_dir / 'weights' / 'best.pt'}")
        logger.info(f"  最後模型: {results.save_dir / 'weights' / 'last.pt'}")

        # 驗證
        logger.info("\n驗證最佳模型...")
        metrics = model.val()

        logger.info("\n驗證結果:")
        logger.info(f"  mAP@0.5: {metrics.box.map50:.4f}")
        logger.info(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
        logger.info(f"  Precision: {metrics.box.mp:.4f}")
        logger.info(f"  Recall: {metrics.box.mr:.4f}")

        # 匯出模型
        logger.info("\n匯出模型...")
        export_formats = ["onnx"]  # 可選: "torchscript", "engine", etc.

        for fmt in export_formats:
            try:
                logger.info(f"  匯出為 {fmt.upper()}...")
                export_path = model.export(format=fmt)
                logger.info(f"  ✓ {fmt.upper()} 已匯出: {export_path}")
            except Exception as e:
                logger.warning(f"  ✗ {fmt.upper()} 匯出失敗: {e}")

        return results

    except ImportError:
        logger.error("\nultralytics 未安裝!")
        logger.info("安裝: pip install ultralytics")
        return None

    except Exception as e:
        logger.error(f"\n訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="YOLO 模型訓練")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/yolo_training.yaml",
        help="訓練配置檔案",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="從檢查點恢復訓練",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="預訓練權重路徑",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="訓練裝置 (覆寫配置)",
    )

    args = parser.parse_args()

    # 設置日誌
    setup_logger(name=__name__, log_dir="outputs/logs")

    # 檢查配置檔案
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置檔案不存在: {config_path}")
        return

    # 覆寫裝置設定 (如果指定)
    if args.device:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config['training']['device'] = args.device
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        logger.info(f"裝置設定已覆寫為: {args.device}")

    # 訓練
    weights_path = Path(args.weights) if args.weights else None
    results = train_yolo(config_path, args.resume, weights_path)

    if results:
        logger.info("\n✓ 訓練流程完成!")
    else:
        logger.error("\n✗ 訓練失敗")


if __name__ == "__main__":
    main()
