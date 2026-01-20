#!/usr/bin/env python3
"""
模型優化器
用於將 PyTorch 模型轉換為 TensorRT 引擎並進行量化
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import logging

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

# 設置日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    模型優化器
    支援多種優化策略用於 Jetson 部署
    """

    def __init__(self):
        """初始化優化器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用裝置: {self.device}")

    # ==================== FP16 量化 ====================

    def convert_to_fp16(
        self, model: torch.nn.Module, save_path: Optional[Path] = None
    ) -> torch.nn.Module:
        """
        轉換模型為 FP16

        Args:
            model: PyTorch 模型
            save_path: 儲存路徑 (可選)

        Returns:
            FP16 模型
        """
        logger.info("轉換模型為 FP16...")

        # 轉換為 FP16
        model = model.half()
        model = model.to(self.device)

        # 儲存
        if save_path:
            torch.save(model.state_dict(), save_path)
            logger.info(f"FP16 模型已儲存至: {save_path}")

        return model

    # ==================== ONNX 匯出 ====================

    def export_to_onnx(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        onnx_path: Path,
        opset_version: int = 13,
        dynamic_axes: Optional[dict] = None,
    ) -> bool:
        """
        匯出模型為 ONNX 格式

        Args:
            model: PyTorch 模型
            input_shape: 輸入形狀 (B, C, H, W)
            onnx_path: ONNX 檔案路徑
            opset_version: ONNX opset 版本
            dynamic_axes: 動態軸配置

        Returns:
            是否成功
        """
        try:
            logger.info(f"匯出 ONNX 模型: {onnx_path}")

            # 建立測試輸入
            dummy_input = torch.randn(*input_shape).to(self.device)

            # 預設動態軸
            if dynamic_axes is None:
                dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

            # 匯出
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )

            logger.info(f"✓ ONNX 模型已匯出: {onnx_path}")

            # 驗證 ONNX 模型
            self._verify_onnx(onnx_path)

            return True

        except Exception as e:
            logger.error(f"ONNX 匯出失敗: {e}")
            return False

    def _verify_onnx(self, onnx_path: Path):
        """驗證 ONNX 模型"""
        try:
            import onnx

            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            logger.info("✓ ONNX 模型驗證通過")

        except ImportError:
            logger.warning("未安裝 onnx 套件,跳過驗證")
        except Exception as e:
            logger.warning(f"ONNX 驗證失敗: {e}")

    # ==================== TensorRT 轉換 ====================

    def convert_to_tensorrt(
        self,
        onnx_path: Path,
        engine_path: Path,
        precision: str = "fp16",
        workspace_size: int = 2,
        max_batch_size: int = 1,
    ) -> bool:
        """
        將 ONNX 模型轉換為 TensorRT 引擎

        Args:
            onnx_path: ONNX 模型路徑
            engine_path: TensorRT 引擎輸出路徑
            precision: 精度 (fp32, fp16, int8)
            workspace_size: 工作空間大小 (GB)
            max_batch_size: 最大批次大小

        Returns:
            是否成功
        """
        try:
            import tensorrt as trt

            logger.info(f"轉換 TensorRT 引擎: {precision.upper()}")
            logger.info(f"  輸入: {onnx_path}")
            logger.info(f"  輸出: {engine_path}")

            # 建立 logger
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)

            # 建立 builder
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)

            # 解析 ONNX
            logger.info("解析 ONNX 模型...")
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    logger.error("ONNX 解析失敗:")
                    for error in range(parser.num_errors):
                        logger.error(f"  {parser.get_error(error)}")
                    return False

            # 建立 config
            config = builder.create_builder_config()

            # 設置工作空間
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30)
            )

            # 設置精度
            if precision == "fp16":
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("✓ 啟用 FP16")
                else:
                    logger.warning("平台不支援 FP16,使用 FP32")

            elif precision == "int8":
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    logger.info("✓ 啟用 INT8")
                    # INT8 需要校準資料,這裡簡化處理
                    logger.warning("INT8 需要校準資料,請參考文檔")
                else:
                    logger.warning("平台不支援 INT8,使用 FP32")

            # 建立引擎
            logger.info("建立 TensorRT 引擎 (可能需要數分鐘)...")
            engine = builder.build_serialized_network(network, config)

            if engine is None:
                logger.error("引擎建立失敗")
                return False

            # 儲存引擎
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            with open(engine_path, "wb") as f:
                f.write(engine)

            logger.info(f"✓ TensorRT 引擎已儲存: {engine_path}")
            logger.info(f"  檔案大小: {engine_path.stat().st_size / 1e6:.2f} MB")

            return True

        except ImportError:
            logger.error("TensorRT 未安裝")
            logger.info("請在 Jetson 上安裝 TensorRT")
            return False

        except Exception as e:
            logger.error(f"TensorRT 轉換失敗: {e}")
            import traceback

            traceback.print_exc()
            return False

    # ==================== 效能測試 ====================

    def benchmark_model(
        self, model: torch.nn.Module, input_shape: Tuple[int, ...], num_runs: int = 100
    ):
        """
        測試模型效能

        Args:
            model: 模型
            input_shape: 輸入形狀
            num_runs: 執行次數
        """
        logger.info(f"測試模型效能 ({num_runs} 次)...")

        model = model.to(self.device)
        model.eval()

        # 建立測試輸入
        dummy_input = torch.randn(*input_shape).to(self.device)

        # 暖身
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # 測試
        import time

        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.time()

                _ = model(dummy_input)

                torch.cuda.synchronize()
                end = time.time()

                times.append(end - start)

        times = np.array(times)

        logger.info(f"  平均時間: {times.mean()*1000:.2f} ms")
        logger.info(f"  中位數: {np.median(times)*1000:.2f} ms")
        logger.info(f"  標準差: {times.std()*1000:.2f} ms")
        logger.info(f"  FPS: {1.0/times.mean():.1f}")


def main():
    """示範使用"""
    logger.info("模型優化器示範")

    # 建立優化器
    optimizer = ModelOptimizer()

    # 建立測試模型
    logger.info("\n建立測試模型...")

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(64, 10)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = SimpleModel()
    model = model.to(optimizer.device)
    model.eval()

    input_shape = (1, 3, 224, 224)

    # 測試原始模型
    logger.info("\n測試原始模型 (FP32):")
    optimizer.benchmark_model(model, input_shape, num_runs=50)

    # 轉換為 FP16
    logger.info("\n轉換為 FP16:")
    fp16_model = optimizer.convert_to_fp16(
        model, save_path=Path("outputs/models/test_fp16.pth")
    )

    # 測試 FP16 模型
    logger.info("\n測試 FP16 模型:")
    optimizer.benchmark_model(fp16_model, input_shape, num_runs=50)

    # 匯出 ONNX
    logger.info("\n匯出 ONNX:")
    onnx_path = Path("outputs/models/test.onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer.export_to_onnx(model, input_shape, onnx_path)

    # 轉換 TensorRT (僅在 Jetson 上執行)
    logger.info("\n轉換 TensorRT:")
    engine_path = Path("outputs/models/test_fp16.engine")
    success = optimizer.convert_to_tensorrt(
        onnx_path, engine_path, precision="fp16", workspace_size=2
    )

    if success:
        logger.info("\n✓ 模型優化完成!")
    else:
        logger.warning("\n部分優化失敗 (可能因為不在 Jetson 平台)")


if __name__ == "__main__":
    main()
