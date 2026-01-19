"""
日誌模組
使用 Loguru 提供統一的日誌介面
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


class Logger:
    """
    日誌管理器
    提供統一的日誌配置與介面
    """

    def __init__(
        self,
        name: str = "InspectionSystem",
        log_dir: Optional[str] = None,
        log_level: str = "INFO",
        rotation: str = "10 MB",
        retention: str = "7 days",
        console_output: bool = True,
    ):
        """
        初始化日誌管理器

        Args:
            name: 日誌名稱
            log_dir: 日誌檔案目錄，None 則不儲存到檔案
            log_level: 日誌等級 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            rotation: 日誌輪換策略 (檔案大小或時間)
            retention: 日誌保留時間
            console_output: 是否輸出到終端
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_level = log_level

        # 移除預設的 handler
        logger.remove()

        # 設定日誌格式
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        # 加入終端輸出
        if console_output:
            logger.add(
                sys.stderr,
                format=log_format,
                level=log_level,
                colorize=True,
                backtrace=True,
                diagnose=True,
            )

        # 加入檔案輸出
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # 一般日誌
            logger.add(
                self.log_dir / f"{name}.log",
                format=log_format,
                level=log_level,
                rotation=rotation,
                retention=retention,
                compression="zip",
                backtrace=True,
                diagnose=True,
            )

            # 錯誤日誌（單獨檔案）
            logger.add(
                self.log_dir / f"{name}_error.log",
                format=log_format,
                level="ERROR",
                rotation=rotation,
                retention=retention,
                compression="zip",
                backtrace=True,
                diagnose=True,
            )

        self.logger = logger

    def debug(self, message: str, **kwargs):
        """除錯訊息"""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """一般訊息"""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """警告訊息"""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """錯誤訊息"""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """嚴重錯誤訊息"""
        self.logger.critical(message, **kwargs)

    def exception(self, message: str, **kwargs):
        """例外訊息（包含堆疊追蹤）"""
        self.logger.exception(message, **kwargs)

    def success(self, message: str, **kwargs):
        """成功訊息"""
        self.logger.success(message, **kwargs)

    def get_logger(self):
        """取得原始 loguru logger 實例"""
        return self.logger


# 全域 logger 實例
_global_logger: Optional[Logger] = None


def get_logger(
    name: str = "InspectionSystem",
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    **kwargs,
) -> Logger:
    """
    取得全域 logger 實例

    Args:
        name: 日誌名稱
        log_dir: 日誌檔案目錄
        log_level: 日誌等級
        **kwargs: 其他 Logger 參數

    Returns:
        Logger 實例
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = Logger(
            name=name, log_dir=log_dir, log_level=log_level, **kwargs
        )

    return _global_logger


def setup_logger(
    name: str = "InspectionSystem",
    log_dir: Optional[str] = "outputs/logs",
    log_level: str = "INFO",
    **kwargs,
) -> Logger:
    """
    設定全域 logger

    Args:
        name: 日誌名稱
        log_dir: 日誌檔案目錄
        log_level: 日誌等級
        **kwargs: 其他 Logger 參數

    Returns:
        Logger 實例
    """
    global _global_logger

    _global_logger = Logger(
        name=name, log_dir=log_dir, log_level=log_level, **kwargs
    )

    return _global_logger


# 便利函數
def debug(message: str, **kwargs):
    """除錯訊息"""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    """一般訊息"""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    """警告訊息"""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs):
    """錯誤訊息"""
    get_logger().error(message, **kwargs)


def critical(message: str, **kwargs):
    """嚴重錯誤訊息"""
    get_logger().critical(message, **kwargs)


def exception(message: str, **kwargs):
    """例外訊息（包含堆疊追蹤）"""
    get_logger().exception(message, **kwargs)


def success(message: str, **kwargs):
    """成功訊息"""
    get_logger().success(message, **kwargs)


if __name__ == "__main__":
    # 測試程式碼
    test_logger = setup_logger(
        name="Test", log_dir="outputs/logs", log_level="DEBUG"
    )

    test_logger.debug("這是除錯訊息")
    test_logger.info("這是一般訊息")
    test_logger.warning("這是警告訊息")
    test_logger.error("這是錯誤訊息")
    test_logger.success("這是成功訊息")

    # 測試例外處理
    try:
        1 / 0
    except Exception as e:
        test_logger.exception("發生例外")

    print("\n日誌測試完成！")
