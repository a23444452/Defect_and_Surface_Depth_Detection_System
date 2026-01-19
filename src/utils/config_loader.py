"""
配置檔案載入模組
支援 YAML 格式的配置檔案載入與驗證
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from pydantic import BaseModel, Field, ValidationError


class CameraConfig(BaseModel):
    """相機配置"""

    model: str = "Gemini 2"
    serial_number: str = ""
    rgb: Dict[str, Any] = Field(default_factory=dict)
    depth: Dict[str, Any] = Field(default_factory=dict)
    alignment: Dict[str, Any] = Field(default_factory=dict)
    auto_exposure: Dict[str, Any] = Field(default_factory=dict)
    auto_white_balance: Dict[str, Any] = Field(default_factory=dict)


class CalibrationConfig(BaseModel):
    """標定配置"""

    rgb_intrinsics: Dict[str, float] = Field(default_factory=dict)
    depth_intrinsics: Dict[str, float] = Field(default_factory=dict)
    distortion: Dict[str, list] = Field(default_factory=dict)
    extrinsics: Dict[str, list] = Field(default_factory=dict)


class DepthProcessingConfig(BaseModel):
    """深度處理配置"""

    depth_range: Dict[str, int] = Field(default_factory=dict)
    filters: Dict[str, Any] = Field(default_factory=dict)


class PreprocessingConfig(BaseModel):
    """預處理配置"""

    rgb: Dict[str, Any] = Field(default_factory=dict)
    depth: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """模型配置"""

    type: str = "yolov11"
    variant: str = "m"
    task: str = "segment"
    pretrained: str = ""
    custom_weights: str = ""


class TrainingConfig(BaseModel):
    """訓練配置"""

    data: str = ""
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    optimizer: str = "AdamW"
    lr0: float = 0.001
    augmentation: Dict[str, Any] = Field(default_factory=dict)


class InferenceConfig(BaseModel):
    """推理配置"""

    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 100
    classes: list = Field(default_factory=list)
    half: bool = False
    batch_size: int = 1
    device: str = "cuda:0"


class ConfigLoader:
    """
    配置檔案載入器
    支援 YAML 格式的配置檔案載入與驗證
    """

    def __init__(self, config_dir: Union[str, Path] = "config"):
        """
        初始化配置載入器

        Args:
            config_dir: 配置檔案目錄
        """
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Dict[str, Any]] = {}

    def load_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        載入 YAML 檔案

        Args:
            file_path: YAML 檔案路徑

        Returns:
            配置字典

        Raises:
            FileNotFoundError: 檔案不存在
            yaml.YAMLError: YAML 格式錯誤
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"配置檔案不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
                return config if config is not None else {}
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"YAML 格式錯誤: {e}")

    def load_camera_config(
        self, config_file: str = "camera_config.yaml"
    ) -> Dict[str, Any]:
        """
        載入相機配置

        Args:
            config_file: 配置檔案名稱

        Returns:
            相機配置字典
        """
        config_path = self.config_dir / config_file
        config = self.load_yaml(config_path)
        self._configs["camera"] = config
        return config

    def load_model_config(
        self, config_file: str = "model_config.yaml"
    ) -> Dict[str, Any]:
        """
        載入模型配置

        Args:
            config_file: 配置檔案名稱

        Returns:
            模型配置字典
        """
        config_path = self.config_dir / config_file
        config = self.load_yaml(config_path)
        self._configs["model"] = config
        return config

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        取得已載入的配置

        Args:
            name: 配置名稱 (camera, model)

        Returns:
            配置字典，如果不存在則返回 None
        """
        return self._configs.get(name)

    def get_camera_setting(self, *keys: str, default: Any = None) -> Any:
        """
        取得相機配置的特定設定

        Args:
            *keys: 配置鍵值路徑 (例如: "rgb", "width")
            default: 預設值

        Returns:
            配置值

        Example:
            >>> loader = ConfigLoader()
            >>> loader.load_camera_config()
            >>> width = loader.get_camera_setting("camera", "rgb", "width")
        """
        config = self._configs.get("camera", {})
        for key in keys:
            if isinstance(config, dict):
                config = config.get(key, default)
            else:
                return default
        return config

    def get_model_setting(self, *keys: str, default: Any = None) -> Any:
        """
        取得模型配置的特定設定

        Args:
            *keys: 配置鍵值路徑
            default: 預設值

        Returns:
            配置值
        """
        config = self._configs.get("model", {})
        for key in keys:
            if isinstance(config, dict):
                config = config.get(key, default)
            else:
                return default
        return config

    def save_yaml(self, config: Dict[str, Any], file_path: Union[str, Path]):
        """
        儲存配置為 YAML 檔案

        Args:
            config: 配置字典
            file_path: 儲存路徑
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                config, f, default_flow_style=False, allow_unicode=True, indent=2
            )

    def validate_camera_config(self, config: Dict[str, Any]) -> bool:
        """
        驗證相機配置

        Args:
            config: 配置字典

        Returns:
            是否有效

        Raises:
            ValidationError: 驗證失敗
        """
        try:
            if "camera" in config:
                CameraConfig(**config["camera"])
            if "calibration" in config:
                CalibrationConfig(**config["calibration"])
            if "depth_processing" in config:
                DepthProcessingConfig(**config["depth_processing"])
            if "preprocessing" in config:
                PreprocessingConfig(**config["preprocessing"])
            return True
        except ValidationError as e:
            raise ValidationError(f"相機配置驗證失敗: {e}")

    def validate_model_config(self, config: Dict[str, Any]) -> bool:
        """
        驗證模型配置

        Args:
            config: 配置字典

        Returns:
            是否有效

        Raises:
            ValidationError: 驗證失敗
        """
        try:
            if "model" in config:
                ModelConfig(**config["model"])
            if "training" in config:
                TrainingConfig(**config["training"])
            if "inference" in config:
                InferenceConfig(**config["inference"])
            return True
        except ValidationError as e:
            raise ValidationError(f"模型配置驗證失敗: {e}")

    def merge_configs(
        self, base_config: Dict[str, Any], override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        合併兩個配置字典（override 會覆蓋 base）

        Args:
            base_config: 基礎配置
            override_config: 覆蓋配置

        Returns:
            合併後的配置
        """
        merged = base_config.copy()

        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged


# 全域配置載入器實例
_global_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: Union[str, Path] = "config") -> ConfigLoader:
    """
    取得全域配置載入器實例

    Args:
        config_dir: 配置檔案目錄

    Returns:
        ConfigLoader 實例
    """
    global _global_config_loader

    if _global_config_loader is None:
        _global_config_loader = ConfigLoader(config_dir=config_dir)

    return _global_config_loader


if __name__ == "__main__":
    # 測試程式碼
    loader = ConfigLoader(config_dir="config")

    # 載入相機配置
    print("載入相機配置...")
    camera_config = loader.load_camera_config()
    print(f"相機型號: {camera_config.get('camera', {}).get('model')}")
    print(f"RGB 解析度: {camera_config.get('camera', {}).get('rgb', {}).get('width')}x{camera_config.get('camera', {}).get('rgb', {}).get('height')}")

    # 載入模型配置
    print("\n載入模型配置...")
    model_config = loader.load_model_config()
    print(f"模型類型: {model_config.get('model', {}).get('type')}")
    print(f"模型變體: {model_config.get('model', {}).get('variant')}")

    # 使用便利方法
    print("\n使用便利方法取得設定...")
    rgb_width = loader.get_camera_setting("camera", "rgb", "width")
    print(f"RGB 寬度: {rgb_width}")

    model_type = loader.get_model_setting("model", "type")
    print(f"模型類型: {model_type}")

    print("\n配置載入測試完成！")
