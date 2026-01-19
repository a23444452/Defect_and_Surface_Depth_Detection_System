# 工具模組使用指南

本文檔說明如何使用專案中的工具模組，包括 Logger、Config Loader 和 Visualizer。

---

## 目錄

1. [Logger 模組](#logger-模組)
2. [Config Loader 模組](#config-loader-模組)
3. [Visualizer 模組](#visualizer-模組)
4. [完整範例](#完整範例)

---

## Logger 模組

基於 Loguru 的日誌系統，提供彩色終端輸出和自動檔案管理。

### 基本用法

#### 方法 1：使用全域 logger（推薦）

```python
from src.utils import setup_logger

# 初始化全域 logger
logger = setup_logger(
    name="MyApp",
    log_dir="outputs/logs",
    log_level="INFO"
)

# 使用 logger
logger.debug("這是除錯訊息")
logger.info("這是一般訊息")
logger.warning("這是警告訊息")
logger.error("這是錯誤訊息")
logger.critical("這是嚴重錯誤訊息")
logger.success("這是成功訊息")
```

#### 方法 2：使用便利函數

```python
from src.utils import info, warning, error, success

# 直接使用函數（會使用全域 logger）
info("應用程式啟動")
success("初始化完成")
warning("記憶體使用率過高")
error("無法連接到相機")
```

#### 方法 3：建立多個 logger 實例

```python
from src.utils import Logger

# 建立不同用途的 logger
camera_logger = Logger(
    name="Camera",
    log_dir="outputs/logs",
    log_level="DEBUG"
)

model_logger = Logger(
    name="Model",
    log_dir="outputs/logs",
    log_level="INFO"
)

camera_logger.info("相機已連接")
model_logger.info("模型已載入")
```

### 例外處理日誌

```python
from src.utils import get_logger

logger = get_logger()

try:
    result = risky_operation()
except Exception as e:
    # exception() 會自動記錄堆疊追蹤
    logger.exception("操作失敗")
    raise
```

### Logger 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `name` | 日誌名稱 | "InspectionSystem" |
| `log_dir` | 日誌檔案目錄 | None（不儲存檔案） |
| `log_level` | 日誌等級 | "INFO" |
| `rotation` | 日誌輪換策略 | "10 MB" |
| `retention` | 日誌保留時間 | "7 days" |
| `console_output` | 是否輸出到終端 | True |

### 日誌檔案

Logger 會自動建立兩個日誌檔案：
- `{name}.log` - 所有日誌
- `{name}_error.log` - 只有錯誤日誌

---

## Config Loader 模組

YAML 配置檔案載入與管理工具。

### 基本用法

```python
from src.utils import get_config_loader

# 取得全域配置載入器
loader = get_config_loader(config_dir="config")

# 載入配置檔案
camera_config = loader.load_camera_config()
model_config = loader.load_model_config()
```

### 讀取配置值

#### 方法 1：直接存取字典

```python
# 載入配置
camera_config = loader.load_camera_config()

# 存取配置值
camera_model = camera_config["camera"]["model"]
rgb_width = camera_config["camera"]["rgb"]["width"]
```

#### 方法 2：使用便利方法（推薦）

```python
# 使用 get_camera_setting（支援預設值）
camera_model = loader.get_camera_setting("camera", "model")
rgb_width = loader.get_camera_setting("camera", "rgb", "width")
depth_min = loader.get_camera_setting(
    "depth_processing", "depth_range", "min",
    default=150
)

# 使用 get_model_setting
model_type = loader.get_model_setting("model", "type")
conf_threshold = loader.get_model_setting("inference", "conf", default=0.25)
```

### 載入自訂配置檔案

```python
from src.utils import ConfigLoader

loader = ConfigLoader(config_dir="config")

# 載入自訂 YAML 檔案
custom_config = loader.load_yaml("config/custom_settings.yaml")
```

### 儲存配置

```python
# 修改配置
new_config = {
    "camera": {
        "model": "Gemini 2",
        "rgb": {"width": 1920, "height": 1080}
    }
}

# 儲存為 YAML
loader.save_yaml(new_config, "config/new_camera_config.yaml")
```

### 合併配置

```python
# 基礎配置
base_config = loader.load_yaml("config/base_config.yaml")

# 覆蓋配置
override_config = {
    "camera": {
        "rgb": {"fps": 60}  # 只覆蓋 FPS
    }
}

# 合併（override 覆蓋 base）
merged_config = loader.merge_configs(base_config, override_config)
```

---

## Visualizer 模組

檢測結果視覺化工具。

### 基本用法

```python
from src.utils import get_visualizer
import numpy as np
import cv2

# 取得全域視覺化工具
visualizer = get_visualizer()

# 讀取影像
image = cv2.imread("test.jpg")
```

### 繪製邊界框

```python
# 單個邊界框
bbox = (100, 100, 300, 250)  # (x1, y1, x2, y2)
result = visualizer.draw_bbox(
    image,
    bbox=bbox,
    label="metal_part",
    conf=0.95,
    class_id=0
)

# 儲存結果
visualizer.save_image(result, "outputs/bbox_result.png")
```

### 繪製完整檢測結果

```python
# 準備檢測結果
boxes = np.array([
    [50, 50, 200, 200],
    [250, 100, 400, 300],
])

labels = ["metal_part", "plastic_part"]
scores = np.array([0.95, 0.87])
class_ids = np.array([0, 1])

# 繪製所有結果
result = visualizer.draw_detection_results(
    image,
    boxes=boxes,
    labels=labels,
    scores=scores,
    class_ids=class_ids
)

visualizer.save_image(result, "outputs/detection_result.png")
```

### 繪製分割遮罩

```python
# 準備遮罩（與 boxes 對應）
masks = np.array([
    mask1,  # (H, W) 二值遮罩
    mask2,
])

# 繪製檢測結果（包含遮罩）
result = visualizer.draw_detection_results(
    image,
    boxes=boxes,
    masks=masks,  # 加入遮罩
    labels=labels,
    scores=scores,
    class_ids=class_ids
)
```

### 繪製深度圖

```python
# 深度資料（單位：mm）
depth = np.random.rand(480, 640) * 1000

# 轉換為彩色深度圖
depth_colored = visualizer.draw_depth_map(
    depth,
    colormap=cv2.COLORMAP_JET  # 可選：TURBO, VIRIDIS, etc.
)

visualizer.save_image(depth_colored, "outputs/depth.png")
```

### 建立比較視圖

```python
# RGB 影像、深度圖、檢測結果並排顯示
comparison = visualizer.create_comparison_view(
    rgb_image=rgb_image,
    depth_image=depth_colored,
    detection_image=detection_result
)

visualizer.save_image(comparison, "outputs/comparison.png")
```

### 繪製文字

```python
result = visualizer.draw_text(
    image,
    text="檢測中...",
    position=(10, 30),
    color=(0, 255, 0),
    background=True
)
```

### 繪製關鍵點

```python
# 關鍵點座標 (N, 2)
keypoints = np.array([
    [100, 100],
    [200, 150],
    [150, 200],
])

result = visualizer.draw_keypoints(
    image,
    keypoints=keypoints,
    color=(0, 255, 0),
    radius=5
)
```

### 繪製指標圖表

```python
# 訓練指標
metrics = {
    "mAP": 0.85,
    "Precision": 0.92,
    "Recall": 0.88,
    "F1-Score": 0.90,
}

# 繪製圖表
fig = visualizer.plot_metrics(
    metrics,
    title="模型效能指標",
    save_path="outputs/metrics.png"
)
```

### Visualizer 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `font_scale` | 字體大小 | 0.5 |
| `font_thickness` | 字體粗細 | 1 |
| `line_width` | 線條寬度 | 2 |
| `alpha` | 遮罩透明度 | 0.4 |

```python
# 自訂參數
visualizer = get_visualizer(
    font_scale=0.8,
    line_width=3,
    alpha=0.5
)
```

---

## 完整範例

### 範例 1：檢測結果處理流程

```python
from src.utils import setup_logger, get_config_loader, get_visualizer
import cv2
import numpy as np

# 1. 初始化
logger = setup_logger(name="Detection", log_dir="outputs/logs")
config = get_config_loader()
visualizer = get_visualizer()

# 2. 載入配置
model_config = config.load_model_config()
conf_threshold = config.get_model_setting("inference", "conf")
logger.info(f"信心閾值: {conf_threshold}")

# 3. 讀取影像
image_path = "data/test.jpg"
image = cv2.imread(image_path)
logger.info(f"載入影像: {image_path}")

# 4. 模擬檢測結果
boxes = np.array([[100, 100, 300, 250]])
labels = ["defect"]
scores = np.array([0.92])
class_ids = np.array([0])

# 5. 視覺化
result = visualizer.draw_detection_results(
    image, boxes, labels=labels, scores=scores, class_ids=class_ids
)

# 6. 儲存結果
output_path = "outputs/detection_result.png"
visualizer.save_image(result, output_path)
logger.success(f"結果已儲存: {output_path}")
```

### 範例 2：RGB-D 處理流程

```python
from src.utils import setup_logger, get_visualizer
import cv2
import numpy as np

# 初始化
logger = setup_logger(name="RGBD", log_dir="outputs/logs")
visualizer = get_visualizer()

# 讀取 RGB 和深度影像
rgb_image = cv2.imread("data/rgb.jpg")
depth_data = np.load("data/depth.npy")

logger.info("RGB-D 影像已載入")

# 深度圖視覺化
depth_colored = visualizer.draw_depth_map(depth_data)

# 建立比較視圖
comparison = visualizer.create_comparison_view(
    rgb_image=rgb_image,
    depth_image=depth_colored
)

# 儲存
visualizer.save_image(comparison, "outputs/rgbd_comparison.png")
logger.success("RGB-D 比較圖已儲存")
```

### 範例 3：配置動態調整

```python
from src.utils import ConfigLoader, setup_logger

logger = setup_logger(name="ConfigDemo")
loader = ConfigLoader(config_dir="config")

# 載入基礎配置
base_config = loader.load_model_config()

# 動態調整（例如：根據平台）
platform = "jetson"  # 或 "pc"

if platform == "jetson":
    # Jetson 平台使用較小模型
    override = {
        "model": {"variant": "s"},
        "inference": {"half": True}
    }
else:
    # PC 平台使用較大模型
    override = {
        "model": {"variant": "m"},
        "inference": {"half": False}
    }

# 合併配置
final_config = loader.merge_configs(base_config, override)

# 使用配置
model_variant = final_config["model"]["variant"]
logger.info(f"使用模型變體: {model_variant}")
```

### 範例 4：錯誤處理與日誌

```python
from src.utils import setup_logger

logger = setup_logger(name="ErrorDemo", log_level="DEBUG")

def process_image(image_path):
    try:
        logger.debug(f"開始處理影像: {image_path}")

        # 模擬處理
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"影像不存在: {image_path}")

        logger.info(f"影像大小: {image.shape}")
        logger.success("處理完成")

        return image

    except FileNotFoundError as e:
        logger.error(f"檔案錯誤: {e}")
        raise

    except Exception as e:
        logger.exception("處理過程發生未預期錯誤")
        raise

# 使用
try:
    result = process_image("data/test.jpg")
except Exception:
    logger.critical("程式終止")
```

---

## 最佳實踐

### 1. Logger

- ✅ 在應用程式入口點初始化 logger
- ✅ 使用適當的日誌等級
- ✅ 重要操作前後都記錄日誌
- ✅ 例外處理使用 `logger.exception()`
- ❌ 避免在迴圈中記錄過多日誌

### 2. Config Loader

- ✅ 在程式開始時載入所有配置
- ✅ 使用便利方法讀取配置值
- ✅ 提供合理的預設值
- ✅ 驗證配置值的有效性
- ❌ 避免重複載入相同配置

### 3. Visualizer

- ✅ 重複使用全域 visualizer 實例
- ✅ 適當調整字體大小和線條寬度
- ✅ 使用比較視圖展示多種結果
- ✅ 儲存重要的視覺化結果
- ❌ 避免在生產環境顯示過多視覺化

---

## 測試與驗證

### 執行測試

```bash
# 執行工具模組測試
python tests/test_utils.py

# 執行示範腳本
python scripts/demo_utils.py
```

### 預期輸出

測試成功後會生成：
- `outputs/logs/*.log` - 日誌檔案
- `outputs/*.png` - 視覺化結果

---

## 常見問題

### Q: 如何改變日誌等級？

```python
logger = setup_logger(log_level="DEBUG")  # DEBUG, INFO, WARNING, ERROR
```

### Q: 如何自訂顏色？

```python
# 直接指定 BGR 顏色
color = (255, 0, 0)  # 藍色
visualizer.draw_bbox(image, bbox, color=color)
```

### Q: 配置檔案找不到？

```python
from pathlib import Path

# 使用絕對路徑
config_dir = Path(__file__).parent / "config"
loader = ConfigLoader(config_dir=str(config_dir))
```

### Q: 如何禁用終端輸出？

```python
logger = Logger(
    name="Silent",
    log_dir="outputs/logs",
    console_output=False  # 只輸出到檔案
)
```

---

## 參考資源

- [Loguru 文檔](https://loguru.readthedocs.io/)
- [PyYAML 文檔](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [OpenCV 文檔](https://docs.opencv.org/)
- [Matplotlib 文檔](https://matplotlib.org/stable/contents.html)
