# 工具模組快速參考

快速查詢常用功能的程式碼片段。

---

## Logger

### 基本設定
```python
from src.utils import setup_logger

logger = setup_logger(name="MyApp", log_dir="outputs/logs", log_level="INFO")
```

### 記錄訊息
```python
logger.debug("除錯訊息")
logger.info("一般訊息")
logger.warning("警告訊息")
logger.error("錯誤訊息")
logger.success("成功訊息")
logger.exception("例外訊息")  # 自動包含堆疊追蹤
```

### 便利函數
```python
from src.utils import info, warning, error

info("快速記錄訊息")
```

---

## Config Loader

### 載入配置
```python
from src.utils import get_config_loader

loader = get_config_loader(config_dir="config")
camera_config = loader.load_camera_config()
model_config = loader.load_model_config()
```

### 讀取配置值
```python
# 直接存取
value = camera_config["camera"]["rgb"]["width"]

# 便利方法（推薦）
value = loader.get_camera_setting("camera", "rgb", "width")
value = loader.get_model_setting("inference", "conf", default=0.25)
```

### 自訂配置
```python
config = loader.load_yaml("config/custom.yaml")
loader.save_yaml(config, "config/new.yaml")
```

---

## Visualizer

### 初始化
```python
from src.utils import get_visualizer

visualizer = get_visualizer()
```

### 繪製邊界框
```python
result = visualizer.draw_bbox(
    image,
    bbox=(x1, y1, x2, y2),
    label="object",
    conf=0.95,
    class_id=0
)
```

### 完整檢測結果
```python
result = visualizer.draw_detection_results(
    image,
    boxes=boxes,        # (N, 4)
    masks=masks,        # (N, H, W) 可選
    labels=labels,      # List[str]
    scores=scores,      # (N,)
    class_ids=class_ids # (N,)
)
```

### 深度圖
```python
depth_colored = visualizer.draw_depth_map(depth, colormap=cv2.COLORMAP_JET)
```

### 比較視圖
```python
comparison = visualizer.create_comparison_view(
    rgb_image=rgb,
    depth_image=depth,
    detection_image=result
)
```

### 儲存結果
```python
visualizer.save_image(result, "outputs/result.png")
```

---

## 完整範例

### 檢測流程
```python
from src.utils import setup_logger, get_config_loader, get_visualizer
import cv2
import numpy as np

# 初始化
logger = setup_logger(name="Detection", log_dir="outputs/logs")
config = get_config_loader()
visualizer = get_visualizer()

# 載入配置
model_config = config.load_model_config()
conf_threshold = config.get_model_setting("inference", "conf")

# 讀取影像
image = cv2.imread("test.jpg")
logger.info(f"載入影像: {image.shape}")

# 模擬檢測（實際使用時替換為真實檢測）
boxes = np.array([[100, 100, 300, 250]])
labels = ["defect"]
scores = np.array([0.92])

# 視覺化
result = visualizer.draw_detection_results(
    image, boxes, labels=labels, scores=scores
)

# 儲存
visualizer.save_image(result, "outputs/result.png")
logger.success("檢測完成")
```

### RGB-D 處理
```python
from src.utils import get_visualizer
import cv2
import numpy as np

visualizer = get_visualizer()

# 讀取資料
rgb = cv2.imread("rgb.jpg")
depth = np.load("depth.npy")

# 視覺化深度
depth_colored = visualizer.draw_depth_map(depth)

# 比較視圖
comparison = visualizer.create_comparison_view(
    rgb_image=rgb,
    depth_image=depth_colored
)

visualizer.save_image(comparison, "outputs/comparison.png")
```

---

## 常用參數

### Logger 參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `name` | "InspectionSystem" | 日誌名稱 |
| `log_dir` | None | 日誌目錄 |
| `log_level` | "INFO" | 等級 |
| `rotation` | "10 MB" | 輪換策略 |
| `retention` | "7 days" | 保留時間 |

### Visualizer 參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `font_scale` | 0.5 | 字體大小 |
| `line_width` | 2 | 線條寬度 |
| `alpha` | 0.4 | 遮罩透明度 |

---

## 檔案路徑

- **配置檔案**: `config/`
- **日誌檔案**: `outputs/logs/`
- **視覺化結果**: `outputs/`
- **測試檔案**: `tests/`

---

## 執行測試

```bash
# 工具模組測試
python tests/test_utils.py

# 示範腳本
python scripts/demo_utils.py

# 互動式展示
python scripts/interactive_demo.py
```

---

## 更多資訊

詳細使用說明請參考: [UTILS_USAGE.md](UTILS_USAGE.md)
