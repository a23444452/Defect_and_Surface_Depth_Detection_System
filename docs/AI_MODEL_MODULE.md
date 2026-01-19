# AI 模型模組文檔

本文檔說明 AI 模型模組的架構、使用方式與實作細節。

---

## 📚 目錄

1. [模組概述](#模組概述)
2. [核心架構](#核心架構)
3. [資料類別](#資料類別)
4. [YOLOv11 檢測器](#yolov11-檢測器)
5. [使用範例](#使用範例)
6. [整合應用](#整合應用)
7. [效能考量](#效能考量)
8. [故障排除](#故障排除)

---

## 模組概述

### 功能特性

AI 模型模組提供物體檢測與實例分割功能:

- ✅ **標準化介面**: 抽象基底類別定義統一的檢測介面
- ✅ **YOLOv11 整合**: 基於 Ultralytics YOLO 框架
- ✅ **靈活的資料結構**: 完整的檢測結果封裝
- ✅ **結果過濾**: 依信心度或類別過濾檢測結果
- ✅ **多任務支援**: 同時支援檢測與分割
- ✅ **效能優化**: 支援 CPU/GPU/MPS 加速

### 技術規格

| 項目 | 規格 |
|------|------|
| 深度學習框架 | PyTorch 2.9.1+ |
| YOLO 版本 | YOLOv11 (Ultralytics) |
| 支援任務 | Detection, Segmentation |
| 輸入格式 | BGR 影像 (OpenCV 格式) |
| 支援裝置 | CPU, CUDA, MPS |
| 最小 Python 版本 | 3.8+ |

---

## 核心架構

### 模組結構

```
src/models/
├── __init__.py              # 模組匯出
├── detector_interface.py    # 抽象介面與資料類別
└── yolo_detector.py         # YOLOv11 檢測器實作
```

### 類別階層

```
DetectorInterface (ABC)
    └── YOLOv11Detector
```

### 設計模式

1. **抽象工廠模式**: `DetectorInterface` 定義標準介面
2. **資料類別模式**: 使用 `@dataclass` 封裝結果
3. **組合模式**: `DetectionResult` 組合多個 `DetectionBox` 與 `SegmentationMask`

---

## 資料類別

### DetectionBox

邊界框資料類別,表示單一物體的檢測框。

**屬性:**

| 屬性 | 型別 | 說明 |
|------|------|------|
| `x1` | float | 左上角 x 座標 |
| `y1` | float | 左上角 y 座標 |
| `x2` | float | 右下角 x 座標 |
| `y2` | float | 右下角 y 座標 |
| `confidence` | float | 信心度 (0-1) |
| `class_id` | int | 類別 ID |
| `class_name` | str | 類別名稱 |

**屬性 (計算):**

| 屬性 | 型別 | 說明 |
|------|------|------|
| `width` | float | 邊界框寬度 |
| `height` | float | 邊界框高度 |
| `center` | Tuple[float, float] | 中心點座標 |
| `area` | float | 邊界框面積 |

**方法:**

```python
# 格式轉換
box.to_xyxy()      # (x1, y1, x2, y2)
box.to_xywh()      # (x, y, w, h) - COCO 格式
box.to_cxcywh()    # (cx, cy, w, h) - YOLO 格式
```

**範例:**

```python
from src.models import DetectionBox

box = DetectionBox(
    x1=100, y1=150, x2=300, y2=400,
    confidence=0.95,
    class_id=0,
    class_name="metal_part"
)

print(f"中心點: {box.center}")       # (200.0, 275.0)
print(f"面積: {box.area}")           # 50000
print(f"COCO 格式: {box.to_xywh()}")  # (100, 150, 200, 250)
```

---

### SegmentationMask

分割遮罩資料類別,表示單一物體的實例分割結果。

**屬性:**

| 屬性 | 型別 | 說明 |
|------|------|------|
| `mask` | np.ndarray | 二值化遮罩 (H, W) |
| `confidence` | float | 信心度 (0-1) |
| `class_id` | int | 類別 ID |
| `class_name` | str | 類別名稱 |
| `bbox` | Optional[DetectionBox] | 對應的邊界框 |

**屬性 (計算):**

| 屬性 | 型別 | 說明 |
|------|------|------|
| `area` | int | 遮罩面積 (像素數) |
| `shape` | Tuple[int, int] | 遮罩尺寸 |

**方法:**

```python
# 取得輪廓
contours = mask.get_contours()

# 取得中心點
center = mask.get_center()
```

**範例:**

```python
from src.models import SegmentationMask
import numpy as np

# 建立遮罩
mask = np.zeros((100, 100), dtype=np.uint8)
mask[30:70, 30:70] = 1  # 方形區域

seg_mask = SegmentationMask(
    mask=mask,
    confidence=0.92,
    class_id=1,
    class_name="defect"
)

print(f"面積: {seg_mask.area} 像素")  # 1600
print(f"中心: {seg_mask.get_center()}")  # (49.5, 49.5)
```

---

### DetectionResult

完整的檢測結果,包含所有檢測框與分割遮罩。

**屬性:**

| 屬性 | 型別 | 說明 |
|------|------|------|
| `boxes` | List[DetectionBox] | 檢測框列表 |
| `masks` | List[SegmentationMask] | 分割遮罩列表 |
| `inference_time` | float | 推論時間 (秒) |
| `image_shape` | Tuple[int, int] | 影像尺寸 (H, W) |
| `metadata` | Dict[str, Any] | 額外元資料 |

**屬性 (計算):**

| 屬性 | 型別 | 說明 |
|------|------|------|
| `num_detections` | int | 檢測數量 |
| `num_masks` | int | 分割遮罩數量 |

**方法:**

```python
# 過濾結果
filtered = result.filter_by_confidence(0.8)
filtered = result.filter_by_class([0, 1, 2])

# 統計資訊
classes = result.get_classes()
counts = result.get_class_counts()
```

**範例:**

```python
from src.models import DetectionResult

# 假設已取得檢測結果
result: DetectionResult = detector.detect(image)

# 基本資訊
print(f"檢測數量: {result.num_detections}")
print(f"推論時間: {result.inference_time:.3f}s")

# 類別統計
class_counts = result.get_class_counts()
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# 過濾高信心度結果
high_conf = result.filter_by_confidence(0.8)
print(f"高信心度結果: {high_conf.num_detections}")
```

---

## YOLOv11 檢測器

### 初始化

```python
from src.models import YOLOv11Detector

# 建立檢測器
detector = YOLOv11Detector(task="detect")  # 或 "segment"
```

**參數:**

- `task` (str): 任務類型
  - `"detect"`: 物體檢測 (僅邊界框)
  - `"segment"`: 實例分割 (邊界框 + 遮罩)

---

### 載入模型

```python
# 使用預訓練模型
detector.load_model("yolo11n.pt", device="cpu")

# 使用自訓練模型
detector.load_model("models/weights/best.pt", device="cuda")

# Apple Silicon (M1/M2) 使用 MPS
detector.load_model("yolo11n.pt", device="mps")
```

**參數:**

- `model_path` (str): 模型路徑或預訓練模型名稱
  - 預訓練: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
  - 自訓練: 自己的 `.pt` 模型檔案路徑
- `device` (str): 執行裝置
  - `"cpu"`: CPU (通用)
  - `"cuda"`: NVIDIA GPU
  - `"mps"`: Apple Silicon GPU
- `verbose` (bool): 是否顯示載入資訊

**模型變體:**

| 模型 | 參數量 | 速度 | 精度 | 適用場景 |
|------|--------|------|------|----------|
| yolo11n | ~2.6M | ⚡⚡⚡ | ⭐⭐ | 邊緣設備、即時應用 |
| yolo11s | ~9.4M | ⚡⚡ | ⭐⭐⭐ | 平衡效能與精度 |
| yolo11m | ~20.1M | ⚡ | ⭐⭐⭐⭐ | 高精度應用 |
| yolo11l | ~25.3M | ⚡ | ⭐⭐⭐⭐⭐ | 伺服器端推論 |
| yolo11x | ~56.9M | 🐌 | ⭐⭐⭐⭐⭐ | 最高精度需求 |

---

### 執行檢測

```python
import cv2
from src.models import YOLOv11Detector

# 載入模型
detector = YOLOv11Detector(task="detect")
detector.load_model("yolo11n.pt", device="cpu")

# 讀取影像
image = cv2.imread("test.jpg")

# 執行檢測
result = detector.detect(
    image=image,
    conf_threshold=0.25,  # 信心度閾值
    iou_threshold=0.45,   # NMS IoU 閾值
    max_det=300           # 最大檢測數量
)

# 顯示結果
print(f"檢測到 {result.num_detections} 個物體")
for box in result.boxes:
    print(f"{box.class_name}: {box.confidence:.2f}")
```

**參數:**

- `image` (np.ndarray): 輸入影像 (H, W, 3) BGR 格式
- `conf_threshold` (float): 信心度閾值,範圍 0-1
  - 較低值: 檢測更多物體,但可能有誤檢
  - 較高值: 只保留高信心度結果
- `iou_threshold` (float): NMS (Non-Maximum Suppression) IoU 閾值
  - 用於抑制重疊的檢測框
  - 較低值: 更積極抑制重疊框
  - 較高值: 允許更多重疊
- `max_det` (int): 最大檢測數量
- `verbose` (bool): 是否顯示推論詳情

---

### 執行分割

```python
# 載入分割模型
detector = YOLOv11Detector(task="segment")
detector.load_model("yolo11n-seg.pt", device="cpu")

# 執行分割
result = detector.segment(
    image=image,
    conf_threshold=0.25,
    iou_threshold=0.45
)

# 顯示結果
print(f"分割到 {result.num_masks} 個物體")
for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
    print(f"物體 {i+1}:")
    print(f"  類別: {box.class_name}")
    print(f"  信心度: {box.confidence:.2f}")
    print(f"  遮罩面積: {mask.area} 像素")
```

---

### 與相機整合

```python
from src.hardware import MockCamera
from src.models import YOLOv11Detector
from src.utils import get_visualizer

# 初始化
camera = MockCamera(mode="objects")
detector = YOLOv11Detector(task="detect")
detector.load_model("yolo11n.pt", device="cpu")
visualizer = get_visualizer()

# 連接相機
camera.connect()
camera.start_streaming()

try:
    # 取得一幀
    frame = camera.get_frame()

    # 執行檢測
    result = detector.detect(frame.rgb, conf_threshold=0.25)

    # 視覺化結果
    result_image = visualizer.draw_detection_results(
        image=frame.rgb.copy(),
        boxes=[b.to_xyxy() for b in result.boxes],
        masks=None,
        labels=[b.class_name for b in result.boxes],
        scores=[b.confidence for b in result.boxes],
        class_ids=[b.class_id for b in result.boxes]
    )

    # 儲存結果
    visualizer.save_image(result_image, "detection_result.png")

finally:
    camera.stop_streaming()
    camera.disconnect()
```

---

## 使用範例

### 範例 1: 基本檢測

```python
from src.models import YOLOv11Detector
import cv2

# 建立並載入模型
detector = YOLOv11Detector(task="detect")
detector.load_model("yolo11n.pt")

# 讀取影像
image = cv2.imread("test.jpg")

# 執行檢測
result = detector.detect(image)

# 處理結果
for box in result.boxes:
    print(f"{box.class_name}: {box.confidence:.2f}")
    print(f"  位置: {box.to_xyxy()}")
```

### 範例 2: 過濾結果

```python
# 執行檢測
result = detector.detect(image, conf_threshold=0.1)

# 過濾低信心度結果
high_conf = result.filter_by_confidence(0.5)
print(f"高信心度結果: {high_conf.num_detections}/{result.num_detections}")

# 只保留特定類別
person_only = result.filter_by_class([0])  # 假設 0 是 person 類別
print(f"人物檢測: {person_only.num_detections}")
```

### 範例 3: 實例分割

```python
# 使用分割模型
detector = YOLOv11Detector(task="segment")
detector.load_model("yolo11n-seg.pt")

# 執行分割
result = detector.segment(image)

# 處理遮罩
for mask in result.masks:
    print(f"{mask.class_name}:")
    print(f"  面積: {mask.area} 像素")
    print(f"  中心: {mask.get_center()}")

    # 取得輪廓
    contours = mask.get_contours()
    print(f"  輪廓數: {len(contours)}")
```

### 範例 4: 效能測試

```python
import time

# 執行多次檢測測試效能
num_tests = 100
times = []

for _ in range(num_tests):
    result = detector.detect(image)
    times.append(result.inference_time)

avg_time = sum(times) / len(times)
avg_fps = 1.0 / avg_time

print(f"平均推論時間: {avg_time:.3f}s")
print(f"平均 FPS: {avg_fps:.2f}")
```

---

## 整合應用

### 完整檢測流程

```python
from src.hardware import MockCamera
from src.models import YOLOv11Detector
from src.utils import setup_logger, get_visualizer

# 初始化
logger = setup_logger("DetectionSystem")
camera = MockCamera(mode="objects")
detector = YOLOv11Detector(task="detect")
visualizer = get_visualizer()

# 載入模型
detector.load_model("yolo11n.pt", device="cpu")

# 開始處理
with camera:
    while True:
        # 取得影像
        frame = camera.get_frame()

        # 執行檢測
        result = detector.detect(frame.rgb, conf_threshold=0.25)

        # 記錄結果
        logger.info(f"幀 {frame.frame_number}: {result.num_detections} 個物體")

        # 視覺化
        result_image = visualizer.draw_detection_results(
            image=frame.rgb.copy(),
            boxes=[b.to_xyxy() for b in result.boxes],
            masks=None,
            labels=[f"{b.class_name} {b.confidence:.2f}" for b in result.boxes],
            scores=[b.confidence for b in result.boxes],
            class_ids=[b.class_id for b in result.boxes]
        )

        # 顯示或儲存
        cv2.imshow("Detection", result_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

---

## 效能考量

### 裝置選擇

**CPU:**
- 適用於開發與測試
- YOLOv11n: ~0.05-0.1s/frame (10-20 FPS)
- YOLOv11s: ~0.1-0.2s/frame (5-10 FPS)

**CUDA (NVIDIA GPU):**
- 生產環境推薦
- YOLOv11n: ~0.01-0.02s/frame (50-100 FPS)
- YOLOv11s: ~0.02-0.03s/frame (30-50 FPS)

**MPS (Apple Silicon):**
- Mac 開發環境
- 效能介於 CPU 與 CUDA 之間
- YOLOv11n: ~0.02-0.05s/frame (20-50 FPS)

### 優化建議

1. **選擇適當的模型變體**
   - 即時應用: yolo11n
   - 平衡: yolo11s
   - 高精度: yolo11m/l

2. **調整閾值**
   - 提高 `conf_threshold` 可減少誤檢
   - 調整 `iou_threshold` 優化 NMS

3. **批次處理**
   - 可同時處理多張影像提升效能

4. **模型量化**
   - 使用 TensorRT 或 ONNX 轉換
   - 可大幅提升推論速度

---

## 故障排除

### Ultralytics YOLO 未安裝

**錯誤訊息:**
```
ModelLoadError: Ultralytics YOLO 未安裝
```

**解決方法:**
```bash
pip install ultralytics
```

---

### 模型下載失敗

**問題:** 首次使用預訓練模型時下載失敗

**解決方法:**
1. 檢查網路連線
2. 手動下載模型:
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
   ```
3. 將模型放在 `models/weights/` 目錄

---

### CUDA Out of Memory

**錯誤訊息:**
```
RuntimeError: CUDA out of memory
```

**解決方法:**
1. 使用較小的模型 (如 yolo11n)
2. 減少 `max_det` 參數
3. 降低輸入影像解析度
4. 使用 CPU 推論

---

### 檢測結果為空

**問題:** `result.num_detections == 0`

**可能原因:**
1. 影像中無目標物體
2. `conf_threshold` 設定過高
3. 模型未針對目標物體訓練

**解決方法:**
1. 降低 `conf_threshold` (如 0.1)
2. 使用針對應用場景訓練的模型
3. 檢查影像品質與光照

---

## 📚 相關文檔

- [硬體介面模組](HARDWARE_MODULE_SUMMARY.md)
- [工具模組使用指南](UTILS_USAGE.md)
- [相機示範圖庫](CAMERA_DEMO_GALLERY.md)
- [專案狀態](PROJECT_STATUS.md)

---

## 📁 相關檔案

- 實作: `src/models/detector_interface.py`, `src/models/yolo_detector.py`
- 測試: `tests/test_models.py`
- 示範: `scripts/demo_detector.py`

---

**更新日期**: 2026-01-19

**版本**: 1.0.0

**作者**: ORBBEC Gemini 2 工業檢測系統開發團隊
