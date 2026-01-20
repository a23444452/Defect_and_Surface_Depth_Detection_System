# 模型訓練指南

**系統**: ORBBEC Gemini 2 工業檢測系統
**任務**: 物件檢測、缺陷識別、實例分割
**日期**: 2026-01-20

---

## 📋 目錄

1. [訓練準備](#訓練準備)
2. [資料收集與標註](#資料收集與標註)
3. [資料集準備](#資料集準備)
4. [模型訓練](#模型訓練)
5. [模型評估](#模型評估)
6. [模型優化](#模型優化)
7. [最佳實踐](#最佳實踐)

---

## 🎯 訓練準備

### 硬體需求

| 項目 | 最低配置 | 建議配置 |
|------|----------|----------|
| **GPU** | GTX 1080 Ti (11GB) | RTX 3090/4090 (24GB) |
| **記憶體** | 16GB | 32GB+ |
| **儲存** | 100GB SSD | 500GB+ NVMe SSD |
| **CPU** | 4核心 | 8核心+ |

### 軟體需求

```bash
# Python 環境
Python 3.8+
PyTorch 2.0+
CUDA 11.8+ / cuDNN 8.9+

# 核心套件
ultralytics  # YOLOv11
opencv-python
numpy
pyyaml
```

### 環境設置

```bash
# 建立虛擬環境
python -m venv venv_training
source venv_training/bin/activate  # Linux/Mac
# venv_training\Scripts\activate  # Windows

# 安裝依賴
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python pyyaml numpy

# 驗證安裝
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📸 資料收集與標註

### 1. 資料收集

使用資料收集工具:

```python
from training.tools.data_collector import DataCollector
from src.hardware import MockCamera  # 或實體相機

# 建立收集器
collector = DataCollector(
    output_dir="outputs/datasets",
    dataset_name="industrial_parts"
)

# 從相機收集
camera = MockCamera(mode="objects")
collector.collect_from_camera(
    camera,
    num_samples=1000,
    interval=1.0,
    auto_label=False  # 手動標註
)

# 儲存元資料
collector.save_metadata()
collector.split_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
```

### 2. 資料標註

#### 使用 Label Studio (推薦)

```bash
# 安裝 Label Studio
pip install label-studio

# 啟動
label-studio

# 瀏覽器開啟: http://localhost:8080
```

**標註流程**:
1. 建立專案 → 選擇 "Object Detection with Bounding Boxes"
2. 匯入影像
3. 定義類別標籤
4. 開始標註
5. 匯出為 YOLO/COCO 格式

#### 使用其他工具

- **CVAT**: https://cvat.ai/ (支援線上與本地)
- **Roboflow**: https://roboflow.com/ (雲端平台)
- **LabelImg**: https://github.com/HumanSignal/labelImg (簡單離線工具)

### 3. 標註品質控管

```python
# 檢查標註品質
from training.tools.annotation_converter import AnnotationConverter

converter = AnnotationConverter()

# 視覺化檢查
import cv2
image = cv2.imread("path/to/image.jpg")
annotation = {...}  # 載入標註
vis_image = converter.visualize_annotation(image, annotation)
cv2.imshow("Annotation", vis_image)
```

**檢查清單**:
- [ ] 邊界框是否緊密貼合物件
- [ ] 類別標籤是否正確
- [ ] 是否有遺漏的物件
- [ ] 是否有重疊的邊界框
- [ ] 小物件是否被標註

---

## 📊 資料集準備

### 資料集結構

```
datasets/industrial_parts/
├── images/
│   ├── train/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── 000001.txt
│   │   ├── 000002.txt
│   │   └── ...
│   ├── val/
│   └── test/
└── data.yaml
```

### data.yaml 配置

```yaml
# data.yaml
path: /path/to/datasets/industrial_parts
train: images/train
val: images/val
test: images/test

# 類別
nc: 10  # 類別數
names:
  - screw_m3
  - screw_m6
  - nut_hex
  - washer
  - pcb_board
  - connector
  - bracket
  - defect_dent
  - defect_crack
  - defect_scratch
```

### 資料增強策略

**幾何變換**:
- ✅ 水平翻轉 (50%)
- ✅ 輕微旋轉 (±5°)
- ✅ 縮放 (0.5-1.5x)
- ✅ 平移 (±10%)
- ❌ 垂直翻轉 (工業場景不常見)
- ❌ 大角度旋轉 (相機通常固定)

**顏色變換**:
- ✅ 亮度調整 (±20%)
- ✅ 對比度調整 (±20%)
- ✅ HSV 增強
- ✅ 高斯噪聲 (模擬相機噪聲)

**進階增強**:
- ✅ Mosaic (拼接 4 張影像)
- ✅ Mixup (混合 2 張影像)
- ❌ CutOut (可能移除關鍵特徵)

---

## 🚀 模型訓練

### 1. 配置訓練參數

編輯 `training/configs/yolo_training.yaml`:

```yaml
# 快速測試配置
training:
  epochs: 50
  batch_size: 32
  lr0: 0.01
  device: "0"  # GPU 0

# 完整訓練配置
training:
  epochs: 300
  batch_size: 16
  lr0: 0.01
  device: "0"
  patience: 50
```

### 2. 執行訓練

```bash
# 從頭訓練
python training/scripts/train_yolo.py \
    --config training/configs/yolo_training.yaml

# 從預訓練權重微調
python training/scripts/train_yolo.py \
    --config training/configs/yolo_training.yaml \
    --weights yolo11n.pt

# 恢復訓練
python training/scripts/train_yolo.py \
    --config training/configs/yolo_training.yaml \
    --resume

# 使用 CPU 訓練 (測試)
python training/scripts/train_yolo.py \
    --config training/configs/yolo_training.yaml \
    --device cpu
```

### 3. 監控訓練

#### TensorBoard

```bash
# 啟動 TensorBoard
tensorboard --logdir outputs/tensorboard

# 瀏覽器開啟: http://localhost:6006
```

#### Weights & Biases (進階)

```python
# 在 yolo_training.yaml 中啟用
logging:
  wandb: true
  wandb_project: "industrial-inspection"
  wandb_name: "yolo11n_v1"
```

### 4. 訓練曲線解讀

**正常訓練**:
```
Epoch   Loss    mAP@0.5    mAP@0.5:0.95
1/300   8.234   0.123      0.056
50/300  3.456   0.678      0.456
100/300 2.123   0.845      0.678
200/300 1.456   0.923      0.789
300/300 1.234   0.945      0.812
```

**過擬合跡象**:
- 訓練 Loss 持續下降
- 驗證 Loss 開始上升
- 訓練 mAP >> 驗證 mAP

**欠擬合跡象**:
- 訓練 Loss 很高
- 驗證 Loss 很高
- mAP 很低 (< 0.5)

---

## 📈 模型評估

### 1. 評估指標

```python
from ultralytics import YOLO

# 載入模型
model = YOLO("outputs/training/yolo11n_industrial/weights/best.pt")

# 驗證
metrics = model.val()

print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

### 2. 各類別表現

```python
# 查看各類別 mAP
for i, name in enumerate(model.names.values()):
    ap50 = metrics.box.maps[i]
    print(f"{name:20s}: {ap50:.4f}")
```

### 3. 混淆矩陣

```python
# 繪製混淆矩陣
model.val(plots=True)
# 查看: outputs/training/.../confusion_matrix.png
```

### 4. 錯誤分析

**常見錯誤類型**:
- **False Positive**: 誤檢測 (降低信心度閾值)
- **False Negative**: 漏檢測 (增加資料, 改善標註)
- **錯誤分類**: 類別混淆 (增加困難樣本, 重新標註)
- **定位不準**: bbox 偏移 (檢查標註品質)

---

## ⚡ 模型優化

### 1. 超參數調優

使用 Ray Tune 或手動調整:

```yaml
# 學習率
lr0: [0.001, 0.01, 0.1]  # 嘗試不同學習率

# Batch size
batch_size: [8, 16, 32]  # 根據 GPU 記憶體

# 增強強度
mosaic: [0.5, 0.8, 1.0]
```

### 2. 模型架構選擇

| 模型 | 參數量 | mAP | 速度 | 建議使用 |
|------|--------|-----|------|----------|
| **YOLOv11n** | 2.6M | 基準 | 最快 | ⭐️ Jetson 部署 |
| **YOLOv11s** | 9.4M | +5% | 中等 | 平衡效能 |
| **YOLOv11m** | 20.1M | +8% | 較慢 | 高精度需求 |
| **YOLOv11l** | 25.3M | +10% | 慢 | 僅雲端推理 |

### 3. 後處理優化

```python
# 調整 NMS 參數
model.predict(
    source="test.jpg",
    conf=0.25,  # 信心度閾值 (↓ = 更多檢測)
    iou=0.45,   # NMS IOU 閾值 (↓ = 更少重疊)
    max_det=300 # 最大檢測數
)
```

### 4. 模型剪枝與量化

```python
# 匯出為 ONNX (FP16)
model.export(format="onnx", half=True)

# 匯出為 TensorRT
model.export(format="engine", half=True, workspace=4)

# INT8 量化 (需要校準資料)
model.export(
    format="engine",
    int8=True,
    data="data.yaml"
)
```

---

## 💡 最佳實踐

### 資料收集

✅ **DO**:
- 收集多樣化的資料 (不同光照、角度、背景)
- 包含困難樣本 (小物件、遮擋、模糊)
- 平衡各類別的樣本數量
- 收集邊界案例 (接近但不是缺陷的樣本)

❌ **DON'T**:
- 只收集理想條件下的資料
- 忽略罕見類別
- 使用過度增強導致不真實的樣本

### 標註品質

✅ **DO**:
- 使用一致的標註標準
- 定期審查標註品質
- 多人標註並交叉驗證
- 記錄困難案例的標註規則

❌ **DON'T**:
- 邊界框過大或過小
- 不一致的類別定義
- 忽略小物件或部分遮擋的物件

### 訓練策略

✅ **DO**:
- 從預訓練模型開始 (遷移學習)
- 使用混合精度訓練 (AMP)
- 監控訓練曲線, 及早發現問題
- 保存多個檢查點
- 在驗證集上選擇最佳模型

❌ **DON'T**:
- 過度依賴單一指標 (mAP)
- 忽略驗證集表現
- batch size 過小 (< 8)
- 過早停止訓練

### 部署優化

✅ **DO**:
- 使用 FP16/INT8 量化
- 轉換為 TensorRT 引擎
- 測試實際推理速度
- 建立完整的測試集

❌ **DON'T**:
- 僅在訓練集上評估
- 忽略推理速度
- 未測試邊界案例

---

## 📊 訓練時程參考

### 小規模資料集 (< 1000 張)

| 階段 | 時長 | 說明 |
|------|------|------|
| 資料收集 | 1-2 天 | 收集 500-1000 張影像 |
| 標註 | 2-3 天 | 每張 2-5 分鐘 |
| 訓練 | 2-4 小時 | 100-300 epochs, RTX 3090 |
| 評估優化 | 1-2 天 | 錯誤分析, 超參數調整 |
| **總計** | **1-2 週** | 端到端流程 |

### 中規模資料集 (1000-10000 張)

| 階段 | 時長 | 說明 |
|------|------|------|
| 資料收集 | 1-2 週 | 收集 5000-10000 張影像 |
| 標註 | 2-4 週 | 可考慮外包或工具輔助 |
| 訓練 | 1-2 天 | 300-500 epochs, 多 GPU |
| 評估優化 | 1 週 | 完整評估與調優 |
| **總計** | **1-2 個月** | 端到端流程 |

---

## 🔗 參考資源

### 官方文檔
- [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- [PyTorch](https://pytorch.org/docs/stable/)
- [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/)

### 標註工具
- [Label Studio](https://labelstud.io/)
- [CVAT](https://cvat.ai/)
- [Roboflow](https://roboflow.com/)

### 學習資源
- [YOLO 論文](https://arxiv.org/abs/2304.00501)
- [深度學習最佳實踐](https://github.com/google/eng-practices)

---

**文檔版本**: 1.0
**最後更新**: 2026-01-20
**作者**: Claude Sonnet 4.5 + Happy
