# ORBBEC Gemini 2 工業檢測系統設計文檔

**版本**: 1.0
**日期**: 2026-01-19
**狀態**: 設計階段

---

## 1. 專案概述

### 1.1 目標
開發基於 ORBBEC Gemini 2 結構光深度攝影機的工業品質檢測系統，實現混合材質零件的自動化辨識、尺寸量測與缺陷檢測。

### 1.2 核心需求
- **應用場景**: 工業生產線零件品質控制
- **檢測對象**: 混合材質零件（金屬、塑膠、電子元件等）
- **檢測項目**:
  - 物體辨識與分類
  - 精確尺寸量測（長寬高、直徑等）
  - 表面缺陷檢測（刮痕、凹陷、裂紋、氣泡等）
  - 組裝完整性驗證（零件存在性、位置正確性）
- **效能要求**:
  - PC 平台: 2-5 秒/件
  - Jetson Orin Nano: 彈性調整（2-10 秒，依功能取捨）

### 1.3 部署策略
- **開發平台**: 高階 PC + NVIDIA GPU（RTX 3060+）
- **最終部署**: NVIDIA Jetson Orin Nano 8GB
- **開發流程**: PC 開發訓練 → 模型優化 → Jetson 部署測試 → 生產環境

---

## 2. 系統架構

### 2.1 整體架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                      硬體層 (Hardware Layer)                 │
├─────────────────────────────────────────────────────────────┤
│  ORBBEC Gemini 2                工業 PC / Jetson Orin Nano   │
│  • RGB: 1920x1080 @ 30fps      • GPU: RTX 3060+ / 40 TOPS   │
│  • Depth: 1280x800 @ 30fps     • 記憶體: 16GB+ / 8GB         │
│  • 範圍: 0.15m - 10m           • 作業系統: Linux/Ubuntu      │
│  • USB 3.0 單線連接                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  資料擷取層 (Data Acquisition)               │
├─────────────────────────────────────────────────────────────┤
│  OrbbecSDK Python              影像預處理                    │
│  • 同步 RGB + 深度擷取         • RGB 正規化 (1920x1080)      │
│  • 硬體時間戳同步              • 深度去噪與孔洞填補          │
│  • 相機內外參讀取              • RGB-D 座標對齊              │
│  • IMU 資料整合                • 相機標定參數載入            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   AI 推理層 (Inference Layer)                │
├─────────────────────────────────────────────────────────────┤
│  YOLOv11 多任務模型            後處理模組                    │
│  • 物體偵測 (Detection)        • NMS 非極大值抑制            │
│  • 實例分割 (Segmentation)     • 信心度過濾                  │
│  • 關鍵點檢測 (Keypoint)       • 結果聚合與追蹤              │
│  • 旋轉邊界框 (OBB)            • 座標轉換                    │
│                                                              │
│  模型版本:                                                   │
│  • PC: YOLOv11m-seg (FP32/FP16)                             │
│  • Jetson: YOLOv11n/s-seg (FP16/INT8 + TensorRT)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              量測與判定層 (Measurement & Decision)           │
├─────────────────────────────────────────────────────────────┤
│  3D 點雲重建                   尺寸計算引擎                  │
│  • 深度圖 → 點雲轉換           • OBB 長寬高計算              │
│  • 點雲濾波去噪                • 關鍵點距離量測              │
│  • 物體分割提取                • 直徑/圓度檢測               │
│  • 座標系變換                  • 體積計算                    │
│                                                              │
│  缺陷分析                      品質判定                      │
│  • 表面粗糙度分析              • 規格資料庫比對              │
│  • 凹陷/凸起檢測               • 公差範圍驗證                │
│  • 邊緣完整性檢查              • Pass/Fail 判定              │
│  • 異常區域標記                • 缺陷嚴重度分級              │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    [檢測報告 + 可視化結果]
```

### 2.2 層次說明

#### 2.2.1 硬體層
**ORBBEC Gemini 2 規格:**
- RGB 解析度: 1920 x 1080 @ 30fps
- 深度解析度: 1280 x 800 @ 30fps (可調 5/10/15/30/60 fps)
- 工作範圍: 0.15m - 10m (建議檢測距離: 0.3m - 1m)
- 精度: 2m 處 RMSE < 2% (約 ±40mm，近距離更精確)
- 視野角度: 深度 H91° x V66°，RGB H86° x V55°
- 技術: 主動式立體紅外線 + ASIC MX6600
- 介面: USB 3.0 (單線供電 + 資料)
- 功耗: ≤ 2.5W

**建議檢測環境配置:**
- 檢測距離: 0.5m - 1m (平衡精度與視野)
- 量測精度估算:
  - 0.5m 處: ±5mm
  - 1.0m 處: ±10mm
  - 2.0m 處: ±20mm
- 照明: 穩定工業照明（雖然深度感測在黑暗中可用，但 RGB 影像仍需光源）
- 背景: 非反光、對比明顯

#### 2.2.2 資料擷取層
**主要功能:**
- 使用 OrbbecSDK Python 綁定進行硬體控制
- 同步擷取 RGB 和深度幀（硬體時間戳對齊）
- RGB 影像預處理:
  - 解析度調整: 1920x1080 → 640x640 (YOLO 輸入)
  - 正規化: [0-255] → [0-1]
  - 可選增強: 亮度、對比度調整
- 深度圖預處理:
  - 無效值處理（0 或超出範圍的深度）
  - 時域濾波（Temporal Filter）減少噪聲
  - 空間濾波（Spatial Filter）平滑深度
  - 孔洞填補（Hole Filling）處理反光/黑色物體
- RGB-D 對齊:
  - 使用 SDK 硬體對齊功能
  - 將深度圖投影到 RGB 座標系
- 相機標定:
  - 棋盤格或 ArUco 標記標定
  - 儲存內外參矩陣
  - 用於精確 3D 重建

#### 2.2.3 AI 推理層
**YOLOv11 配置:**

| 項目 | PC 開發平台 | Jetson Orin Nano |
|------|------------|------------------|
| 模型 | YOLOv11m-seg | YOLOv11n/s-seg (依測試選擇) |
| 預訓練權重 | yolo11m-seg.pt (COCO) | 從 PC 模型蒸餾 |
| 參數量 | ~25M | ~2.5M (n) / ~9M (s) |
| 輸入尺寸 | 640x640 | 640x640 |
| 推理精度 | FP32/FP16 | FP16/INT8 |
| 優化 | PyTorch | TensorRT |
| 預估推理時間 | 100-200ms | 300-800ms (依模型與量化) |

**輸出內容:**
- 邊界框 (Bounding Box): [x1, y1, x2, y2]
- 類別 (Class): 零件類型 ID
- 信心度 (Confidence): [0-1]
- 分割遮罩 (Segmentation Mask): 像素級別的物體邊界
- 關鍵點 (Keypoints, 可選): 特徵點座標

**後處理:**
- NMS (Non-Maximum Suppression): IoU 閾值 0.45
- 信心度過濾: 閾值 0.5-0.7
- 遮罩後處理: 形態學操作（開運算、閉運算）

#### 2.2.4 量測與判定層
**3D 點雲處理:**
```python
# 深度像素 → 3D 點雲轉換
X = (u - cx) * depth / fx
Y = (v - cy) * depth / fy
Z = depth
```
- 點雲濾波: 統計離群值移除 + 半徑濾波
- 物體分割: 基於 YOLO 遮罩提取
- 降採樣: VoxelGrid 減少計算量

**尺寸量測方法:**
1. **Oriented Bounding Box (OBB)**
   - PCA 找主軸方向
   - 計算最小外接長方體
   - 適合: 規則形狀零件

2. **關鍵點距離量測**
   - 檢測特徵點（孔位、邊角）
   - 3D 歐氏距離計算
   - 適合: 有明確特徵的零件

3. **直徑/圓度檢測**
   - RANSAC 圓/圓柱擬合
   - 適合: 圓形零件（螺絲、軸承）

4. **體積計算**
   - Convex Hull 或 Alpha Shape
   - 適合: 物流包裝尺寸

**缺陷檢測方法:**
- **表面缺陷**: RGB 紋理分析 + 深度變異檢測
- **幾何缺陷**: 點雲平面擬合 + 偏差計算
- **組裝缺陷**: 物體存在性 + 相對位置驗證
- **邊緣缺陷**: 輪廓平滑度 + 深度邊緣跳變

---

## 3. Jetson Orin Nano 部署策略

### 3.1 三種運行模式

| 模式 | 功能範圍 | 模型 | 預估時間 | 適用場景 |
|------|---------|------|---------|---------|
| **完整模式** | 辨識 + 分割 + 量測 + 缺陷檢測 | YOLOv11s + INT8 | 5-8 秒 | 高精度需求 |
| **快速模式** | 辨識 + 基本量測 | YOLOv11n + INT8 | 2-4 秒 | 平衡速度與精度 |
| **極速模式** | 辨識 + 良品/不良品分類 | YOLOv11n 簡化 | 1-2 秒 | 快速初篩 |

### 3.2 優化技術
- **模型量化**: FP32 → FP16 → INT8 (TensorRT)
- **知識蒸餾**: YOLOv11m → YOLOv11s/n
- **模型剪枝**: 移除冗餘通道和層
- **算子融合**: TensorRT 自動優化
- **記憶體優化**: 統一記憶體管理、批次處理優化

### 3.3 部署工作流程
```
[PC 階段]
1. 使用 YOLOv11m 完整訓練
2. 驗證精度基準
   ↓
[優化階段]
3. 知識蒸餾到輕量模型
4. 量化到 INT8
5. 導出 TensorRT Engine
   ↓
[Jetson 測試]
6. 三種模式性能測試
7. 精度 vs 速度權衡分析
8. 選擇最佳配置
   ↓
[生產部署]
9. 固化配置參數
10. 系統整合測試
```

---

## 4. 資料準備策略

### 4.1 資料收集計畫

**階段 1: 初始資料收集 (1-2 週)**
- 每種零件類型: 50-100 張影像
- 拍攝角度: 至少 8 個不同視角
- 包含樣本:
  - 良品: 60%
  - 各類缺陷品: 40%（刮痕、凹陷、尺寸不符等）
- 環境變化:
  - 多種光照條件
  - 不同背景
  - 各種擺放姿態
- 資料格式: 同步儲存 RGB + 深度圖 + 元資料

**階段 2: 合成資料生成**
使用工具:
- Blender + BlenderProc (如有 CAD 模型)
- 3D 掃描或 Photogrammetry (無 CAD)

生成內容:
- 隨機化光照、角度、位置
- 生成配對的深度圖和分割遮罩
- 目標數量: 2000-3000 張

**階段 3: 資料擴增**
訓練時動態套用:
- 幾何: 旋轉、翻轉、縮放、裁切、透視變換
- 色彩: 亮度、對比、飽和度、HSV 調整
- 噪聲: 高斯噪聲、椒鹽噪聲
- 模糊: 高斯模糊、運動模糊
- 深度專用: 深度噪聲、孔洞模擬

### 4.2 YOLO 格式資料集結構

```
industrial_parts_dataset/
├── images/
│   ├── train/              # 70% 訓練集
│   │   ├── part_0001.jpg
│   │   ├── part_0002.jpg
│   │   └── ...
│   ├── val/                # 20% 驗證集
│   │   └── ...
│   └── test/               # 10% 測試集
│       └── ...
├── labels/
│   ├── train/              # 對應標註
│   │   ├── part_0001.txt   # YOLO 格式標註
│   │   ├── part_0002.txt
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── depth/                  # 可選：深度圖資料
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml            # 資料集配置檔
```

### 4.3 標註格式

**YOLO 實例分割格式 (.txt):**
```
# 每行一個物體
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
# class_id: 類別編號
# (x, y): 多邊形頂點座標（正規化 0-1）
```

**dataset.yaml 範例:**
```yaml
path: /path/to/industrial_parts_dataset
train: images/train
val: images/val
test: images/test

nc: 15  # 類別數量
names:
  0: screw_m6
  1: screw_m8
  2: nut_hex
  3: washer
  4: metal_bracket
  5: plastic_housing
  6: pcb_board
  7: connector
  8: spring
  9: gear
  10: defect_scratch
  11: defect_dent
  12: defect_crack
  13: defect_burr
  14: defect_missing_part
```

### 4.4 資料集組成建議

**初期 MVP (最小可行產品):**
- 真實影像: 1000-2000 張 (30-40%)
- 合成影像: 2000-3000 張 (40-50%)
- 增強後總數: 5000-10000 張

**成熟階段:**
- 真實影像: 5000+ 張
- 合成影像: 3000-5000 張
- 持續更新: 加入生產環境的困難案例

---

## 5. 模型訓練與優化

### 5.1 訓練流程

**使用預訓練權重 (遷移學習):**
```python
from ultralytics import YOLO

# 載入 COCO 預訓練模型
model = YOLO('yolo11m-seg.pt')

# 微調訓練
results = model.train(
    data='dataset.yaml',
    epochs=200,
    imgsz=640,
    batch=16,              # 依 GPU 記憶體調整
    device=0,              # GPU 編號
    pretrained=True,       # 使用預訓練權重
    patience=50,           # Early stopping
    optimizer='AdamW',
    lr0=0.001,             # 初始學習率
    weight_decay=0.0005,
    augment=True,          # 啟用資料增強
    cache=True,            # 快取資料到記憶體
    workers=8,             # 資料載入執行緒
    project='runs/segment',
    name='industrial_yolo11m'
)
```

### 5.2 驗證與評估

**評估指標:**
- **物體檢測**: mAP@0.5, mAP@0.5:0.95, Precision, Recall
- **實例分割**: Mask mAP, IoU
- **推理速度**: FPS, 延遲時間
- **記憶體使用**: GPU/CPU 記憶體峰值

**驗證流程:**
```python
# 驗證模型
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Mask mAP50: {metrics.seg.map50}")

# 測試集推理
results = model.predict(
    source='test_images/',
    conf=0.5,              # 信心度閾值
    iou=0.45,              # NMS IoU 閾值
    save=True,             # 儲存結果
    save_txt=True,         # 儲存預測標註
    save_conf=True         # 包含信心度
)
```

### 5.3 模型導出

**TensorRT 優化 (用於 Jetson):**
```python
# 導出 FP16
model.export(format='engine', half=True, device=0)

# 導出 INT8 (需要校準資料)
model.export(
    format='engine',
    int8=True,
    data='dataset.yaml',  # 用於 INT8 校準
    device=0
)

# 導出 ONNX (中間格式)
model.export(format='onnx', dynamic=True)
```

### 5.4 知識蒸餾 (Teacher-Student)

```python
# Teacher: YOLOv11m (已訓練好)
teacher = YOLO('runs/segment/industrial_yolo11m/weights/best.pt')

# Student: YOLOv11n (輕量模型)
student = YOLO('yolo11n-seg.pt')

# 蒸餾訓練 (需要自定義實現或使用第三方工具)
# 目標: 讓 student 學習 teacher 的輸出分佈
```

---

## 6. 品質判定系統

### 6.1 規格資料庫

```python
# 零件規格定義
PART_SPECIFICATIONS = {
    "screw_m6": {
        "dimensions": {
            "length": {
                "nominal": 20.0,        # mm
                "upper_tolerance": 0.5,
                "lower_tolerance": -0.5,
                "critical": True
            },
            "diameter": {
                "nominal": 6.0,
                "upper_tolerance": 0.1,
                "lower_tolerance": -0.1,
                "critical": True
            },
            "head_diameter": {
                "nominal": 10.0,
                "upper_tolerance": 0.2,
                "lower_tolerance": -0.2,
                "critical": False
            }
        },
        "defects": {
            "max_scratch_area": 2.0,        # mm²
            "max_scratch_depth": 0.2,       # mm
            "max_dent_depth": 0.3,          # mm
            "max_surface_roughness": 0.5,   # mm RMS
            "allow_minor_defects": 2        # 最多容許數量
        },
        "assembly": {
            "required": True,
            "position_tolerance": 5.0,      # mm
            "angle_tolerance": 5.0          # degrees
        }
    },
    "plastic_housing": {
        # 其他零件規格...
    }
}
```

### 6.2 判定邏輯

```python
def quality_inspection(detection_result, measurement_result, spec):
    """
    綜合品質判定

    Returns:
        result: "PASS" / "FAIL"
        severity: "OK" / "MINOR" / "CRITICAL"
        details: 詳細檢查結果
    """
    checks = []

    # 1. 尺寸檢查
    for dim_name, dim_value in measurement_result['dimensions'].items():
        spec_dim = spec['dimensions'][dim_name]
        upper_limit = spec_dim['nominal'] + spec_dim['upper_tolerance']
        lower_limit = spec_dim['nominal'] + spec_dim['lower_tolerance']

        in_spec = lower_limit <= dim_value <= upper_limit
        checks.append({
            'type': 'dimension',
            'name': dim_name,
            'value': dim_value,
            'spec': f"{spec_dim['nominal']} ±{spec_dim['upper_tolerance']}",
            'pass': in_spec,
            'critical': spec_dim['critical']
        })

    # 2. 缺陷檢查
    for defect in measurement_result['defects']:
        defect_ok = defect['severity'] <= spec['defects'][f"max_{defect['type']}_depth"]
        checks.append({
            'type': 'defect',
            'name': defect['type'],
            'value': defect['severity'],
            'spec': spec['defects'][f"max_{defect['type']}_depth"],
            'pass': defect_ok,
            'critical': defect['type'] in ['crack', 'missing_part']
        })

    # 3. 組裝檢查
    if spec['assembly']['required']:
        assembly_ok = check_assembly(detection_result, spec['assembly'])
        checks.append({
            'type': 'assembly',
            'pass': assembly_ok,
            'critical': True
        })

    # 4. 綜合判定
    critical_fails = [c for c in checks if not c['pass'] and c['critical']]
    minor_fails = [c for c in checks if not c['pass'] and not c['critical']]

    if critical_fails:
        return "FAIL", "CRITICAL", checks
    elif len(minor_fails) > spec['defects']['allow_minor_defects']:
        return "FAIL", "MINOR", checks
    else:
        return "PASS", "OK", checks
```

### 6.3 檢測模式

**嚴格模式 (Strict):**
- 所有尺寸必須在公差範圍內
- 不容許任何可見缺陷
- 用於: 新品檢驗、高精度零件

**標準模式 (Standard):**
- 關鍵尺寸嚴格，非關鍵尺寸略寬鬆
- 容許少量次要缺陷
- 用於: 一般生產檢驗

**寬鬆模式 (Loose):**
- 僅檢查關鍵功能性尺寸
- 外觀缺陷容許度高
- 用於: 內部使用零件、二手零件檢驗

### 6.4 輸出報告格式

```json
{
    "inspection_id": "INS_20260119_153000_001",
    "timestamp": "2026-01-19T15:30:00.123Z",
    "part": {
        "id": "screw_m6_batch_A_001",
        "class": "screw_m6",
        "confidence": 0.95
    },
    "detection": {
        "bbox": [120, 150, 280, 310],
        "mask_area": 12450,
        "orientation": 45.2
    },
    "measurements": {
        "length": {
            "value": 19.85,
            "unit": "mm",
            "spec": "20.0 ±0.5",
            "status": "PASS"
        },
        "diameter": {
            "value": 6.03,
            "unit": "mm",
            "spec": "6.0 ±0.1",
            "status": "PASS"
        },
        "measurement_confidence": 0.92
    },
    "defects": [
        {
            "type": "scratch",
            "area": 1.2,
            "depth": 0.15,
            "severity": "minor",
            "location": [180, 220],
            "bbox": [175, 215, 185, 225]
        }
    ],
    "quality_decision": {
        "result": "PASS",
        "severity": "OK",
        "mode": "standard",
        "checks_passed": 12,
        "checks_failed": 0
    },
    "performance": {
        "total_time": 3.24,
        "inference_time": 0.18,
        "postprocess_time": 2.86,
        "measurement_time": 0.20
    },
    "images": {
        "rgb": "data/inspections/20260119/INS_001_rgb.jpg",
        "depth": "data/inspections/20260119/INS_001_depth.png",
        "annotated": "data/inspections/20260119/INS_001_result.jpg",
        "point_cloud": "data/inspections/20260119/INS_001_cloud.ply"
    }
}
```

---

## 7. 深度整合與 3D 量測

### 7.1 兩階段處理流程

```
┌─────────────────────────────────────────────────┐
│ 階段 1: RGB 影像推理                             │
├─────────────────────────────────────────────────┤
│ 輸入: RGB 影像 (1920x1080)                       │
│   ↓                                              │
│ Resize: 640x640                                  │
│   ↓                                              │
│ YOLOv11 推理                                     │
│   ↓                                              │
│ 輸出: 邊界框 + 分割遮罩 + 類別 + 信心度           │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 階段 2: 深度後處理與 3D 量測                     │
├─────────────────────────────────────────────────┤
│ 1. 遮罩上採樣到原始解析度 (1920x1080)            │
│ 2. 深度圖對齊到 RGB 座標系                       │
│ 3. 使用遮罩提取物體深度區域                      │
│ 4. 深度像素 → 3D 點雲轉換                        │
│ 5. 點雲濾波與去噪                                │
│ 6. 3D 量測 (OBB / 關鍵點距離 / 體積)            │
│ 7. 缺陷分析 (表面粗糙度 / 凹陷檢測)              │
└─────────────────────────────────────────────────┘
```

### 7.2 點雲處理範例

```python
import numpy as np
import open3d as o3d

def depth_to_pointcloud(depth_map, mask, intrinsics):
    """
    將深度圖轉換為 3D 點雲

    Args:
        depth_map: (H, W) 深度影像 (mm)
        mask: (H, W) 布林遮罩
        intrinsics: 相機內參 (fx, fy, cx, cy)

    Returns:
        points: (N, 3) 3D 點雲陣列
    """
    fx, fy, cx, cy = intrinsics
    h, w = depth_map.shape

    # 建立像素座標網格
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # 只保留遮罩內的點
    u = u[mask]
    v = v[mask]
    depth = depth_map[mask]

    # 過濾無效深度
    valid = (depth > 0) & (depth < 10000)  # 0-10m
    u = u[valid]
    v = v[valid]
    depth = depth[valid]

    # 深度 → 3D 座標轉換
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    points = np.stack([x, y, z], axis=-1)
    return points

def filter_pointcloud(points):
    """點雲濾波"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 統計離群值移除
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 半徑濾波
    pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=5.0)

    # 降採樣
    pcd = pcd.voxel_down_sample(voxel_size=1.0)  # 1mm

    return np.asarray(pcd.points)

def measure_obb_dimensions(points):
    """使用 OBB 量測物體尺寸"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 計算 Oriented Bounding Box
    obb = pcd.get_oriented_bounding_box()

    # 取得尺寸
    extent = obb.extent  # [length, width, height]
    center = obb.center
    rotation = obb.R

    return {
        'length': float(extent[0]),
        'width': float(extent[1]),
        'height': float(extent[2]),
        'center': center.tolist(),
        'rotation': rotation.tolist()
    }
```

### 7.3 表面缺陷檢測

```python
def detect_surface_defects(points, mask_2d, rgb_image):
    """
    結合 3D 點雲和 2D 影像的表面缺陷檢測

    Returns:
        defects: 缺陷清單
    """
    defects = []

    # 1. 深度異常檢測 (凹陷/凸起)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 擬合平面 (RANSAC)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=1.0,  # mm
        ransac_n=3,
        num_iterations=1000
    )

    # 計算每點到平面的距離
    points_array = np.asarray(pcd.points)
    a, b, c, d = plane_model
    distances = np.abs(a*points_array[:,0] +
                       b*points_array[:,1] +
                       c*points_array[:,2] + d) / np.sqrt(a**2 + b**2 + c**2)

    # 找出異常點 (凹陷/凸起)
    dent_threshold = 0.5  # mm
    bump_threshold = 0.3  # mm

    dents = points_array[distances < -dent_threshold]
    bumps = points_array[distances > bump_threshold]

    if len(dents) > 100:  # 最小面積閾值
        defects.append({
            'type': 'dent',
            'depth': float(np.abs(distances[distances < -dent_threshold]).max()),
            'area': len(dents),
            'severity': 'critical' if len(dents) > 1000 else 'minor'
        })

    # 2. RGB 紋理分析 (刮痕/裂紋)
    masked_rgb = rgb_image * mask_2d[..., None]
    # ... 邊緣檢測、紋理分析等

    return defects
```

---

## 8. 實作時程與里程碑

### 8.1 開發階段規劃

**第一階段: 基礎環境建置 (1-2 週)**
- [ ] PC 開發環境設定 (Python, CUDA, PyTorch, Ultralytics)
- [ ] OrbbecSDK 安裝與測試
- [ ] Gemini 2 硬體連接與基本影像擷取
- [ ] 相機標定與內參校驗
- [ ] 資料收集工作站建置

**第二階段: 資料準備 (2-3 週)**
- [ ] 初始真實資料收集 (1000+ 張)
- [ ] 標註工具選擇與設定 (Labelme/CVAT/Roboflow)
- [ ] YOLO 格式標註完成
- [ ] 合成資料產生管線建立
- [ ] 資料集劃分與驗證

**第三階段: 模型訓練 (2-3 週)**
- [ ] YOLOv11m-seg 基準模型訓練
- [ ] 驗證集評估與超參數調整
- [ ] 遷移學習效果驗證
- [ ] 精度基準建立 (mAP, IoU)
- [ ] 困難案例分析與改進

**第四階段: 深度整合與量測 (2-3 週)**
- [ ] RGB-D 對齊管線實作
- [ ] 點雲重建與濾波
- [ ] 尺寸量測算法實作 (OBB, 關鍵點)
- [ ] 量測精度驗證 (與實際尺寸比對)
- [ ] 缺陷檢測算法實作

**第五階段: 品質判定系統 (1-2 週)**
- [ ] 規格資料庫設計與實作
- [ ] 判定邏輯開發
- [ ] 報告生成與可視化
- [ ] 測試案例驗證

**第六階段: 系統整合與測試 (1-2 週)**
- [ ] 完整檢測流程整合
- [ ] 效能優化 (GPU 加速、記憶體管理)
- [ ] 穩定性測試 (長時間運行)
- [ ] 使用者介面開發 (可選)

**第七階段: Jetson 部署優化 (3-4 週)**
- [ ] Jetson Orin Nano 環境建置
- [ ] 模型量化 (FP16/INT8)
- [ ] TensorRT 轉換與優化
- [ ] 知識蒸餾 (YOLOv11m → s/n)
- [ ] 三種模式實作與測試
- [ ] 速度與精度權衡分析
- [ ] 最佳配置選擇

**第八階段: 生產部署 (1-2 週)**
- [ ] 生產環境配置
- [ ] 系統文件編寫
- [ ] 操作手冊與訓練
- [ ] 試運行與調整

**總預估時間: 13-20 週 (約 3-5 個月)**

### 8.2 關鍵里程碑

| 里程碑 | 驗收標準 | 預估時間 |
|--------|---------|---------|
| M1: 首次成功影像擷取 | 能穩定擷取 RGB + 深度影像 | 週 2 |
| M2: 基準資料集完成 | 1000+ 張標註完成的影像 | 週 5 |
| M3: 首個訓練模型 | mAP@0.5 > 0.7 | 週 8 |
| M4: 量測功能驗證 | 尺寸量測誤差 < ±5mm @ 0.5m | 週 11 |
| M5: PC 系統整合完成 | 完整檢測流程 < 5秒 | 週 13 |
| M6: Jetson 模型部署 | TensorRT 模型成功運行 | 週 16 |
| M7: 最佳模式選定 | 三種模式測試完成，選定生產配置 | 週 18 |
| M8: 生產環境上線 | 試運行穩定 24 小時無故障 | 週 20 |

---

## 9. 技術風險與應對

### 9.1 風險評估

| 風險項目 | 影響 | 機率 | 應對策略 |
|---------|-----|-----|---------|
| 反光/黑色材質深度孔洞嚴重 | 高 | 中 | 多幀平均、孔洞填補算法、調整照明 |
| Jetson 效能不足達不到 2 秒 | 高 | 中 | 接受較長時間、功能簡化、模型極致優化 |
| 小缺陷檢測精度不足 | 中 | 高 | 加入專門缺陷檢測模組、提高解析度 |
| 標註資料不足影響精度 | 中 | 中 | 增加合成資料、資料擴增、小樣本學習 |
| 不同零件類型差異大難以統一 | 中 | 中 | 分類別訓練專門模型、增加訓練資料 |
| RGB-D 對齊誤差影響量測 | 中 | 低 | 精確標定、使用硬體對齊、誤差補償 |

### 9.2 技術挑戰與解法

**挑戰 1: 混合材質的深度品質差異**
- 問題: 反光金屬、黑色塑膠可能導致深度孔洞
- 解法:
  - 使用 Gemini 2 的 HDR 模式
  - 時域濾波 (多幀融合)
  - 孔洞填補算法
  - 調整照明角度避免鏡面反射

**挑戰 2: 即時性要求與精度平衡**
- 問題: Jetson Orin Nano 算力有限
- 解法:
  - 知識蒸餾製作輕量模型
  - INT8 量化
  - TensorRT 優化
  - 提供多種運行模式選擇

**挑戰 3: 細微缺陷檢測困難**
- 問題: 小刮痕、淺凹陷可能漏檢
- 解法:
  - 局部高解析度處理
  - 結合 RGB 和深度雙重驗證
  - 異常檢測算法
  - 階段二加入專門缺陷檢測模組

**挑戰 4: 量測精度受限於深度解析度**
- 問題: 1280x800 解析度限制精度
- 解法:
  - 優化檢測距離 (0.5-1m)
  - 使用關鍵點而非整體尺寸
  - 子像素級別的邊緣檢測
  - 多次量測取平均

---

## 10. 系統可擴展性

### 10.1 未來擴展方向

**短期 (6 個月內):**
- 增加更多零件類型支援
- 提高缺陷檢測細緻度
- 加入檢測歷史資料分析
- 開發 Web 可視化介面

**中期 (1 年內):**
- RGB-D 融合網路研究與實作
- 多攝影機聯合檢測 (360° 視角)
- 即時統計與品質趨勢分析
- 與 MES/ERP 系統整合

**長期 (1 年以上):**
- 機械手臂整合 (自動分類分揀)
- 遷移學習與少樣本學習
- 聯邦學習 (多產線資料共享)
- AI 自動調整檢測標準

### 10.2 模組化設計

系統採用模組化設計，便於替換和升級：

```
├── hardware/              # 硬體抽象層
│   ├── camera_interface.py
│   └── gemini2_driver.py
├── models/                # AI 模型
│   ├── yolo_detector.py
│   └── model_loader.py
├── processing/            # 影像處理
│   ├── alignment.py
│   ├── pointcloud.py
│   └── filters.py
├── measurement/           # 量測模組
│   ├── dimension.py
│   ├── defect.py
│   └── assembly.py
├── decision/              # 判定邏輯
│   ├── spec_db.py
│   └── quality_judge.py
├── utils/                 # 工具函數
└── main.py               # 主程式
```

---

## 11. 效能指標與驗收標準

### 11.1 PC 平台驗收標準

| 指標 | 目標值 | 測試方法 |
|------|--------|---------|
| 檢測準確率 (mAP@0.5) | > 0.85 | 測試集評估 |
| 分割 IoU | > 0.75 | 測試集評估 |
| 尺寸量測誤差 @ 0.5m | < ±5mm | 與游標卡尺比對 (20 個樣本) |
| 尺寸量測誤差 @ 1m | < ±10mm | 與游標卡尺比對 (20 個樣本) |
| 缺陷檢出率 | > 0.90 | 人工標註對照 (100 個樣本) |
| 誤報率 (False Positive) | < 0.05 | 良品誤判為不良品比例 |
| 漏報率 (False Negative) | < 0.10 | 不良品漏判為良品比例 |
| 處理時間 | 2-5 秒 | 單一零件檢測 |
| 系統穩定性 | 連續 8 小時無故障 | 壓力測試 |

### 11.2 Jetson Orin Nano 驗收標準

| 模式 | mAP@0.5 | 尺寸誤差 | 處理時間 | 記憶體使用 |
|------|---------|---------|---------|-----------|
| 完整模式 | > 0.80 | < ±8mm | 5-8 秒 | < 6GB |
| 快速模式 | > 0.75 | < ±10mm | 2-4 秒 | < 5GB |
| 極速模式 | > 0.70 | 不量測 | 1-2 秒 | < 4GB |

---

## 12. 參考資源

### 12.1 硬體文件
- [ORBBEC Gemini 2 產品頁](https://www.orbbec.com/products/stereo-vision-camera/gemini-2/)
- [ORBBEC SDK GitHub](https://github.com/orbbec/OrbbecSDK)
- [Jetson Orin Nano 開發指南](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit)

### 12.2 軟體框架
- [Ultralytics YOLOv11 文件](https://docs.ultralytics.com/)
- [Open3D 點雲處理](http://www.open3d.org/docs/)
- [TensorRT 優化指南](https://docs.nvidia.com/deeplearning/tensorrt/)

### 12.3 相關論文
- YOLOv11: [https://arxiv.org/abs/2409.xxxxx](待發表)
- RGB-D Object Detection: 相關綜述論文
- Industrial Defect Detection: AOI 檢測相關文獻

---

## 附錄 A: 開發環境配置清單

### PC 開發平台
```
硬體:
- CPU: Intel i7/i9 或 AMD Ryzen 7/9
- GPU: NVIDIA RTX 3060 以上 (12GB+ VRAM)
- RAM: 32GB+
- 儲存: 500GB+ SSD

軟體:
- OS: Ubuntu 22.04 LTS
- Python: 3.10+
- CUDA: 12.1+
- cuDNN: 8.9+
- PyTorch: 2.1+
- Ultralytics: 8.1+
- Open3D: 0.18+
- OrbbecSDK: 最新版
```

### Jetson Orin Nano
```
硬體:
- Jetson Orin Nano 8GB
- 散熱風扇 (必須)
- 64GB+ microSD (建議 NVMe SSD)

軟體:
- JetPack: 6.0+
- Python: 3.10
- PyTorch: 2.1 (ARM 版本)
- TensorRT: 8.6+
- ONNX Runtime: 1.16+
- OrbbecSDK: ARM 版本
```

---

## 附錄 B: 專案目錄結構

```
orbbec_industrial_inspection/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── camera_config.yaml
│   ├── model_config.yaml
│   └── spec_database.json
├── data/
│   ├── raw/
│   ├── processed/
│   └── datasets/
├── models/
│   ├── weights/
│   ├── onnx/
│   └── trt/
├── src/
│   ├── hardware/
│   ├── models/
│   ├── processing/
│   ├── measurement/
│   ├── decision/
│   └── utils/
├── scripts/
│   ├── train.py
│   ├── export.py
│   ├── calibrate.py
│   └── inference.py
├── tests/
│   ├── test_camera.py
│   ├── test_model.py
│   └── test_measurement.py
├── docs/
│   ├── plans/
│   ├── api/
│   └── user_manual/
└── outputs/
    ├── inspections/
    ├── logs/
    └── reports/
```

---

## 變更紀錄

| 版本 | 日期 | 變更內容 | 作者 |
|------|------|---------|------|
| 1.0 | 2026-01-19 | 初始設計文檔 | Claude Sonnet 4.5 |

---

**文檔結束**
