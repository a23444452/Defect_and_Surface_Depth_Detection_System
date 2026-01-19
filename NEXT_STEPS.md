# 下一步行動計畫

**最後更新**: 2026-01-19 20:30
**當前狀態**: 設計階段完成，準備開始實作

---

## ✅ 已完成

### 1. 專案初始化
- [x] 創建專案資料夾結構
- [x] 建立 Git 倉庫並連結到 GitHub
- [x] 完成完整的系統設計文檔（35KB+）
- [x] 撰寫 README.md 和 .gitignore

### 2. 設計文檔內容
- [x] 四層系統架構設計（硬體、資料擷取、AI 推理、量測判定）
- [x] YOLOv11 模型配置與訓練策略
- [x] 雙平台部署方案（PC + Jetson Orin Nano）
- [x] 資料準備與 YOLO 格式標註流程
- [x] 3D 點雲量測與缺陷檢測算法
- [x] 品質判定系統設計
- [x] 實作時程規劃（13-20 週）

---

## 📋 下一步行動（優先順序排序）

### 階段 1: 專案結構建立（1-2 小時）

**目標**: 建立完整的專案目錄結構和基礎配置檔

#### 行動項目：

1. **建立專案目錄結構**
   ```bash
   cd ~/Defect_and_Surface_Depth_Detection_System
   mkdir -p {config,data/{raw,processed,datasets},models/{weights,onnx,trt},src/{hardware,models,processing,measurement,decision,utils},scripts,tests,outputs/{inspections,logs,reports}}
   touch data/raw/.gitkeep data/processed/.gitkeep models/weights/.gitkeep
   ```

2. **建立 requirements.txt**
   - 列出所有 Python 套件需求
   - 包含: ultralytics, torch, open3d, opencv-python, numpy, pyyaml 等
   - 分開 PC 和 Jetson 的需求（如果有差異）

3. **建立配置檔範本**
   - `config/camera_config.yaml` - 相機參數配置
   - `config/model_config.yaml` - 模型配置
   - `config/spec_database.json` - 零件規格資料庫範本

4. **提交到 Git**
   ```bash
   git add .
   git commit -m "Add project structure and configuration templates"
   git push
   ```

---

### 階段 2: 開發環境準備（預估 1 天）

**目標**: 設定 PC 開發環境，確保所有工具可用

#### 行動項目：

1. **安裝基礎環境**
   - [ ] 確認 Python 3.10+ 已安裝
   - [ ] 確認 CUDA 12.1+ 和 cuDNN 8.9+ 已安裝
   - [ ] 安裝 PyTorch 2.1+ (with CUDA support)
   - [ ] 安裝 Ultralytics YOLOv11

2. **安裝 OrbbecSDK**
   - [ ] 從 GitHub 下載: https://github.com/orbbec/OrbbecSDK
   - [ ] 安裝 Python 綁定
   - [ ] 測試相機連接（如果硬體已到位）

3. **安裝其他工具**
   - [ ] Open3D (點雲處理)
   - [ ] OpenCV (影像處理)
   - [ ] 標註工具選擇（Labelme / CVAT / Roboflow）

4. **環境驗證腳本**
   - 建立 `scripts/check_environment.py` 驗證所有套件可用

---

### 階段 3: 基礎程式開發（預估 3-5 天）

**目標**: 實作核心模組的基礎框架

#### 優先順序 1: 相機介面模組

建立 `src/hardware/camera_interface.py`:
- 抽象相機介面類別
- 擷取 RGB 和深度影像
- 影像對齊功能
- 相機參數讀取

建立 `src/hardware/gemini2_driver.py`:
- OrbbecSDK 包裝器
- 初始化和關閉
- 同步擷取功能

#### 優先順序 2: 測試腳本

建立 `scripts/test_camera.py`:
- 測試相機連接
- 顯示 RGB 和深度影像
- 儲存測試影像

#### 優先順序 3: 模型載入器

建立 `src/models/yolo_detector.py`:
- 載入 YOLOv11 預訓練模型
- 推理介面
- 結果解析

建立 `scripts/test_yolo.py`:
- 測試模型載入
- 使用範例影像推理

---

### 階段 4: 資料收集準備（如果硬體已到位）

**目標**: 開始收集第一批訓練資料

#### 行動項目：

1. **建立資料收集腳本**
   - `scripts/collect_data.py`
   - 自動擷取並儲存 RGB + 深度影像
   - 加上時間戳和元資料

2. **設定標註工具**
   - 選擇並安裝標註工具
   - 建立標註指南
   - 定義類別清單

3. **開始收集樣本**
   - 每種零件 50-100 張
   - 不同角度、光照、背景
   - 包含良品和缺陷品

---

## 🎯 立即可執行的任務（下次對話開始）

**建議從這裡開始：**

1. **執行階段 1** - 建立專案結構（最快速，立即可完成）
2. **建立 requirements.txt** - 列出所有需要的 Python 套件
3. **建立配置檔範本** - camera_config.yaml, model_config.yaml

**指令範例：**
```bash
# 可以直接執行
cd ~/Defect_and_Surface_Depth_Detection_System
mkdir -p config data/raw data/processed data/datasets models/weights models/onnx models/trt
mkdir -p src/hardware src/models src/processing src/measurement src/decision src/utils
mkdir -p scripts tests outputs/inspections outputs/logs outputs/reports
touch data/raw/.gitkeep data/processed/.gitkeep models/weights/.gitkeep
```

---

## 📝 重要決策待確認

在下次對話時可能需要確認：

1. **是否已有 ORBBEC Gemini 2 硬體？**
   - 有 → 優先開發相機介面和資料收集
   - 沒有 → 優先使用公開資料集或模擬資料進行開發

2. **PC 開發環境配置如何？**
   - GPU 型號和記憶體
   - CUDA 版本
   - 是否已安裝 PyTorch

3. **是否有現成的零件樣本可以開始拍攝？**
   - 有 → 立即開始資料收集
   - 沒有 → 先開發框架，使用 COCO 資料集測試

---

## 📚 參考連結

- [完整設計文檔](docs/plans/2026-01-19-orbbec-gemini2-industrial-inspection-design.md)
- [GitHub 專案](https://github.com/a23444452/Defect_and_Surface_Depth_Detection_System)
- [OrbbecSDK](https://github.com/orbbec/OrbbecSDK)
- [Ultralytics YOLOv11](https://docs.ultralytics.com/)

---

## 💡 提示

下次對話可以這樣開始：
- 「繼續上次的專案，執行下一步」
- 「開始建立專案結構」
- 「我的硬體環境是...，我們從哪開始？」
