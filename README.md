# Defect and Surface Depth Detection System

基於 ORBBEC Gemini 2 深度攝影機的工業缺陷與表面深度檢測系統

## 專案概述

這是一個使用 ORBBEC Gemini 2 結構光深度攝影機實現的工業品質檢測系統，結合 YOLOv11 深度學習模型進行物體辨識、尺寸量測與缺陷檢測。

### 核心功能
- 🔍 **物體辨識與分類** - 識別混合材質零件（金屬、塑膠、電子元件）
- 📏 **精確尺寸量測** - 長寬高、直徑、體積等 3D 量測
- 🔬 **表面缺陷檢測** - 刮痕、凹陷、裂紋、氣泡等缺陷識別
- ✅ **品質判定** - 自動化 Pass/Fail 判定與報告生成

### 技術架構
- **AI 模型**: YOLOv11 (物體檢測 + 實例分割)
- **深度攝影機**: ORBBEC Gemini 2 (RGB 1920x1080 + 深度 1280x800)
- **開發平台**: Python + PyTorch + Ultralytics
- **部署平台**:
  - PC (NVIDIA GPU RTX 3060+) - 開發與訓練
  - Jetson Orin Nano 8GB - 生產部署

### 處理效能
- PC 平台: 2-5 秒/件
- Jetson 平台: 彈性配置 (2-10 秒，依模式調整)

## 專案結構

```
Defect_and_Surface_Depth_Detection_System/
├── README.md                    # 專案說明
├── docs/                        # 文件目錄
│   └── plans/                   # 設計文檔
│       └── 2026-01-19-orbbec-gemini2-industrial-inspection-design.md
├── config/                      # 配置檔案 (待建立)
├── data/                        # 資料目錄 (待建立)
├── models/                      # 模型檔案 (待建立)
├── src/                         # 原始碼 (待建立)
├── scripts/                     # 腳本工具 (待建立)
└── tests/                       # 測試程式 (待建立)
```

## 快速開始

### 環境需求

**開發平台 (PC):**
- Ubuntu 22.04 LTS
- Python 3.10+
- NVIDIA GPU (RTX 3060+, 12GB+ VRAM)
- CUDA 12.1+ / cuDNN 8.9+

**部署平台 (Jetson):**
- Jetson Orin Nano 8GB
- JetPack 6.0+
- 散熱風扇

### 安裝步驟

```bash
# 1. Clone 專案
git clone https://github.com/a23444452/Defect_and_Surface_Depth_Detection_System.git
cd Defect_and_Surface_Depth_Detection_System

# 2. 安裝相依套件 (待建立 requirements.txt)
pip install -r requirements.txt

# 3. 安裝 OrbbecSDK
# 請參考 https://github.com/orbbec/OrbbecSDK

# 4. 下載預訓練模型
# 詳見文件說明
```

## 文件

- [完整設計文檔](docs/plans/2026-01-19-orbbec-gemini2-industrial-inspection-design.md) - 系統架構、技術細節、實作規劃

## 開發狀態

**目前階段**: 設計階段

**預估時程**: 13-20 週 (約 3-5 個月)

### 開發里程碑

- [ ] M1: 首次成功影像擷取 (週 2)
- [ ] M2: 基準資料集完成 (週 5)
- [ ] M3: 首個訓練模型 (週 8)
- [ ] M4: 量測功能驗證 (週 11)
- [ ] M5: PC 系統整合完成 (週 13)
- [ ] M6: Jetson 模型部署 (週 16)
- [ ] M7: 最佳模式選定 (週 18)
- [ ] M8: 生產環境上線 (週 20)

## 授權

待定

## 聯絡方式

如有任何問題或建議，歡迎提出 Issue。

---

**Last Updated**: 2026-01-19
