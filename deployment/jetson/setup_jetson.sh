#!/bin/bash
##############################################################################
# Jetson Orin Nano 環境設置腳本
# 用於在 Jetson 平台上設置工業檢測系統
##############################################################################

set -e  # 遇到錯誤立即退出

echo "========================================================================"
echo "  Jetson Orin Nano 環境設置"
echo "========================================================================"

# 檢查是否在 Jetson 平台
if [ ! -f /etc/nv_tegra_release ]; then
    echo "警告: 未檢測到 Jetson 平台"
    echo "此腳本設計用於 Jetson Orin Nano"
    read -p "是否繼續? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 顯示 Jetson 資訊
echo ""
echo "Jetson 平台資訊:"
if [ -f /etc/nv_tegra_release ]; then
    cat /etc/nv_tegra_release
fi

# 檢查 JetPack 版本
if command -v jetson_release &> /dev/null; then
    echo ""
    echo "JetPack 版本:"
    jetson_release -v
fi

# 1. 系統更新
echo ""
echo "步驟 1: 系統更新"
echo "----------------------------------------"
sudo apt-get update
sudo apt-get upgrade -y

# 2. 安裝基礎套件
echo ""
echo "步驟 2: 安裝基礎套件"
echo "----------------------------------------"
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    nano \
    htop \
    jtop

# 3. 安裝 Python 依賴
echo ""
echo "步驟 3: 安裝 Python 依賴"
echo "----------------------------------------"

# 升級 pip
python3 -m pip install --upgrade pip

# NumPy (使用針對 ARM 優化的版本)
echo "安裝 NumPy..."
python3 -m pip install numpy==1.24.3

# OpenCV (Jetson 通常預裝,但確保版本)
echo "檢查 OpenCV..."
python3 -c "import cv2; print(f'OpenCV 版本: {cv2.__version__}')" || \
    python3 -m pip install opencv-python

# PyTorch (使用 NVIDIA 提供的 Jetson 專用版本)
echo ""
echo "步驟 4: 安裝 PyTorch (Jetson 版本)"
echo "----------------------------------------"
echo "請根據 JetPack 版本安裝對應的 PyTorch wheel"
echo "參考: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
echo ""
echo "JetPack 5.x (Python 3.8):"
echo "  wget https://nvidia.box.com/shared/static/[...].whl"
echo "  pip3 install torch-*.whl"
echo ""
read -p "是否已手動安裝 PyTorch? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "請先安裝 PyTorch for Jetson"
    echo "然後重新執行此腳本"
    exit 1
fi

# 驗證 PyTorch
python3 -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"

# 5. 安裝 torchvision
echo ""
echo "步驟 5: 安裝 torchvision"
echo "----------------------------------------"
# 需要與 PyTorch 版本匹配
python3 -m pip install torchvision

# 6. 安裝 ONNX Runtime (GPU 版本)
echo ""
echo "步驟 6: 安裝 ONNX Runtime"
echo "----------------------------------------"
# Jetson 使用特殊的 ONNX Runtime 版本
python3 -m pip install onnxruntime-gpu

# 7. 安裝 TensorRT (通常隨 JetPack 預裝)
echo ""
echo "步驟 7: 檢查 TensorRT"
echo "----------------------------------------"
python3 -c "import tensorrt as trt; print(f'TensorRT 版本: {trt.__version__}')" || \
    echo "TensorRT 未安裝或未正確配置"

# 8. 安裝專案依賴
echo ""
echo "步驟 8: 安裝專案依賴"
echo "----------------------------------------"
cd "$(dirname "$0")/../.."
if [ -f requirements.txt ]; then
    echo "從 requirements.txt 安裝依賴..."
    python3 -m pip install -r requirements.txt
fi

# 9. 設置 swap (提高記憶體)
echo ""
echo "步驟 9: 設置 swap 空間"
echo "----------------------------------------"
SWAP_SIZE=8G
if [ ! -f /swapfile ]; then
    echo "建立 ${SWAP_SIZE} swap 檔案..."
    sudo fallocate -l $SWAP_SIZE /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile

    # 永久啟用
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

    echo "Swap 已設置完成"
else
    echo "Swap 檔案已存在"
fi

# 顯示 swap 狀態
free -h

# 10. 優化 Jetson 效能模式
echo ""
echo "步驟 10: 設置效能模式"
echo "----------------------------------------"
if command -v nvpmodel &> /dev/null; then
    echo "設置為最大效能模式..."
    sudo nvpmodel -m 0

    echo "設置風扇為最大轉速..."
    sudo jetson_clocks

    echo "當前效能模式:"
    sudo nvpmodel -q
else
    echo "nvpmodel 命令未找到,跳過效能設置"
fi

# 11. 建立測試腳本
echo ""
echo "步驟 11: 建立測試腳本"
echo "----------------------------------------"
cat > test_jetson_setup.py << 'EOF'
#!/usr/bin/env python3
"""
測試 Jetson 環境設置
"""

import sys

def test_imports():
    """測試關鍵套件導入"""
    print("\n測試套件導入:")
    print("-" * 50)

    tests = [
        ("NumPy", "import numpy as np; print(f'  NumPy {np.__version__}')"),
        ("OpenCV", "import cv2; print(f'  OpenCV {cv2.__version__}')"),
        ("PyTorch", "import torch; print(f'  PyTorch {torch.__version__}')"),
        ("CUDA", "import torch; print(f'  CUDA 可用: {torch.cuda.is_available()}')"),
        ("cuDNN", "import torch; print(f'  cuDNN 版本: {torch.backends.cudnn.version()}')"),
        ("TensorRT", "import tensorrt as trt; print(f'  TensorRT {trt.__version__}')"),
        ("ONNX Runtime", "import onnxruntime as ort; print(f'  ONNX Runtime {ort.__version__}')"),
    ]

    failed = []
    for name, code in tests:
        try:
            exec(code)
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)

    if failed:
        print(f"\n失敗的套件: {', '.join(failed)}")
        return False
    else:
        print(f"\n✓ 所有套件測試通過")
        return True

def test_gpu():
    """測試 GPU"""
    print("\n測試 GPU:")
    print("-" * 50)

    try:
        import torch

        if not torch.cuda.is_available():
            print("  ✗ CUDA 不可用")
            return False

        print(f"  GPU 數量: {torch.cuda.device_count()}")
        print(f"  GPU 名稱: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # 測試張量運算
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)

        print(f"  ✓ GPU 運算測試通過")
        return True

    except Exception as e:
        print(f"  ✗ GPU 測試失敗: {e}")
        return False

def main():
    """主函數"""
    print("=" * 50)
    print("  Jetson 環境設置測試")
    print("=" * 50)

    results = []
    results.append(("套件導入", test_imports()))
    results.append(("GPU", test_gpu()))

    print("\n" + "=" * 50)
    print("  測試結果")
    print("=" * 50)

    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n✓ Jetson 環境設置完成!")
        return 0
    else:
        print("\n✗ 部分測試失敗,請檢查設置")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x test_jetson_setup.py

# 12. 執行測試
echo ""
echo "步驟 12: 執行環境測試"
echo "----------------------------------------"
python3 test_jetson_setup.py

# 完成
echo ""
echo "========================================================================"
echo "  ✓ Jetson Orin Nano 環境設置完成!"
echo "========================================================================"
echo ""
echo "建議:"
echo "  1. 重新啟動系統以確保所有設置生效"
echo "  2. 使用 jtop 監控系統資源: sudo jtop"
echo "  3. 查看效能模式: sudo nvpmodel -q"
echo "  4. 調整風扇: sudo jetson_clocks"
echo ""
