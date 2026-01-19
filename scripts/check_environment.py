#!/usr/bin/env python3
"""
環境驗證腳本
檢查所有必要的依賴項目是否正確安裝
"""

import sys
import platform
from typing import Dict, List, Tuple


def print_header(text: str):
    """列印標題"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_status(name: str, status: bool, version: str = "", message: str = ""):
    """列印檢查狀態"""
    status_symbol = "✓" if status else "✗"
    status_text = "已安裝" if status else "未安裝"

    if version:
        print(f"  {status_symbol} {name:30s} {status_text:10s} v{version}")
    else:
        print(f"  {status_symbol} {name:30s} {status_text:10s}")

    if message:
        print(f"      → {message}")


def check_python_version() -> Tuple[bool, str]:
    """檢查 Python 版本"""
    version = platform.python_version()
    major, minor = sys.version_info[:2]
    is_valid = major == 3 and minor >= 10
    return is_valid, version


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """檢查 Python 套件是否安裝"""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, ""


def check_cuda() -> Tuple[bool, str]:
    """檢查 CUDA 是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return True, f"{version} ({device_count} GPU: {device_name})"
        else:
            return False, "CUDA 不可用"
    except ImportError:
        return False, "PyTorch 未安裝"


def check_tensorrt() -> Tuple[bool, str]:
    """檢查 TensorRT 是否可用（Jetson 平台）"""
    try:
        import tensorrt
        version = tensorrt.__version__
        return True, version
    except ImportError:
        return False, "未安裝（僅 Jetson 需要）"


def check_orbbec_sdk() -> Tuple[bool, str]:
    """檢查 OrbbecSDK 是否安裝"""
    try:
        import pyorbbecsdk as orbbec
        # OrbbecSDK 可能沒有 __version__ 屬性
        return True, "已安裝"
    except ImportError:
        return False, "需要手動安裝"


def main():
    """主函數"""
    print_header("ORBBEC Gemini 2 工業檢測系統 - 環境檢查")

    # 系統資訊
    print_header("系統資訊")
    print(f"  作業系統: {platform.system()} {platform.release()}")
    print(f"  處理器: {platform.processor()}")
    print(f"  Python 版本: {platform.python_version()}")

    # Python 版本檢查
    print_header("Python 版本檢查")
    is_valid, version = check_python_version()
    print_status("Python 3.10+", is_valid, version,
                 "需要 Python 3.10 或更高版本" if not is_valid else "")

    # 核心依賴項目
    print_header("核心依賴項目")

    core_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("ultralytics", "Ultralytics (YOLOv11)"),
        ("cv2", "OpenCV", "opencv-python"),
        ("open3d", "Open3D"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML", "pyyaml"),
    ]

    core_status = []
    for pkg_info in core_packages:
        import_name = pkg_info[0]
        display_name = pkg_info[1]
        package_name = pkg_info[2] if len(pkg_info) > 2 else pkg_info[0]

        is_installed, version = check_package(package_name, import_name)
        print_status(display_name, is_installed, version)
        core_status.append(is_installed)

    # CUDA 檢查
    print_header("CUDA 支援")
    cuda_available, cuda_info = check_cuda()
    print_status("CUDA", cuda_available, "", cuda_info)

    # TensorRT 檢查（可選）
    print_header("TensorRT 支援（Jetson 平台）")
    trt_available, trt_version = check_tensorrt()
    print_status("TensorRT", trt_available, trt_version)

    # OrbbecSDK 檢查
    print_header("硬體驅動")
    orbbec_available, orbbec_info = check_orbbec_sdk()
    print_status("OrbbecSDK", orbbec_available, "",
                 "請參考: https://github.com/orbbec/pyorbbecsdk" if not orbbec_available else "")

    # 其他重要套件
    print_header("其他重要套件")

    other_packages = [
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("PIL", "Pillow", "Pillow"),
        ("tqdm", "tqdm"),
        ("loguru", "Loguru"),
    ]

    for pkg_info in other_packages:
        import_name = pkg_info[0]
        display_name = pkg_info[1]
        package_name = pkg_info[2] if len(pkg_info) > 2 else pkg_info[0]

        is_installed, version = check_package(package_name, import_name)
        print_status(display_name, is_installed, version)

    # 總結
    print_header("檢查總結")

    all_core_installed = all(core_status)

    if all_core_installed:
        print("  ✓ 所有核心依賴項目已正確安裝")
    else:
        print("  ✗ 部分核心依賴項目缺失，請安裝:")
        print("      pip install -r requirements.txt")

    if not cuda_available:
        print("  ⚠ CUDA 不可用，將使用 CPU 進行推理（速度較慢）")

    if not orbbec_available:
        print("  ⚠ OrbbecSDK 未安裝，無法使用 Gemini 2 相機")
        print("      請參考: https://github.com/orbbec/pyorbbecsdk")

    print("\n" + "=" * 60 + "\n")

    # 返回狀態碼
    return 0 if all_core_installed else 1


if __name__ == "__main__":
    sys.exit(main())
