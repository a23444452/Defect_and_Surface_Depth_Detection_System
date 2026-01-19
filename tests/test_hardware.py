"""
硬體介面模組測試
測試相機介面與驅動程式
"""

import sys
from pathlib import Path

# 加入專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import cv2
from src.hardware import MockCamera, GEMINI2_AVAILABLE
from src.utils import setup_logger, get_visualizer


def test_mock_camera_basic():
    """測試模擬相機基本功能"""
    print("\n" + "=" * 60)
    print("測試模擬相機 - 基本功能")
    print("=" * 60)

    camera = MockCamera(mode="objects")

    # 測試連接
    assert camera.connect() == True
    assert camera.is_connected == True
    print("  ✓ 相機連接成功")

    # 測試相機資訊
    info = camera.get_camera_info()
    assert info is not None
    print(f"  ✓ 相機型號: {info.model}")
    print(f"    序號: {info.serial_number}")
    print(f"    RGB 解析度: {info.rgb_resolution}")
    print(f"    深度解析度: {info.depth_resolution}")

    # 測試內參
    rgb_intrinsics = camera.get_rgb_intrinsics()
    assert rgb_intrinsics is not None
    print(f"  ✓ RGB 內參: fx={rgb_intrinsics.fx:.2f}, fy={rgb_intrinsics.fy:.2f}")

    depth_intrinsics = camera.get_depth_intrinsics()
    assert depth_intrinsics is not None
    print(f"  ✓ 深度內參: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}")

    # 測試中斷
    assert camera.disconnect() == True
    assert camera.is_connected == False
    print("  ✓ 相機中斷成功")


def test_mock_camera_streaming():
    """測試模擬相機串流功能"""
    print("\n" + "=" * 60)
    print("測試模擬相機 - 串流功能")
    print("=" * 60)

    camera = MockCamera(mode="objects")
    camera.connect()

    # 測試開始串流
    assert camera.start_streaming() == True
    assert camera.is_streaming == True
    print("  ✓ 開始串流")

    # 測試取得幀
    frame = camera.get_frame()
    assert frame is not None
    assert frame.rgb.shape == (1080, 1920, 3)
    assert frame.depth.shape == (800, 1280)
    print(f"  ✓ 取得幀 #{frame.frame_number}")
    print(f"    時間戳: {frame.timestamp:.3f}s")
    print(f"    RGB 大小: {frame.rgb.shape}")
    print(f"    深度大小: {frame.depth.shape}")
    print(f"    深度範圍: {frame.depth.min():.1f} - {frame.depth.max():.1f} mm")

    # 測試連續取得多幀
    print("\n  取得 5 幀測試:")
    for i in range(5):
        frame = camera.get_frame()
        print(f"    幀 #{frame.frame_number}: {frame.timestamp:.3f}s")

    # 測試停止串流
    assert camera.stop_streaming() == True
    assert camera.is_streaming == False
    print("\n  ✓ 停止串流")

    camera.disconnect()


def test_mock_camera_context_manager():
    """測試模擬相機 Context Manager"""
    print("\n" + "=" * 60)
    print("測試模擬相機 - Context Manager")
    print("=" * 60)

    # 使用 with 語句
    with MockCamera(mode="pattern") as camera:
        print("  ✓ 進入 context (自動連接與開始串流)")
        assert camera.is_connected == True
        assert camera.is_streaming == True

        # 取得幀
        frame = camera.get_frame()
        print(f"  ✓ 取得幀 #{frame.frame_number}")

    print("  ✓ 離開 context (自動停止與中斷)")


def test_mock_camera_modes():
    """測試模擬相機不同模式"""
    print("\n" + "=" * 60)
    print("測試模擬相機 - 不同模式")
    print("=" * 60)

    modes = ["random", "pattern", "objects"]
    visualizer = get_visualizer()

    for mode in modes:
        print(f"\n  測試模式: {mode}")

        with MockCamera(mode=mode) as camera:
            frame = camera.get_frame()

            # 儲存範例影像
            rgb_path = f"outputs/mock_camera_{mode}_rgb.png"
            depth_colored = visualizer.draw_depth_map(frame.depth)
            depth_path = f"outputs/mock_camera_{mode}_depth.png"

            visualizer.save_image(frame.rgb, rgb_path)
            visualizer.save_image(depth_colored, depth_path)

            print(f"    ✓ RGB 影像: {rgb_path}")
            print(f"    ✓ 深度影像: {depth_path}")


def test_mock_camera_with_logger():
    """測試模擬相機與日誌整合"""
    print("\n" + "=" * 60)
    print("測試模擬相機 - 日誌整合")
    print("=" * 60)

    logger = setup_logger(name="CameraTest", log_dir="outputs/logs", log_level="INFO")

    with MockCamera(mode="objects") as camera:
        logger.info("相機已連接並開始串流")

        # 取得並處理幀
        frame = camera.get_frame()
        logger.info(f"取得幀 #{frame.frame_number}")
        logger.info(f"  RGB 大小: {frame.rgb.shape}")
        logger.info(f"  深度範圍: {frame.depth.min():.1f} - {frame.depth.max():.1f} mm")

        # 檢查深度資料品質
        valid_depth = frame.depth[(frame.depth > 0) & (frame.depth < 10000)]
        if len(valid_depth) > 0:
            logger.success(f"有效深度像素: {len(valid_depth)}/{frame.depth.size} ({len(valid_depth)/frame.depth.size*100:.1f}%)")
        else:
            logger.warning("無有效深度資料")

    logger.info("相機已停止並中斷")
    print("  ✓ 日誌記錄完成 (outputs/logs/CameraTest.log)")


def test_mock_camera_performance():
    """測試模擬相機效能"""
    print("\n" + "=" * 60)
    print("測試模擬相機 - 效能測試")
    print("=" * 60)

    camera = MockCamera(mode="objects")
    camera.connect()
    camera.start_streaming()

    # 測試幀率
    num_frames = 30
    start_time = time.time()

    for i in range(num_frames):
        frame = camera.get_frame()

    elapsed = time.time() - start_time
    actual_fps = num_frames / elapsed

    print(f"  取得 {num_frames} 幀")
    print(f"  耗時: {elapsed:.3f}s")
    print(f"  實際 FPS: {actual_fps:.2f}")
    print(f"  目標 FPS: {camera.fps}")
    print(f"  ✓ 效能測試完成")

    camera.stop_streaming()
    camera.disconnect()


def main():
    """主測試函數"""
    print("\n" + "=" * 60)
    print("硬體介面模組測試開始")
    print("=" * 60)

    print(f"\nGemini2Camera 可用: {GEMINI2_AVAILABLE}")
    if not GEMINI2_AVAILABLE:
        print("  ⚠ OrbbecSDK 未安裝，將只測試模擬相機")

    # 執行測試
    test_mock_camera_basic()
    test_mock_camera_streaming()
    test_mock_camera_context_manager()
    test_mock_camera_modes()
    test_mock_camera_with_logger()
    test_mock_camera_performance()

    print("\n" + "=" * 60)
    print("所有測試完成！")
    print("=" * 60)

    print("\n生成的檔案：")
    print("  - outputs/mock_camera_random_rgb.png")
    print("  - outputs/mock_camera_random_depth.png")
    print("  - outputs/mock_camera_pattern_rgb.png")
    print("  - outputs/mock_camera_pattern_depth.png")
    print("  - outputs/mock_camera_objects_rgb.png")
    print("  - outputs/mock_camera_objects_depth.png")
    print("  - outputs/logs/CameraTest.log")


if __name__ == "__main__":
    main()
