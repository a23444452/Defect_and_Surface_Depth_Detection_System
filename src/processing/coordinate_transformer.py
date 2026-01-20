"""
座標轉換器
處理 2D 像素座標與 3D 空間座標之間的轉換
"""

from typing import Tuple, Optional
import numpy as np


class CoordinateTransformer:
    """
    座標轉換器
    基於相機內參進行 2D ↔ 3D 座標轉換
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        """
        初始化座標轉換器

        Args:
            fx: 焦距 x (像素)
            fy: 焦距 y (像素)
            cx: 主點 x (像素)
            cy: 主點 y (像素)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    # ==================== 2D → 3D ====================

    def pixel_to_point(
        self,
        u: float,
        v: float,
        depth: float,
    ) -> Tuple[float, float, float]:
        """
        單一像素轉換為 3D 點

        Args:
            u: 像素座標 x
            v: 像素座標 y
            depth: 深度值 (mm)

        Returns:
            (x, y, z) 3D 座標 (mm)
        """
        if depth <= 0:
            return (0.0, 0.0, 0.0)

        # 針孔相機模型
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth

        return (x, y, z)

    def pixels_to_points(
        self,
        uvs: np.ndarray,
        depths: np.ndarray,
    ) -> np.ndarray:
        """
        批次像素轉換為 3D 點

        Args:
            uvs: 像素座標 (N, 2) [u, v]
            depths: 深度值 (N,) mm

        Returns:
            3D 點 (N, 3) [x, y, z] mm
        """
        if len(uvs) == 0:
            return np.array([]).reshape(0, 3)

        # 過濾無效深度
        valid_mask = depths > 0
        valid_uvs = uvs[valid_mask]
        valid_depths = depths[valid_mask]

        if len(valid_uvs) == 0:
            return np.array([]).reshape(0, 3)

        # 向量化計算
        u = valid_uvs[:, 0]
        v = valid_uvs[:, 1]

        x = (u - self.cx) * valid_depths / self.fx
        y = (v - self.cy) * valid_depths / self.fy
        z = valid_depths

        points = np.column_stack([x, y, z])

        return points

    def depth_image_to_points(
        self,
        depth: np.ndarray,
        subsample: int = 1,
    ) -> np.ndarray:
        """
        深度影像轉換為 3D 點雲

        Args:
            depth: 深度影像 (H, W) float32, 單位 mm
            subsample: 降採樣係數 (1 = 不降採樣)

        Returns:
            3D 點 (N, 3) [x, y, z] mm
        """
        h, w = depth.shape

        # 建立像素網格
        v_coords, u_coords = np.mgrid[0:h:subsample, 0:w:subsample]

        # 展平
        u = u_coords.flatten()
        v = v_coords.flatten()
        d = depth[::subsample, ::subsample].flatten()

        # 過濾無效深度
        valid_mask = d > 0
        u = u[valid_mask]
        v = v[valid_mask]
        d = d[valid_mask]

        # 轉換為 3D
        x = (u - self.cx) * d / self.fx
        y = (v - self.cy) * d / self.fy
        z = d

        points = np.column_stack([x, y, z])

        return points

    # ==================== 3D → 2D ====================

    def point_to_pixel(
        self,
        x: float,
        y: float,
        z: float,
    ) -> Tuple[int, int]:
        """
        單一 3D 點轉換為像素座標

        Args:
            x: 3D 座標 x (mm)
            y: 3D 座標 y (mm)
            z: 3D 座標 z (mm)

        Returns:
            (u, v) 像素座標
        """
        if z <= 0:
            return (-1, -1)

        # 針孔相機模型反向
        u = int(round(x * self.fx / z + self.cx))
        v = int(round(y * self.fy / z + self.cy))

        return (u, v)

    def points_to_pixels(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        批次 3D 點轉換為像素座標

        Args:
            points: 3D 點 (N, 3) [x, y, z] mm

        Returns:
            像素座標 (N, 2) [u, v]
        """
        if len(points) == 0:
            return np.array([]).reshape(0, 2)

        # 過濾無效深度
        valid_mask = points[:, 2] > 0
        valid_points = points[valid_mask]

        if len(valid_points) == 0:
            return np.array([]).reshape(0, 2)

        # 向量化計算
        x = valid_points[:, 0]
        y = valid_points[:, 1]
        z = valid_points[:, 2]

        u = (x * self.fx / z + self.cx).astype(int)
        v = (y * self.fy / z + self.cy).astype(int)

        pixels = np.column_stack([u, v])

        return pixels

    def points_to_depth_image(
        self,
        points: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        3D 點雲投影回深度影像

        Args:
            points: 3D 點 (N, 3) [x, y, z] mm
            image_shape: 影像尺寸 (H, W)

        Returns:
            深度影像 (H, W) float32
        """
        h, w = image_shape
        depth_image = np.zeros((h, w), dtype=np.float32)

        # 轉換為像素座標
        pixels = self.points_to_pixels(points)

        # 取得深度值
        valid_mask = points[:, 2] > 0
        depths = points[valid_mask, 2]

        # 檢查邊界
        valid_pixels = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < w)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < h)
        )

        pixels = pixels[valid_pixels]
        depths = depths[valid_pixels]

        # 填入深度值
        if len(pixels) > 0:
            depth_image[pixels[:, 1], pixels[:, 0]] = depths

        return depth_image

    # ==================== 座標系變換 ====================

    def camera_to_world(
        self,
        points: np.ndarray,
        transform_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        相機座標系轉換為世界座標系

        Args:
            points: 相機座標系下的點 (N, 3)
            transform_matrix: 變換矩陣 (4, 4) [R|t]

        Returns:
            世界座標系下的點 (N, 3)
        """
        if len(points) == 0:
            return np.array([]).reshape(0, 3)

        # 轉換為齊次座標
        ones = np.ones((len(points), 1))
        points_homogeneous = np.hstack([points, ones])

        # 應用變換
        transformed = (transform_matrix @ points_homogeneous.T).T

        # 轉回笛卡爾座標
        world_points = transformed[:, :3]

        return world_points

    def world_to_camera(
        self,
        points: np.ndarray,
        transform_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        世界座標系轉換為相機座標系

        Args:
            points: 世界座標系下的點 (N, 3)
            transform_matrix: 變換矩陣 (4, 4) [R|t]

        Returns:
            相機座標系下的點 (N, 3)
        """
        if len(points) == 0:
            return np.array([]).reshape(0, 3)

        # 使用逆矩陣
        inv_transform = np.linalg.inv(transform_matrix)

        # 轉換為齊次座標
        ones = np.ones((len(points), 1))
        points_homogeneous = np.hstack([points, ones])

        # 應用變換
        transformed = (inv_transform @ points_homogeneous.T).T

        # 轉回笛卡爾座標
        camera_points = transformed[:, :3]

        return camera_points

    # ==================== 工具函數 ====================

    def get_intrinsic_matrix(self) -> np.ndarray:
        """
        取得內參矩陣

        Returns:
            內參矩陣 K (3, 3)
        """
        K = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32
        )
        return K

    def compute_field_of_view(self, image_width: int, image_height: int) -> Tuple[float, float]:
        """
        計算視野角度

        Args:
            image_width: 影像寬度
            image_height: 影像高度

        Returns:
            (fov_x, fov_y) 視野角度 (度)
        """
        fov_x = 2 * np.arctan(image_width / (2 * self.fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(image_height / (2 * self.fy)) * 180 / np.pi

        return (fov_x, fov_y)


if __name__ == "__main__":
    # 測試座標轉換器
    print("座標轉換器測試\n")

    # 建立轉換器 (使用模擬相機的內參)
    transformer = CoordinateTransformer(fx=720.91, fy=720.91, cx=640, cy=400)

    print(f"內參:")
    print(f"  fx: {transformer.fx:.2f}")
    print(f"  fy: {transformer.fy:.2f}")
    print(f"  cx: {transformer.cx:.2f}")
    print(f"  cy: {transformer.cy:.2f}")

    # 測試單點轉換
    print(f"\n測試 2D → 3D:")
    u, v, depth = 640, 400, 1000  # 影像中心,深度 1000mm
    x, y, z = transformer.pixel_to_point(u, v, depth)
    print(f"  像素 ({u}, {v}), 深度 {depth}mm")
    print(f"  → 3D 點 ({x:.2f}, {y:.2f}, {z:.2f}) mm")

    # 測試反向轉換
    print(f"\n測試 3D → 2D:")
    u2, v2 = transformer.point_to_pixel(x, y, z)
    print(f"  3D 點 ({x:.2f}, {y:.2f}, {z:.2f}) mm")
    print(f"  → 像素 ({u2}, {v2})")
    print(f"  誤差: ({abs(u - u2)}, {abs(v - v2)}) 像素")

    # 測試批次轉換
    print(f"\n測試批次轉換:")
    test_uvs = np.array([[320, 200], [640, 400], [960, 600]])
    test_depths = np.array([800, 1000, 1200])

    points = transformer.pixels_to_points(test_uvs, test_depths)
    print(f"  輸入像素: {len(test_uvs)} 個")
    print(f"  輸出 3D 點: {len(points)} 個")

    # 測試深度影像轉點雲
    print(f"\n測試深度影像轉點雲:")
    test_depth = np.random.randint(800, 1200, (800, 1280)).astype(np.float32)
    pointcloud = transformer.depth_image_to_points(test_depth, subsample=4)
    print(f"  深度影像尺寸: {test_depth.shape}")
    print(f"  生成點雲: {len(pointcloud)} 個點")

    # 計算視野角度
    fov_x, fov_y = transformer.compute_field_of_view(1280, 800)
    print(f"\n視野角度:")
    print(f"  水平 FOV: {fov_x:.1f}°")
    print(f"  垂直 FOV: {fov_y:.1f}°")

    print(f"\n✓ 座標轉換器測試完成")
