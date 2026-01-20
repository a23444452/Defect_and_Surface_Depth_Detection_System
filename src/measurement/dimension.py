"""
3D 尺寸量測模組
提供基於點雲的尺寸量測功能
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


@dataclass
class MeasurementResult:
    """量測結果"""

    length: float  # 長度 (mm)
    width: float  # 寬度 (mm)
    height: float  # 高度 (mm)
    volume: float  # 體積 (mm³)
    center: np.ndarray  # 中心點 (3,)
    rotation: np.ndarray  # 旋轉矩陣 (3, 3)
    confidence: float  # 信心度 [0-1]
    method: str  # 量測方法
    metadata: Dict[str, Any] = field(default_factory=dict)


class DimensionMeasurement:
    """
    3D 尺寸量測器
    提供多種 3D 尺寸量測方法
    """

    def __init__(self):
        """初始化量測器"""
        pass

    # ==================== OBB 尺寸量測 ====================

    def measure_obb(
        self, points: np.ndarray, use_open3d: bool = True
    ) -> MeasurementResult:
        """
        使用 Oriented Bounding Box (OBB) 量測尺寸

        Args:
            points: 3D 點雲 (N, 3) [x, y, z] mm
            use_open3d: 是否使用 Open3D (更精確)

        Returns:
            MeasurementResult
        """
        if len(points) < 3:
            return self._empty_result()

        if use_open3d:
            try:
                return self._measure_obb_open3d(points)
            except ImportError:
                # Fallback to PCA method
                pass

        return self._measure_obb_pca(points)

    def _measure_obb_open3d(self, points: np.ndarray) -> MeasurementResult:
        """使用 Open3D 計算 OBB"""
        import open3d as o3d

        # 建立點雲
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 計算 OBB
        obb = pcd.get_oriented_bounding_box()

        # 取得尺寸
        extent = obb.extent  # [length, width, height]
        center = obb.center
        rotation = obb.R

        # 排序尺寸 (長>寬>高)
        sorted_extent = np.sort(extent)[::-1]

        # 計算體積
        volume = float(np.prod(extent))

        return MeasurementResult(
            length=float(sorted_extent[0]),
            width=float(sorted_extent[1]),
            height=float(sorted_extent[2]),
            volume=volume,
            center=np.array(center),
            rotation=np.array(rotation),
            confidence=0.95 if len(points) > 1000 else 0.8,
            method="OBB_Open3D",
        )

    def _measure_obb_pca(self, points: np.ndarray) -> MeasurementResult:
        """
        使用 PCA 計算 OBB

        原理:
        1. 中心化點雲
        2. 計算協方差矩陣
        3. 特徵值分解找到主軸
        4. 投影到主軸計算範圍
        """
        # 中心化
        center = np.mean(points, axis=0)
        centered_points = points - center

        # 協方差矩陣
        cov = np.cov(centered_points.T)

        # 特徵值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 排序 (大到小)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 投影到主軸
        projected = centered_points @ eigenvectors

        # 計算範圍
        min_vals = np.min(projected, axis=0)
        max_vals = np.max(projected, axis=0)
        extents = max_vals - min_vals

        # 排序尺寸
        sorted_extents = np.sort(extents)[::-1]

        # 計算體積
        volume = float(np.prod(extents))

        return MeasurementResult(
            length=float(sorted_extents[0]),
            width=float(sorted_extents[1]),
            height=float(sorted_extents[2]),
            volume=volume,
            center=center,
            rotation=eigenvectors,
            confidence=0.85 if len(points) > 1000 else 0.7,
            method="OBB_PCA",
        )

    # ==================== AABB 尺寸量測 ====================

    def measure_aabb(self, points: np.ndarray) -> MeasurementResult:
        """
        使用 Axis-Aligned Bounding Box (AABB) 量測尺寸

        Args:
            points: 3D 點雲 (N, 3)

        Returns:
            MeasurementResult
        """
        if len(points) < 3:
            return self._empty_result()

        # 計算 AABB
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        extents = max_vals - min_vals

        # 中心點
        center = (min_vals + max_vals) / 2

        # 排序尺寸
        sorted_extents = np.sort(extents)[::-1]

        # 體積
        volume = float(np.prod(extents))

        return MeasurementResult(
            length=float(sorted_extents[0]),
            width=float(sorted_extents[1]),
            height=float(sorted_extents[2]),
            volume=volume,
            center=center,
            rotation=np.eye(3),  # AABB 無旋轉
            confidence=0.9 if len(points) > 1000 else 0.75,
            method="AABB",
        )

    # ==================== 關鍵點距離量測 ====================

    def measure_keypoint_distance(
        self, point1: np.ndarray, point2: np.ndarray
    ) -> float:
        """
        計算兩個關鍵點之間的 3D 距離

        Args:
            point1: 第一個點 (3,) [x, y, z]
            point2: 第二個點 (3,) [x, y, z]

        Returns:
            距離 (mm)
        """
        return float(np.linalg.norm(point2 - point1))

    def measure_multiple_keypoints(
        self, keypoints: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        計算多個關鍵點之間的距離

        Args:
            keypoints: 關鍵點列表 [(3,), (3,), ...]

        Returns:
            距離字典 {"point_i_to_j": distance}
        """
        distances = {}

        for i in range(len(keypoints)):
            for j in range(i + 1, len(keypoints)):
                key = f"point_{i}_to_{j}"
                dist = self.measure_keypoint_distance(keypoints[i], keypoints[j])
                distances[key] = dist

        return distances

    # ==================== 直徑/圓度檢測 ====================

    def measure_diameter(
        self, points: np.ndarray, method: str = "ransac"
    ) -> Tuple[float, float, np.ndarray]:
        """
        測量圓形物體直徑

        Args:
            points: 2D 投影點雲 (N, 2) [x, y]
            method: 方法 ("ransac", "least_squares")

        Returns:
            (直徑, 圓度誤差, 圓心)
        """
        if len(points) < 3:
            return 0.0, 0.0, np.zeros(2)

        if method == "ransac":
            return self._fit_circle_ransac(points)
        else:
            return self._fit_circle_least_squares(points)

    def _fit_circle_ransac(
        self, points: np.ndarray, iterations: int = 100
    ) -> Tuple[float, float, np.ndarray]:
        """使用 RANSAC 擬合圓"""
        best_radius = 0.0
        best_center = np.zeros(2)
        best_inliers = 0

        for _ in range(iterations):
            # 隨機選 3 個點
            if len(points) < 3:
                break

            idx = np.random.choice(len(points), 3, replace=False)
            sample = points[idx]

            # 擬合圓
            center, radius = self._fit_circle_3points(sample)

            if radius <= 0:
                continue

            # 計算內點
            distances = np.linalg.norm(points - center, axis=1)
            inliers = np.sum(np.abs(distances - radius) < 5.0)  # 5mm 容差

            if inliers > best_inliers:
                best_inliers = inliers
                best_radius = radius
                best_center = center

        # 計算圓度誤差
        distances = np.linalg.norm(points - best_center, axis=1)
        circularity_error = np.std(distances - best_radius)

        diameter = best_radius * 2
        return diameter, circularity_error, best_center

    def _fit_circle_3points(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """通過 3 個點擬合圓"""
        if len(points) != 3:
            return np.zeros(2), 0.0

        p1, p2, p3 = points

        # 使用解析法
        ax, ay = p1
        bx, by = p2
        cx, cy = p3

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

        if abs(d) < 1e-6:
            return np.zeros(2), 0.0

        ux = (
            (ax**2 + ay**2) * (by - cy)
            + (bx**2 + by**2) * (cy - ay)
            + (cx**2 + cy**2) * (ay - by)
        ) / d

        uy = (
            (ax**2 + ay**2) * (cx - bx)
            + (bx**2 + by**2) * (ax - cx)
            + (cx**2 + cy**2) * (bx - ax)
        ) / d

        center = np.array([ux, uy])
        radius = np.linalg.norm(p1 - center)

        return center, radius

    def _fit_circle_least_squares(
        self, points: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """使用最小二乘法擬合圓"""
        # 簡化版本: 使用質心作為圓心估計
        center = np.mean(points, axis=0)

        # 計算平均半徑
        distances = np.linalg.norm(points - center, axis=1)
        radius = np.mean(distances)

        # 圓度誤差
        circularity_error = np.std(distances)

        diameter = radius * 2
        return diameter, circularity_error, center

    # ==================== 體積計算 ====================

    def measure_volume(
        self, points: np.ndarray, method: str = "convex_hull"
    ) -> float:
        """
        計算物體體積

        Args:
            points: 3D 點雲 (N, 3)
            method: 方法 ("convex_hull", "voxel")

        Returns:
            體積 (mm³)
        """
        if len(points) < 4:
            return 0.0

        if method == "convex_hull":
            return self._volume_convex_hull(points)
        else:
            return self._volume_voxel(points)

    def _volume_convex_hull(self, points: np.ndarray) -> float:
        """使用凸包計算體積"""
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(points)
            return float(hull.volume)

        except (ImportError, Exception):
            # Fallback to bounding box
            extents = np.ptp(points, axis=0)
            return float(np.prod(extents))

    def _volume_voxel(self, points: np.ndarray, voxel_size: float = 1.0) -> float:
        """使用體素計算體積"""
        # 體素化
        voxel_indices = np.floor(points / voxel_size).astype(int)

        # 去重計算體素數
        unique_voxels = np.unique(voxel_indices, axis=0)
        num_voxels = len(unique_voxels)

        # 體積
        volume = num_voxels * (voxel_size**3)

        return float(volume)

    # ==================== 工具函數 ====================

    def _empty_result(self) -> MeasurementResult:
        """返回空結果"""
        return MeasurementResult(
            length=0.0,
            width=0.0,
            height=0.0,
            volume=0.0,
            center=np.zeros(3),
            rotation=np.eye(3),
            confidence=0.0,
            method="Empty",
        )


if __name__ == "__main__":
    # 測試 3D 尺寸量測
    print("3D 尺寸量測模組測試\n")

    measurer = DimensionMeasurement()

    # 建立測試點雲 (長方體)
    print("測試 1: OBB 量測 (長方體)")
    points = []
    for x in np.linspace(0, 100, 20):
        for y in np.linspace(0, 50, 10):
            for z in np.linspace(0, 30, 6):
                points.append([x, y, z])
    points = np.array(points)

    print(f"  點數: {len(points)}")

    # OBB 量測
    result_obb_pca = measurer.measure_obb(points, use_open3d=False)
    print(f"\nOBB (PCA) 量測結果:")
    print(f"  長度: {result_obb_pca.length:.2f} mm (實際: 100mm)")
    print(f"  寬度: {result_obb_pca.width:.2f} mm (實際: 50mm)")
    print(f"  高度: {result_obb_pca.height:.2f} mm (實際: 30mm)")
    print(f"  體積: {result_obb_pca.volume:.0f} mm³ (實際: 150000mm³)")
    print(f"  信心度: {result_obb_pca.confidence:.2f}")

    # AABB 量測
    result_aabb = measurer.measure_aabb(points)
    print(f"\nAABB 量測結果:")
    print(f"  長度: {result_aabb.length:.2f} mm")
    print(f"  寬度: {result_aabb.width:.2f} mm")
    print(f"  高度: {result_aabb.height:.2f} mm")
    print(f"  體積: {result_aabb.volume:.0f} mm³")

    # 測試關鍵點距離
    print(f"\n測試 2: 關鍵點距離")
    keypoints = [np.array([0, 0, 0]), np.array([100, 0, 0]), np.array([100, 50, 0])]

    distances = measurer.measure_multiple_keypoints(keypoints)
    print(f"  關鍵點距離:")
    for key, dist in distances.items():
        print(f"    {key}: {dist:.2f} mm")

    # 測試直徑量測
    print(f"\n測試 3: 直徑量測 (圓形)")
    # 生成圓形點雲
    radius = 25.0
    circle_points = []
    for angle in np.linspace(0, 2 * np.pi, 100):
        x = radius * np.cos(angle) + np.random.normal(0, 0.5)
        y = radius * np.sin(angle) + np.random.normal(0, 0.5)
        circle_points.append([x, y])
    circle_points = np.array(circle_points)

    diameter, circularity, center = measurer.measure_diameter(circle_points, method="ransac")
    print(f"  測量直徑: {diameter:.2f} mm (實際: {radius*2:.2f}mm)")
    print(f"  圓度誤差: {circularity:.2f} mm")
    print(f"  圓心: ({center[0]:.2f}, {center[1]:.2f})")

    # 測試體積
    print(f"\n測試 4: 體積計算")
    volume_ch = measurer.measure_volume(points, method="convex_hull")
    volume_vx = measurer.measure_volume(points, method="voxel")
    print(f"  凸包體積: {volume_ch:.0f} mm³")
    print(f"  體素體積: {volume_vx:.0f} mm³")

    print(f"\n✓ 3D 尺寸量測模組測試完成")
