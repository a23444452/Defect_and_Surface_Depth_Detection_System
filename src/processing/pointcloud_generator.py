"""
點雲生成器
將深度影像轉換為 3D 點雲,支援 Open3D 整合
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np

try:
    from .coordinate_transformer import CoordinateTransformer
except ImportError:
    # 用於獨立測試
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.processing.coordinate_transformer import CoordinateTransformer


@dataclass
class PointCloud:
    """點雲資料類別"""

    points: np.ndarray  # 3D 點 (N, 3) [x, y, z] mm
    colors: Optional[np.ndarray] = None  # RGB 顏色 (N, 3) [0-255]
    normals: Optional[np.ndarray] = None  # 法向量 (N, 3)
    metadata: dict = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        """點數量"""
        return len(self.points)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """邊界 (min, max)"""
        if len(self.points) == 0:
            return (np.array([0, 0, 0]), np.array([0, 0, 0]))
        return (np.min(self.points, axis=0), np.max(self.points, axis=0))

    @property
    def center(self) -> np.ndarray:
        """中心點"""
        if len(self.points) == 0:
            return np.array([0, 0, 0])
        return np.mean(self.points, axis=0)

    def has_colors(self) -> bool:
        """是否有顏色"""
        return self.colors is not None and len(self.colors) > 0

    def has_normals(self) -> bool:
        """是否有法向量"""
        return self.normals is not None and len(self.normals) > 0


class PointCloudGenerator:
    """
    點雲生成器
    將 RGB-D 資料轉換為 3D 點雲
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ):
        """
        初始化點雲生成器

        Args:
            fx: 焦距 x (像素)
            fy: 焦距 y (像素)
            cx: 主點 x (像素)
            cy: 主點 y (像素)
        """
        self.transformer = CoordinateTransformer(fx, fy, cx, cy)

    # ==================== 點雲生成 ====================

    def generate(
        self,
        depth: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        subsample: int = 1,
        depth_scale: float = 1.0,
    ) -> PointCloud:
        """
        生成點雲

        Args:
            depth: 深度影像 (H, W) float32, 單位 mm
            rgb: RGB 影像 (H, W, 3) uint8, BGR 格式 (可選)
            subsample: 降採樣係數 (1 = 不降採樣)
            depth_scale: 深度縮放係數 (用於單位轉換)

        Returns:
            PointCloud
        """
        # 生成 3D 點
        points = self.transformer.depth_image_to_points(depth, subsample=subsample)

        # 縮放深度
        if depth_scale != 1.0:
            points = points * depth_scale

        # 提取對應的 RGB 顏色
        colors = None
        if rgb is not None:
            colors = self._extract_colors(depth, rgb, subsample)

        return PointCloud(
            points=points,
            colors=colors,
            metadata={
                "subsample": subsample,
                "depth_scale": depth_scale,
                "depth_shape": depth.shape,
            },
        )

    def generate_from_rgbd(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        subsample: int = 1,
    ) -> PointCloud:
        """
        從 RGB-D 生成彩色點雲

        Args:
            depth: 深度影像 (H, W) float32
            rgb: RGB 影像 (H, W, 3) uint8
            subsample: 降採樣係數

        Returns:
            彩色點雲
        """
        return self.generate(depth, rgb=rgb, subsample=subsample)

    def _extract_colors(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        subsample: int,
    ) -> np.ndarray:
        """
        提取對應的 RGB 顏色

        Args:
            depth: 深度影像 (H, W)
            rgb: RGB 影像 (H, W, 3)
            subsample: 降採樣係數

        Returns:
            顏色陣列 (N, 3) [0-255]
        """
        # 降採樣 RGB 影像
        rgb_sampled = rgb[::subsample, ::subsample]

        # 只保留有效深度的顏色
        valid_mask = depth[::subsample, ::subsample] > 0
        colors = rgb_sampled[valid_mask]

        # 轉換 BGR → RGB
        colors = colors[:, ::-1]

        return colors

    # ==================== 點雲處理 ====================

    def downsample(
        self,
        pointcloud: PointCloud,
        voxel_size: float,
    ) -> PointCloud:
        """
        體素降採樣

        Args:
            pointcloud: 輸入點雲
            voxel_size: 體素大小 (mm)

        Returns:
            降採樣後的點雲
        """
        if len(pointcloud.points) == 0:
            return pointcloud

        # 簡單的體素網格降採樣
        points = pointcloud.points
        colors = pointcloud.colors

        # 計算體素索引
        voxel_indices = np.floor(points / voxel_size).astype(int)

        # 使用字典去重
        voxel_dict = {}
        for i, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)

        # 每個體素取平均
        downsampled_points = []
        downsampled_colors = [] if colors is not None else None

        for indices in voxel_dict.values():
            # 點的平均
            downsampled_points.append(np.mean(points[indices], axis=0))

            # 顏色的平均
            if colors is not None:
                downsampled_colors.append(np.mean(colors[indices], axis=0).astype(np.uint8))

        downsampled_points = np.array(downsampled_points)
        if downsampled_colors is not None:
            downsampled_colors = np.array(downsampled_colors)

        return PointCloud(
            points=downsampled_points,
            colors=downsampled_colors,
            metadata=pointcloud.metadata,
        )

    def remove_outliers(
        self,
        pointcloud: PointCloud,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
    ) -> PointCloud:
        """
        移除統計異常值

        Args:
            pointcloud: 輸入點雲
            nb_neighbors: 鄰居數量
            std_ratio: 標準差倍數

        Returns:
            移除異常值後的點雲
        """
        if len(pointcloud.points) < nb_neighbors:
            return pointcloud

        from scipy.spatial import cKDTree

        points = pointcloud.points

        # 建立 KD 樹
        tree = cKDTree(points)

        # 計算每個點到鄰居的平均距離
        distances, _ = tree.query(points, k=nb_neighbors + 1)
        avg_distances = np.mean(distances[:, 1:], axis=1)  # 排除自己

        # 統計閾值
        mean_dist = np.mean(avg_distances)
        std_dist = np.std(avg_distances)
        threshold = mean_dist + std_ratio * std_dist

        # 保留內點
        inlier_mask = avg_distances < threshold
        filtered_points = points[inlier_mask]

        filtered_colors = None
        if pointcloud.colors is not None:
            filtered_colors = pointcloud.colors[inlier_mask]

        return PointCloud(
            points=filtered_points,
            colors=filtered_colors,
            metadata=pointcloud.metadata,
        )

    def estimate_normals(
        self,
        pointcloud: PointCloud,
        search_radius: float = 50.0,
        max_nn: int = 30,
    ) -> PointCloud:
        """
        估計法向量

        Args:
            pointcloud: 輸入點雲
            search_radius: 搜尋半徑 (mm)
            max_nn: 最大鄰居數

        Returns:
            帶法向量的點雲
        """
        if len(pointcloud.points) < 3:
            return pointcloud

        try:
            from scipy.spatial import cKDTree

            points = pointcloud.points
            normals = np.zeros_like(points)

            # 建立 KD 樹
            tree = cKDTree(points)

            # 對每個點估計法向量
            for i, point in enumerate(points):
                # 搜尋鄰居
                indices = tree.query_ball_point(point, search_radius)
                if len(indices) < 3:
                    continue

                indices = indices[:max_nn]
                neighbors = points[indices]

                # PCA 估計法向量
                centered = neighbors - neighbors.mean(axis=0)
                cov = centered.T @ centered
                eigenvalues, eigenvectors = np.linalg.eigh(cov)

                # 最小特徵值對應的特徵向量即為法向量
                normal = eigenvectors[:, 0]

                # 確保法向量朝向相機
                if normal[2] > 0:
                    normal = -normal

                normals[i] = normal

            return PointCloud(
                points=pointcloud.points,
                colors=pointcloud.colors,
                normals=normals,
                metadata=pointcloud.metadata,
            )

        except ImportError:
            print("需要 scipy 來估計法向量")
            return pointcloud

    # ==================== Open3D 整合 ====================

    def to_open3d(self, pointcloud: PointCloud):
        """
        轉換為 Open3D 點雲

        Args:
            pointcloud: 輸入點雲

        Returns:
            open3d.geometry.PointCloud
        """
        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()

            # 設定點
            pcd.points = o3d.utility.Vector3dVector(pointcloud.points)

            # 設定顏色
            if pointcloud.has_colors():
                # 正規化到 0-1
                colors_normalized = pointcloud.colors.astype(np.float64) / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

            # 設定法向量
            if pointcloud.has_normals():
                pcd.normals = o3d.utility.Vector3dVector(pointcloud.normals)

            return pcd

        except ImportError:
            raise ImportError("需要 open3d 套件: pip install open3d")

    def from_open3d(self, pcd) -> PointCloud:
        """
        從 Open3D 點雲轉換

        Args:
            pcd: open3d.geometry.PointCloud

        Returns:
            PointCloud
        """
        points = np.asarray(pcd.points)

        colors = None
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

        normals = None
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)

        return PointCloud(points=points, colors=colors, normals=normals)

    # ==================== 檔案 I/O ====================

    def save_ply(self, pointcloud: PointCloud, filepath: str):
        """
        儲存為 PLY 格式

        Args:
            pointcloud: 輸入點雲
            filepath: 檔案路徑
        """
        try:
            import open3d as o3d

            pcd = self.to_open3d(pointcloud)
            o3d.io.write_point_cloud(filepath, pcd)
            print(f"✓ 點雲已儲存: {filepath}")

        except ImportError:
            # 簡單的 PLY 寫入
            self._save_ply_simple(pointcloud, filepath)

    def _save_ply_simple(self, pointcloud: PointCloud, filepath: str):
        """簡單的 PLY 寫入 (不依賴 Open3D)"""
        with open(filepath, "w") as f:
            # PLY 標頭
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {pointcloud.num_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")

            if pointcloud.has_colors():
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

            f.write("end_header\n")

            # 寫入點資料
            for i in range(pointcloud.num_points):
                x, y, z = pointcloud.points[i]
                f.write(f"{x} {y} {z}")

                if pointcloud.has_colors():
                    r, g, b = pointcloud.colors[i]
                    f.write(f" {r} {g} {b}")

                f.write("\n")

        print(f"✓ 點雲已儲存: {filepath}")

    def load_ply(self, filepath: str) -> PointCloud:
        """
        載入 PLY 格式

        Args:
            filepath: 檔案路徑

        Returns:
            PointCloud
        """
        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(filepath)
            return self.from_open3d(pcd)

        except ImportError:
            raise ImportError("需要 open3d 套件來載入 PLY: pip install open3d")


if __name__ == "__main__":
    # 測試點雲生成器
    print("點雲生成器測試\n")

    # 修正導入 (用於測試)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.processing.coordinate_transformer import CoordinateTransformer

    # 建立測試資料
    depth = np.random.randint(800, 1500, (480, 640)).astype(np.float32)
    rgb = np.random.randint(0, 255, (480, 640, 3)).astype(np.uint8)

    # 加入一些洞
    hole_mask = np.random.rand(480, 640) < 0.1
    depth[hole_mask] = 0

    print(f"測試資料:")
    print(f"  深度影像: {depth.shape}")
    print(f"  RGB 影像: {rgb.shape}")
    print(f"  有效深度: {np.sum(depth > 0)} / {depth.size}")

    # 建立生成器 (使用模擬相機內參)
    generator = PointCloudGenerator(fx=720.91, fy=720.91, cx=640, cy=400)

    # 生成點雲
    print(f"\n生成點雲...")
    pointcloud = generator.generate_from_rgbd(depth, rgb, subsample=2)

    print(f"\n點雲資訊:")
    print(f"  點數量: {pointcloud.num_points:,}")
    print(f"  有顏色: {pointcloud.has_colors()}")
    print(f"  邊界: {pointcloud.bounds[0]} ~ {pointcloud.bounds[1]}")
    print(f"  中心: {pointcloud.center}")

    # 測試降採樣
    print(f"\n測試體素降採樣 (50mm)...")
    downsampled = generator.downsample(pointcloud, voxel_size=50.0)
    print(f"  原始點數: {pointcloud.num_points:,}")
    print(f"  降採樣後: {downsampled.num_points:,}")
    print(f"  壓縮率: {downsampled.num_points / pointcloud.num_points * 100:.1f}%")

    # 測試儲存
    print(f"\n測試儲存 PLY...")
    try:
        generator.save_ply(downsampled, "outputs/test_pointcloud.ply")
    except Exception as e:
        print(f"  儲存失敗: {e}")

    print(f"\n✓ 點雲生成器測試完成")
