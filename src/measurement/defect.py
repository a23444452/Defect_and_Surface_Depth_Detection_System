"""
缺陷深度分析模組
檢測表面缺陷、凹陷、凸起、粗糙度等
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


@dataclass
class DefectResult:
    """缺陷檢測結果"""

    defect_type: str  # 缺陷類型 (dent, bump, scratch, crack, rough_surface)
    severity: str  # 嚴重程度 (minor, moderate, critical)
    depth: float  # 深度/高度 (mm)
    area: float  # 面積 (mm²)
    location: np.ndarray  # 位置 (3,) [x, y, z]
    confidence: float  # 信心度 [0-1]
    points: Optional[np.ndarray] = None  # 缺陷點雲 (N, 3)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DefectAnalyzer:
    """
    缺陷分析器
    基於 3D 點雲檢測各種表面缺陷
    """

    def __init__(
        self,
        dent_threshold: float = 0.5,
        bump_threshold: float = 0.3,
        roughness_threshold: float = 0.5,
    ):
        """
        初始化缺陷分析器

        Args:
            dent_threshold: 凹陷深度閾值 (mm)
            bump_threshold: 凸起高度閾值 (mm)
            roughness_threshold: 粗糙度閾值 (mm RMS)
        """
        self.dent_threshold = dent_threshold
        self.bump_threshold = bump_threshold
        self.roughness_threshold = roughness_threshold

    # ==================== 表面平面擬合 ====================

    def fit_plane_ransac(
        self,
        points: np.ndarray,
        iterations: int = 1000,
        distance_threshold: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 RANSAC 擬合平面

        Args:
            points: 3D 點雲 (N, 3)
            iterations: RANSAC 迭代次數
            distance_threshold: 內點距離閾值 (mm)

        Returns:
            (plane_model, inlier_mask)
            plane_model: [a, b, c, d] 表示 ax + by + cz + d = 0
            inlier_mask: (N,) bool 內點遮罩
        """
        if len(points) < 3:
            return np.array([0, 0, 1, 0]), np.zeros(len(points), dtype=bool)

        best_model = None
        best_inliers = 0
        best_mask = None

        for _ in range(iterations):
            # 隨機選 3 個點
            idx = np.random.choice(len(points), 3, replace=False)
            sample = points[idx]

            # 擬合平面
            model = self._fit_plane_3points(sample)

            if model is None:
                continue

            # 計算點到平面距離
            distances = self._point_to_plane_distance(points, model)

            # 計算內點
            inlier_mask = np.abs(distances) < distance_threshold
            num_inliers = np.sum(inlier_mask)

            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_model = model
                best_mask = inlier_mask

        if best_model is None:
            return np.array([0, 0, 1, 0]), np.zeros(len(points), dtype=bool)

        return best_model, best_mask

    def _fit_plane_3points(self, points: np.ndarray) -> Optional[np.ndarray]:
        """通過 3 個點擬合平面"""
        if len(points) != 3:
            return None

        p1, p2, p3 = points

        # 計算法向量 (叉積)
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        # 正規化
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None

        normal = normal / norm

        # 計算 d
        d = -np.dot(normal, p1)

        # 平面方程: ax + by + cz + d = 0
        return np.array([normal[0], normal[1], normal[2], d])

    def _point_to_plane_distance(
        self, points: np.ndarray, plane_model: np.ndarray
    ) -> np.ndarray:
        """
        計算點到平面的有向距離

        Args:
            points: (N, 3)
            plane_model: [a, b, c, d]

        Returns:
            distances: (N,) 有向距離 (正=上方, 負=下方)
        """
        a, b, c, d = plane_model

        # 有向距離 = (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2)
        numerator = points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d
        denominator = np.sqrt(a**2 + b**2 + c**2)

        distances = numerator / denominator

        return distances

    # ==================== 凹陷檢測 ====================

    def detect_dents(
        self, points: np.ndarray, min_area: int = 100
    ) -> List[DefectResult]:
        """
        檢測凹陷缺陷

        Args:
            points: 3D 點雲 (N, 3)
            min_area: 最小凹陷面積 (點數)

        Returns:
            缺陷列表
        """
        if len(points) < 10:
            return []

        # 1. 擬合平面
        plane_model, inlier_mask = self.fit_plane_ransac(points)

        # 2. 計算點到平面距離
        distances = self._point_to_plane_distance(points, plane_model)

        # 3. 找出凹陷點 (負距離)
        dent_mask = distances < -self.dent_threshold

        if np.sum(dent_mask) < min_area:
            return []

        # 4. 聚類凹陷區域
        dent_clusters = self._cluster_defects(points[dent_mask])

        # 5. 建立缺陷結果
        defects = []
        for cluster_points in dent_clusters:
            if len(cluster_points) < min_area:
                continue

            # 計算深度
            cluster_distances = self._point_to_plane_distance(cluster_points, plane_model)
            max_depth = float(np.abs(cluster_distances.min()))

            # 嚴重程度
            if max_depth > 2.0:
                severity = "critical"
            elif max_depth > 1.0:
                severity = "moderate"
            else:
                severity = "minor"

            # 位置 (質心)
            location = np.mean(cluster_points, axis=0)

            defect = DefectResult(
                defect_type="dent",
                severity=severity,
                depth=max_depth,
                area=float(len(cluster_points)),  # 簡化: 點數作為面積
                location=location,
                confidence=0.9 if len(cluster_points) > 500 else 0.75,
                points=cluster_points,
                metadata={"plane_model": plane_model.tolist()},
            )

            defects.append(defect)

        return defects

    # ==================== 凸起檢測 ====================

    def detect_bumps(
        self, points: np.ndarray, min_area: int = 100
    ) -> List[DefectResult]:
        """
        檢測凸起缺陷

        Args:
            points: 3D 點雲 (N, 3)
            min_area: 最小凸起面積 (點數)

        Returns:
            缺陷列表
        """
        if len(points) < 10:
            return []

        # 1. 擬合平面
        plane_model, inlier_mask = self.fit_plane_ransac(points)

        # 2. 計算點到平面距離
        distances = self._point_to_plane_distance(points, plane_model)

        # 3. 找出凸起點 (正距離)
        bump_mask = distances > self.bump_threshold

        if np.sum(bump_mask) < min_area:
            return []

        # 4. 聚類凸起區域
        bump_clusters = self._cluster_defects(points[bump_mask])

        # 5. 建立缺陷結果
        defects = []
        for cluster_points in bump_clusters:
            if len(cluster_points) < min_area:
                continue

            # 計算高度
            cluster_distances = self._point_to_plane_distance(cluster_points, plane_model)
            max_height = float(cluster_distances.max())

            # 嚴重程度
            if max_height > 1.5:
                severity = "critical"
            elif max_height > 0.8:
                severity = "moderate"
            else:
                severity = "minor"

            # 位置
            location = np.mean(cluster_points, axis=0)

            defect = DefectResult(
                defect_type="bump",
                severity=severity,
                depth=max_height,
                area=float(len(cluster_points)),
                location=location,
                confidence=0.9 if len(cluster_points) > 500 else 0.75,
                points=cluster_points,
                metadata={"plane_model": plane_model.tolist()},
            )

            defects.append(defect)

        return defects

    # ==================== 表面粗糙度分析 ====================

    def analyze_surface_roughness(self, points: np.ndarray) -> Dict[str, float]:
        """
        分析表面粗糙度

        Args:
            points: 3D 點雲 (N, 3)

        Returns:
            粗糙度指標字典
        """
        if len(points) < 10:
            return {"rms": 0.0, "ra": 0.0, "rz": 0.0}

        # 擬合平面
        plane_model, _ = self.fit_plane_ransac(points)

        # 計算點到平面距離
        distances = self._point_to_plane_distance(points, plane_model)

        # 粗糙度指標
        # Ra (算術平均粗糙度)
        ra = float(np.mean(np.abs(distances)))

        # RMS (均方根粗糙度)
        rms = float(np.sqrt(np.mean(distances**2)))

        # Rz (最大高度粗糙度)
        rz = float(np.max(distances) - np.min(distances))

        return {"rms": rms, "ra": ra, "rz": rz}

    # ==================== 缺陷聚類 ====================

    def _cluster_defects(
        self, points: np.ndarray, eps: float = 5.0
    ) -> List[np.ndarray]:
        """
        使用簡單的距離聚類分組缺陷

        Args:
            points: 缺陷點 (N, 3)
            eps: 鄰域距離 (mm)

        Returns:
            聚類列表
        """
        if len(points) == 0:
            return []

        # 簡化版 DBSCAN
        clusters = []
        visited = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if visited[i]:
                continue

            # 找鄰居
            distances = np.linalg.norm(points - points[i], axis=1)
            neighbors = distances < eps

            if np.sum(neighbors) < 5:  # 最小點數
                continue

            # 建立聚類
            cluster = points[neighbors]
            clusters.append(cluster)

            visited[neighbors] = True

        return clusters

    # ==================== 綜合缺陷檢測 ====================

    def detect_all_defects(
        self, points: np.ndarray, min_area: int = 100
    ) -> List[DefectResult]:
        """
        檢測所有類型缺陷

        Args:
            points: 3D 點雲 (N, 3)
            min_area: 最小缺陷面積

        Returns:
            所有缺陷列表
        """
        defects = []

        # 凹陷
        dents = self.detect_dents(points, min_area)
        defects.extend(dents)

        # 凸起
        bumps = self.detect_bumps(points, min_area)
        defects.extend(bumps)

        # 粗糙度
        roughness = self.analyze_surface_roughness(points)
        if roughness["rms"] > self.roughness_threshold:
            defect = DefectResult(
                defect_type="rough_surface",
                severity="moderate" if roughness["rms"] > 1.0 else "minor",
                depth=roughness["rms"],
                area=float(len(points)),
                location=np.mean(points, axis=0),
                confidence=0.85,
                metadata=roughness,
            )
            defects.append(defect)

        return defects


if __name__ == "__main__":
    # 測試缺陷分析
    print("缺陷分析模組測試\n")

    analyzer = DefectAnalyzer(dent_threshold=0.5, bump_threshold=0.3)

    # 建立測試表面 (平面 + 凹陷 + 凸起)
    print("測試 1: 凹陷檢測")

    # 基礎平面
    points = []
    for x in np.linspace(0, 100, 50):
        for y in np.linspace(0, 100, 50):
            z = 0 + np.random.normal(0, 0.1)  # 輕微噪聲
            points.append([x, y, z])

    # 加入凹陷
    for x in np.linspace(30, 40, 10):
        for y in np.linspace(30, 40, 10):
            z = -1.5 + np.random.normal(0, 0.1)  # 1.5mm 深凹陷
            points.append([x, y, z])

    # 加入凸起
    for x in np.linspace(60, 70, 10):
        for y in np.linspace(60, 70, 10):
            z = 1.0 + np.random.normal(0, 0.1)  # 1.0mm 高凸起
            points.append([x, y, z])

    points = np.array(points)

    print(f"  點數: {len(points)}")

    # 檢測凹陷
    dents = analyzer.detect_dents(points, min_area=50)
    print(f"\n檢測到 {len(dents)} 個凹陷:")
    for i, dent in enumerate(dents):
        print(f"  凹陷 {i+1}:")
        print(f"    深度: {dent.depth:.2f} mm")
        print(f"    面積: {dent.area:.0f} 點")
        print(f"    嚴重度: {dent.severity}")
        print(f"    位置: ({dent.location[0]:.1f}, {dent.location[1]:.1f}, {dent.location[2]:.1f})")

    # 檢測凸起
    bumps = analyzer.detect_bumps(points, min_area=50)
    print(f"\n檢測到 {len(bumps)} 個凸起:")
    for i, bump in enumerate(bumps):
        print(f"  凸起 {i+1}:")
        print(f"    高度: {bump.depth:.2f} mm")
        print(f"    面積: {bump.area:.0f} 點")
        print(f"    嚴重度: {bump.severity}")

    # 表面粗糙度
    print(f"\n測試 2: 表面粗糙度分析")
    roughness = analyzer.analyze_surface_roughness(points)
    print(f"  Ra (算術平均): {roughness['ra']:.3f} mm")
    print(f"  RMS (均方根): {roughness['rms']:.3f} mm")
    print(f"  Rz (最大高度): {roughness['rz']:.3f} mm")

    # 綜合檢測
    print(f"\n測試 3: 綜合缺陷檢測")
    all_defects = analyzer.detect_all_defects(points, min_area=50)
    print(f"  總共檢測到 {len(all_defects)} 個缺陷:")
    for defect in all_defects:
        print(f"    - {defect.defect_type}: {defect.severity} (深度/粗糙度: {defect.depth:.2f}mm)")

    print(f"\n✓ 缺陷分析模組測試完成")
