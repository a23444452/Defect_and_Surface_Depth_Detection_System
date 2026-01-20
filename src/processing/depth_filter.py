"""
深度影像濾波器
提供多種深度影像處理與濾波方法
"""

from typing import Optional, List
import numpy as np
import cv2


class DepthFilter:
    """
    深度影像濾波器
    提供去噪、平滑、填洞等功能
    """

    def __init__(self):
        """初始化深度濾波器"""
        pass

    # ==================== 空間濾波 ====================

    def median_filter(
        self,
        depth: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """
        中值濾波

        Args:
            depth: 深度影像 (H, W) float32
            kernel_size: 核大小 (奇數)

        Returns:
            濾波後的深度影像
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # 確保為奇數

        # 只對有效深度進行濾波
        valid_mask = depth > 0
        filtered = depth.copy()

        # 使用 cv2.medianBlur
        # 需要轉換為 uint16 進行處理
        depth_uint16 = (depth * 1.0).astype(np.uint16)
        filtered_uint16 = cv2.medianBlur(depth_uint16, kernel_size)
        filtered = filtered_uint16.astype(np.float32)

        # 只保留原本有效的區域
        filtered[~valid_mask] = 0

        return filtered

    def bilateral_filter(
        self,
        depth: np.ndarray,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
    ) -> np.ndarray:
        """
        雙邊濾波 (保留邊緣)

        Args:
            depth: 深度影像 (H, W) float32
            d: 濾波器直徑
            sigma_color: 色彩空間的標準差
            sigma_space: 座標空間的標準差

        Returns:
            濾波後的深度影像
        """
        valid_mask = depth > 0
        filtered = depth.copy()

        # 雙邊濾波
        filtered = cv2.bilateralFilter(
            depth.astype(np.float32),
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
        )

        # 只保留原本有效的區域
        filtered[~valid_mask] = 0

        return filtered

    def gaussian_filter(
        self,
        depth: np.ndarray,
        kernel_size: int = 5,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """
        高斯濾波

        Args:
            depth: 深度影像 (H, W) float32
            kernel_size: 核大小 (奇數)
            sigma: 標準差

        Returns:
            濾波後的深度影像
        """
        if kernel_size % 2 == 0:
            kernel_size += 1

        valid_mask = depth > 0
        filtered = depth.copy()

        # 高斯濾波
        filtered = cv2.GaussianBlur(
            depth.astype(np.float32),
            (kernel_size, kernel_size),
            sigma,
        )

        # 只保留原本有效的區域
        filtered[~valid_mask] = 0

        return filtered

    def edge_preserving_filter(
        self,
        depth: np.ndarray,
        flags: int = cv2.RECURS_FILTER,
        sigma_s: float = 60.0,
        sigma_r: float = 0.4,
    ) -> np.ndarray:
        """
        邊緣保持濾波

        Args:
            depth: 深度影像 (H, W) float32
            flags: 濾波類型 (RECURS_FILTER 或 NORMCONV_FILTER)
            sigma_s: 空間標準差
            sigma_r: 範圍標準差

        Returns:
            濾波後的深度影像
        """
        valid_mask = depth > 0

        # 正規化到 0-255
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # 邊緣保持濾波
        filtered_normalized = cv2.edgePreservingFilter(
            depth_normalized,
            flags=flags,
            sigma_s=sigma_s,
            sigma_r=sigma_r,
        )

        # 反正規化
        filtered = filtered_normalized.astype(np.float32)
        filtered = filtered / 255.0 * depth.max()

        # 只保留原本有效的區域
        filtered[~valid_mask] = 0

        return filtered

    # ==================== 時間濾波 ====================

    def temporal_filter(
        self,
        depth_history: List[np.ndarray],
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        時間濾波 (指數移動平均)

        Args:
            depth_history: 歷史深度影像列表
            alpha: 平滑係數 (0-1)

        Returns:
            濾波後的深度影像
        """
        if len(depth_history) == 0:
            raise ValueError("深度歷史為空")

        if len(depth_history) == 1:
            return depth_history[0].copy()

        # 指數移動平均
        filtered = depth_history[0].copy()
        for depth in depth_history[1:]:
            valid_mask = (depth > 0) & (filtered > 0)
            filtered[valid_mask] = alpha * depth[valid_mask] + (1 - alpha) * filtered[valid_mask]

        return filtered

    # ==================== 填洞 ====================

    def fill_small_holes(
        self,
        depth: np.ndarray,
        max_hole_size: int = 10,
    ) -> np.ndarray:
        """
        填補小洞

        Args:
            depth: 深度影像 (H, W) float32
            max_hole_size: 最大填補的洞大小 (像素)

        Returns:
            填補後的深度影像
        """
        filled = depth.copy()

        # 找出無效區域
        invalid_mask = (depth == 0)

        # 形態學閉運算
        kernel_size = min(max_hole_size, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed_mask = cv2.morphologyEx(invalid_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # 找出需要填補的區域
        fill_mask = (closed_mask == 0) & invalid_mask

        # 使用周圍像素的平均值填補
        if np.any(fill_mask):
            filled = self.inpaint_depth(filled, fill_mask)

        return filled

    def inpaint_depth(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        inpaint_radius: int = 3,
    ) -> np.ndarray:
        """
        修補深度影像

        Args:
            depth: 深度影像 (H, W) float32
            mask: 需要修補的區域 (H, W) bool
            inpaint_radius: 修補半徑

        Returns:
            修補後的深度影像
        """
        # 轉換為 uint8 格式進行 inpainting
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        mask_uint8 = mask.astype(np.uint8) * 255

        # 使用 Telea 方法修補
        inpainted = cv2.inpaint(
            depth_normalized,
            mask_uint8,
            inpaintRadius=inpaint_radius,
            flags=cv2.INPAINT_TELEA,
        )

        # 反正規化
        filled = inpainted.astype(np.float32)
        filled = filled / 255.0 * depth.max()

        return filled

    def interpolate_depth(
        self,
        depth: np.ndarray,
        method: str = "linear",
    ) -> np.ndarray:
        """
        插值深度影像

        Args:
            depth: 深度影像 (H, W) float32
            method: 插值方法 ("linear", "cubic", "nearest")

        Returns:
            插值後的深度影像
        """
        from scipy import interpolate

        h, w = depth.shape

        # 找出有效點
        valid_mask = depth > 0
        valid_points = np.column_stack(np.where(valid_mask))
        valid_values = depth[valid_mask]

        if len(valid_points) < 4:
            # 太少有效點,無法插值
            return depth.copy()

        # 建立插值函數
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        try:
            if method == "nearest":
                interpolated = interpolate.griddata(
                    valid_points,
                    valid_values,
                    (y_coords, x_coords),
                    method="nearest",
                )
            elif method == "cubic":
                interpolated = interpolate.griddata(
                    valid_points,
                    valid_values,
                    (y_coords, x_coords),
                    method="cubic",
                )
            else:  # linear
                interpolated = interpolate.griddata(
                    valid_points,
                    valid_values,
                    (y_coords, x_coords),
                    method="linear",
                )

                # 對於插值失敗的區域,使用最近鄰插值
                invalid_after_linear = np.isnan(interpolated)
                if np.any(invalid_after_linear):
                    nearest = interpolate.griddata(
                        valid_points,
                        valid_values,
                        (y_coords, x_coords),
                        method="nearest",
                    )
                    interpolated[invalid_after_linear] = nearest[invalid_after_linear]

            # 處理可能的 NaN
            interpolated = np.nan_to_num(interpolated, nan=0.0)

            return interpolated

        except Exception as e:
            print(f"插值失敗: {e}")
            return depth.copy()

    # ==================== 異常值處理 ====================

    def remove_outliers(
        self,
        depth: np.ndarray,
        threshold: float = 3.0,
    ) -> np.ndarray:
        """
        移除異常值 (基於標準差)

        Args:
            depth: 深度影像 (H, W) float32
            threshold: 異常值閾值 (標準差倍數)

        Returns:
            移除異常值後的深度影像
        """
        filtered = depth.copy()

        # 只考慮有效深度
        valid_mask = depth > 0
        if not np.any(valid_mask):
            return filtered

        valid_depth = depth[valid_mask]

        # 計算統計量
        mean = np.mean(valid_depth)
        std = np.std(valid_depth)

        # 找出異常值
        outlier_mask = valid_mask & (
            (depth < mean - threshold * std) | (depth > mean + threshold * std)
        )

        # 移除異常值
        filtered[outlier_mask] = 0

        return filtered

    def clamp_range(
        self,
        depth: np.ndarray,
        min_depth: float = 300.0,
        max_depth: float = 3000.0,
    ) -> np.ndarray:
        """
        限制深度範圍

        Args:
            depth: 深度影像 (H, W) float32
            min_depth: 最小深度 (mm)
            max_depth: 最大深度 (mm)

        Returns:
            限制範圍後的深度影像
        """
        filtered = depth.copy()

        # 超出範圍的設為 0
        filtered[filtered < min_depth] = 0
        filtered[filtered > max_depth] = 0

        return filtered

    # ==================== 組合處理 ====================

    def process_depth(
        self,
        depth: np.ndarray,
        remove_outliers: bool = True,
        clamp_range: bool = True,
        filter_method: str = "bilateral",
        fill_holes: bool = True,
        min_depth: float = 300.0,
        max_depth: float = 3000.0,
    ) -> np.ndarray:
        """
        完整的深度處理流程

        Args:
            depth: 深度影像 (H, W) float32
            remove_outliers: 是否移除異常值
            clamp_range: 是否限制範圍
            filter_method: 濾波方法 ("median", "bilateral", "gaussian", "edge")
            fill_holes: 是否填洞
            min_depth: 最小深度
            max_depth: 最大深度

        Returns:
            處理後的深度影像
        """
        processed = depth.copy()

        # 1. 限制範圍
        if clamp_range:
            processed = self.clamp_range(processed, min_depth, max_depth)

        # 2. 移除異常值
        if remove_outliers:
            processed = self.remove_outliers(processed, threshold=3.0)

        # 3. 濾波
        if filter_method == "median":
            processed = self.median_filter(processed, kernel_size=5)
        elif filter_method == "bilateral":
            processed = self.bilateral_filter(processed, d=9)
        elif filter_method == "gaussian":
            processed = self.gaussian_filter(processed, kernel_size=5)
        elif filter_method == "edge":
            processed = self.edge_preserving_filter(processed)

        # 4. 填洞
        if fill_holes:
            processed = self.fill_small_holes(processed, max_hole_size=10)

        return processed


if __name__ == "__main__":
    # 測試深度濾波器
    print("深度濾波器測試\n")

    # 建立測試深度影像
    depth = np.random.randint(800, 1500, (480, 640)).astype(np.float32)

    # 加入一些噪點
    noise_mask = np.random.rand(480, 640) < 0.1
    depth[noise_mask] = np.random.randint(100, 2500, np.sum(noise_mask))

    # 加入一些洞
    hole_mask = np.random.rand(480, 640) < 0.05
    depth[hole_mask] = 0

    print(f"原始深度影像:")
    print(f"  尺寸: {depth.shape}")
    print(f"  有效像素: {np.sum(depth > 0)} / {depth.size}")
    print(f"  深度範圍: {depth[depth > 0].min():.1f} - {depth[depth > 0].max():.1f} mm")

    # 建立濾波器
    filter = DepthFilter()

    # 測試完整處理流程
    print(f"\n執行完整處理流程...")
    processed = filter.process_depth(
        depth,
        remove_outliers=True,
        clamp_range=True,
        filter_method="bilateral",
        fill_holes=True,
    )

    print(f"\n處理後深度影像:")
    print(f"  有效像素: {np.sum(processed > 0)} / {processed.size}")
    print(f"  深度範圍: {processed[processed > 0].min():.1f} - {processed[processed > 0].max():.1f} mm")

    print(f"\n✓ 深度濾波器測試完成")
