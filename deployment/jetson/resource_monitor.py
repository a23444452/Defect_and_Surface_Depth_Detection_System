#!/usr/bin/env python3
"""
資源監控工具
監控 Jetson 平台的 CPU、GPU、記憶體使用情況
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 設置日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """資源快照"""

    timestamp: datetime = field(default_factory=datetime.now)

    # CPU
    cpu_percent: float = 0.0  # CPU 使用率 (%)
    cpu_temp: float = 0.0  # CPU 溫度 (°C)

    # GPU
    gpu_percent: float = 0.0  # GPU 使用率 (%)
    gpu_memory_used: float = 0.0  # GPU 記憶體使用 (MB)
    gpu_memory_total: float = 0.0  # GPU 記憶體總量 (MB)
    gpu_temp: float = 0.0  # GPU 溫度 (°C)

    # 系統記憶體
    ram_used: float = 0.0  # RAM 使用 (MB)
    ram_total: float = 0.0  # RAM 總量 (MB)
    ram_percent: float = 0.0  # RAM 使用率 (%)

    # 交換記憶體
    swap_used: float = 0.0  # Swap 使用 (MB)
    swap_total: float = 0.0  # Swap 總量 (MB)

    # 功率
    power_draw: float = 0.0  # 功率消耗 (W)


class ResourceMonitor:
    """
    資源監控器
    支援 Jetson 和一般 Linux 平台
    """

    def __init__(self, use_jetson_stats: bool = True):
        """
        初始化監控器

        Args:
            use_jetson_stats: 是否使用 jetson-stats (僅 Jetson 平台)
        """
        self.use_jetson_stats = use_jetson_stats
        self.is_jetson = self._check_jetson()

        # 嘗試導入 jetson-stats
        self.jtop = None
        if self.is_jetson and use_jetson_stats:
            try:
                from jtop import jtop

                self.jtop = jtop
                logger.info("✓ jetson-stats 可用")
            except ImportError:
                logger.warning("jetson-stats 未安裝,使用備用方法")
                logger.info("安裝: sudo pip3 install jetson-stats")

        # 嘗試導入 psutil
        try:
            import psutil

            self.psutil = psutil
            logger.info("✓ psutil 可用")
        except ImportError:
            self.psutil = None
            logger.warning("psutil 未安裝")

        # 歷史記錄
        self.history = []
        self.max_history = 1000

    def _check_jetson(self) -> bool:
        """檢查是否為 Jetson 平台"""
        return Path("/etc/nv_tegra_release").exists()

    # ==================== 監控方法 ====================

    def get_snapshot(self) -> ResourceSnapshot:
        """
        取得當前資源快照

        Returns:
            ResourceSnapshot
        """
        snapshot = ResourceSnapshot()

        if self.jtop and self.is_jetson:
            # 使用 jetson-stats
            snapshot = self._get_snapshot_jtop()
        elif self.psutil:
            # 使用 psutil
            snapshot = self._get_snapshot_psutil()
        else:
            # 使用系統檔案
            snapshot = self._get_snapshot_sys()

        # 記錄歷史
        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return snapshot

    def _get_snapshot_jtop(self) -> ResourceSnapshot:
        """使用 jetson-stats 取得快照"""
        snapshot = ResourceSnapshot()

        try:
            with self.jtop() as jetson:
                # 等待資料
                if jetson.ok():
                    # CPU
                    snapshot.cpu_percent = jetson.stats.get("CPU", 0.0)
                    cpu_temps = jetson.temperature.get("CPU", {})
                    snapshot.cpu_temp = (
                        cpu_temps.get("temp", 0.0) if isinstance(cpu_temps, dict) else 0.0
                    )

                    # GPU
                    snapshot.gpu_percent = jetson.stats.get("GPU", 0.0)
                    gpu_mem = jetson.memory.get("GPU", {})
                    snapshot.gpu_memory_used = gpu_mem.get("used", 0) / 1024  # KB -> MB
                    snapshot.gpu_memory_total = gpu_mem.get("tot", 0) / 1024
                    gpu_temp = jetson.temperature.get("GPU", {})
                    snapshot.gpu_temp = (
                        gpu_temp.get("temp", 0.0) if isinstance(gpu_temp, dict) else 0.0
                    )

                    # RAM
                    ram = jetson.memory.get("RAM", {})
                    snapshot.ram_used = ram.get("used", 0) / 1024
                    snapshot.ram_total = ram.get("tot", 0) / 1024
                    snapshot.ram_percent = (
                        snapshot.ram_used / snapshot.ram_total * 100
                        if snapshot.ram_total > 0
                        else 0
                    )

                    # Swap
                    swap = jetson.memory.get("SWAP", {})
                    snapshot.swap_used = swap.get("used", 0) / 1024
                    snapshot.swap_total = swap.get("tot", 0) / 1024

                    # 功率
                    power = jetson.power.get("total", {})
                    snapshot.power_draw = (
                        power.get("power", 0.0) / 1000
                        if isinstance(power, dict)
                        else 0.0
                    )

        except Exception as e:
            logger.error(f"jtop 讀取失敗: {e}")

        return snapshot

    def _get_snapshot_psutil(self) -> ResourceSnapshot:
        """使用 psutil 取得快照"""
        snapshot = ResourceSnapshot()

        try:
            # CPU
            snapshot.cpu_percent = self.psutil.cpu_percent(interval=0.1)

            # 溫度 (需要 sensors)
            try:
                temps = self.psutil.sensors_temperatures()
                if temps:
                    # 嘗試找 CPU 溫度
                    for name, entries in temps.items():
                        if "cpu" in name.lower() or "coretemp" in name.lower():
                            snapshot.cpu_temp = entries[0].current if entries else 0.0
                            break
            except:
                pass

            # RAM
            mem = self.psutil.virtual_memory()
            snapshot.ram_used = mem.used / (1024**2)  # Bytes -> MB
            snapshot.ram_total = mem.total / (1024**2)
            snapshot.ram_percent = mem.percent

            # Swap
            swap = self.psutil.swap_memory()
            snapshot.swap_used = swap.used / (1024**2)
            snapshot.swap_total = swap.total / (1024**2)

        except Exception as e:
            logger.error(f"psutil 讀取失敗: {e}")

        return snapshot

    def _get_snapshot_sys(self) -> ResourceSnapshot:
        """使用系統檔案取得快照"""
        snapshot = ResourceSnapshot()

        try:
            # RAM (從 /proc/meminfo)
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
                mem_info = {}
                for line in lines:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = int(parts[1].strip().split()[0])  # kB
                        mem_info[key] = value

                mem_total = mem_info.get("MemTotal", 0) / 1024  # kB -> MB
                mem_free = mem_info.get("MemFree", 0) / 1024
                mem_available = mem_info.get("MemAvailable", 0) / 1024

                snapshot.ram_total = mem_total
                snapshot.ram_used = mem_total - mem_available
                snapshot.ram_percent = (
                    snapshot.ram_used / mem_total * 100 if mem_total > 0 else 0
                )

                # Swap
                swap_total = mem_info.get("SwapTotal", 0) / 1024
                swap_free = mem_info.get("SwapFree", 0) / 1024
                snapshot.swap_total = swap_total
                snapshot.swap_used = swap_total - swap_free

        except Exception as e:
            logger.error(f"系統檔案讀取失敗: {e}")

        return snapshot

    # ==================== 統計分析 ====================

    def get_statistics(self) -> Dict[str, Any]:
        """
        取得歷史統計

        Returns:
            統計字典
        """
        if len(self.history) == 0:
            return {}

        import numpy as np

        # 收集資料
        cpu_percents = [s.cpu_percent for s in self.history]
        gpu_percents = [s.gpu_percent for s in self.history]
        ram_percents = [s.ram_percent for s in self.history]
        cpu_temps = [s.cpu_temp for s in self.history if s.cpu_temp > 0]
        gpu_temps = [s.gpu_temp for s in self.history if s.gpu_temp > 0]
        power_draws = [s.power_draw for s in self.history if s.power_draw > 0]

        stats = {
            "samples": len(self.history),
            "cpu": {
                "avg": np.mean(cpu_percents),
                "max": np.max(cpu_percents),
                "min": np.min(cpu_percents),
            },
            "gpu": {
                "avg": np.mean(gpu_percents),
                "max": np.max(gpu_percents),
                "min": np.min(gpu_percents),
            },
            "ram": {
                "avg": np.mean(ram_percents),
                "max": np.max(ram_percents),
                "min": np.min(ram_percents),
            },
        }

        if cpu_temps:
            stats["cpu_temp"] = {
                "avg": np.mean(cpu_temps),
                "max": np.max(cpu_temps),
                "min": np.min(cpu_temps),
            }

        if gpu_temps:
            stats["gpu_temp"] = {
                "avg": np.mean(gpu_temps),
                "max": np.max(gpu_temps),
                "min": np.min(gpu_temps),
            }

        if power_draws:
            stats["power"] = {
                "avg": np.mean(power_draws),
                "max": np.max(power_draws),
                "min": np.min(power_draws),
            }

        return stats

    # ==================== 顯示方法 ====================

    def print_snapshot(self, snapshot: ResourceSnapshot):
        """
        列印快照

        Args:
            snapshot: 資源快照
        """
        print(f"\n資源使用狀況 ({snapshot.timestamp.strftime('%H:%M:%S')})")
        print("-" * 60)

        # CPU
        print(f"CPU:  {snapshot.cpu_percent:5.1f}%", end="")
        if snapshot.cpu_temp > 0:
            print(f"  溫度: {snapshot.cpu_temp:.1f}°C")
        else:
            print()

        # GPU
        if snapshot.gpu_percent > 0 or snapshot.gpu_memory_total > 0:
            print(f"GPU:  {snapshot.gpu_percent:5.1f}%", end="")
            if snapshot.gpu_memory_total > 0:
                print(
                    f"  記憶體: {snapshot.gpu_memory_used:.0f}/{snapshot.gpu_memory_total:.0f} MB",
                    end="",
                )
            if snapshot.gpu_temp > 0:
                print(f"  溫度: {snapshot.gpu_temp:.1f}°C")
            else:
                print()

        # RAM
        print(
            f"RAM:  {snapshot.ram_percent:5.1f}%  "
            f"({snapshot.ram_used:.0f}/{snapshot.ram_total:.0f} MB)"
        )

        # Swap
        if snapshot.swap_total > 0:
            swap_percent = (
                snapshot.swap_used / snapshot.swap_total * 100
                if snapshot.swap_total > 0
                else 0
            )
            print(
                f"Swap: {swap_percent:5.1f}%  "
                f"({snapshot.swap_used:.0f}/{snapshot.swap_total:.0f} MB)"
            )

        # 功率
        if snapshot.power_draw > 0:
            print(f"功率: {snapshot.power_draw:.2f} W")

    def print_statistics(self):
        """列印統計資訊"""
        stats = self.get_statistics()

        if not stats:
            print("無歷史資料")
            return

        print(f"\n資源使用統計 ({stats['samples']} 個樣本)")
        print("=" * 60)

        print(f"\nCPU:     平均 {stats['cpu']['avg']:5.1f}%  " f"最大 {stats['cpu']['max']:5.1f}%")

        print(f"GPU:     平均 {stats['gpu']['avg']:5.1f}%  " f"最大 {stats['gpu']['max']:5.1f}%")

        print(f"RAM:     平均 {stats['ram']['avg']:5.1f}%  " f"最大 {stats['ram']['max']:5.1f}%")

        if "cpu_temp" in stats:
            print(
                f"CPU 溫度: 平均 {stats['cpu_temp']['avg']:5.1f}°C  "
                f"最大 {stats['cpu_temp']['max']:5.1f}°C"
            )

        if "gpu_temp" in stats:
            print(
                f"GPU 溫度: 平均 {stats['gpu_temp']['avg']:5.1f}°C  "
                f"最大 {stats['gpu_temp']['max']:5.1f}°C"
            )

        if "power" in stats:
            print(
                f"功率:    平均 {stats['power']['avg']:5.2f} W  "
                f"最大 {stats['power']['max']:5.2f} W"
            )

    # ==================== 連續監控 ====================

    def monitor_continuous(self, duration: float = 60.0, interval: float = 1.0):
        """
        連續監控

        Args:
            duration: 監控時長 (秒)
            interval: 採樣間隔 (秒)
        """
        logger.info(f"開始連續監控 (時長: {duration}s, 間隔: {interval}s)")

        start_time = time.time()
        sample_count = 0

        try:
            while time.time() - start_time < duration:
                snapshot = self.get_snapshot()
                self.print_snapshot(snapshot)

                sample_count += 1
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("\n監控已中斷")

        # 顯示統計
        self.print_statistics()

        logger.info(f"\n完成 {sample_count} 個樣本")


def main():
    """示範使用"""
    logger.info("資源監控器示範")

    # 建立監控器
    monitor = ResourceMonitor()

    # 取得單次快照
    logger.info("\n單次快照:")
    snapshot = monitor.get_snapshot()
    monitor.print_snapshot(snapshot)

    # 連續監控
    logger.info("\n連續監控 10 秒...")
    monitor.monitor_continuous(duration=10.0, interval=1.0)


if __name__ == "__main__":
    main()
