"""
算法执行引擎
- TWAP: 时间加权均价
- VWAP: 成交量加权均价
- 智能拆单 & 滑点控制
"""
import time
import math
import threading
import numpy as np
from enum import Enum
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

from execution.qmt_api import QMTOrderManager, OrderRequest, OrderSide


class AlgoType(Enum):
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"  # 冰山单


@dataclass
class AlgoOrder:
    """算法母单"""
    algo_id: str
    symbol: str
    side: OrderSide
    total_volume: int
    algo_type: AlgoType
    duration_seconds: int = 300    # 执行时长
    price_limit: float = 0.0      # 价格限制
    max_participation: float = 0.1 # 最大参与率
    urgency: float = 0.5          # 紧急度 0~1

    # 执行状态
    filled_volume: int = 0
    filled_amount: float = 0.0
    child_orders: list = None
    status: str = "pending"
    start_time: float = 0.0

    def __post_init__(self):
        if self.child_orders is None:
            self.child_orders = []

    @property
    def remaining(self) -> int:
        return self.total_volume - self.filled_volume

    @property
    def avg_price(self) -> float:
        return self.filled_amount / self.filled_volume if self.filled_volume > 0 else 0

    @property
    def completion_pct(self) -> float:
        return self.filled_volume / self.total_volume if self.total_volume > 0 else 0


class AlgoExecutionEngine:
    """算法执行引擎"""

    def __init__(self, order_manager: QMTOrderManager, config: dict):
        self.om = order_manager
        self.config = config
        self._active_algos: Dict[str, AlgoOrder] = {}
        self._lock = threading.Lock()
        self._running = False

        # 注册成交回调
        self.om.on_fill(self._on_child_fill)

    def start(self):
        self._running = True
        self._exec_thread = threading.Thread(
            target=self._execution_loop, daemon=True
        )
        self._exec_thread.start()
        logger.info("算法执行引擎已启动")

    def stop(self):
        self._running = False
        # 撤销所有活跃算法单的子单
        for algo in self._active_algos.values():
            algo.status = "cancelled"
        logger.info("算法执行引擎已停止")

    # ==================== 提交算法单 ====================

    def submit(self, algo_order: AlgoOrder) -> str:
        """提交算法母单"""
        algo_order.start_time = time.time()
        algo_order.status = "running"

        with self._lock:
            self._active_algos[algo_order.algo_id] = algo_order

        logger.info(
            f"📋 算法单提交: {algo_order.algo_id} "
            f"{algo_order.symbol} {algo_order.side.value} "
            f"{algo_order.total_volume}股 "
            f"算法={algo_order.algo_type.value} "
            f"时长={algo_order.duration_seconds}s"
        )
        return algo_order.algo_id

    def cancel_algo(self, algo_id: str):
        """取消算法单"""
        with self._lock:
            if algo_id in self._active_algos:
                algo = self._active_algos[algo_id]
                algo.status = "cancelled"
                # 撤销所有子单
                self.om.cancel_all(algo.symbol)
                logger.info(f"算法单已取消: {algo_id}")

    # ==================== 执行循环 ====================

    def _execution_loop(self):
        """主执行循环"""
        while self._running:
            try:
                with self._lock:
                    active = {
                        k: v for k, v in self._active_algos.items()
                        if v.status == "running"
                    }

                for algo_id, algo in active.items():
                    self._execute_slice(algo)

                time.sleep(0.5)  # 500ms 检查一次

            except Exception as e:
                logger.error(f"执行循环异常: {e}")
                time.sleep(1)

    def _execute_slice(self, algo: AlgoOrder):
        """执行一个切片"""
        if algo.remaining <= 0:
            algo.status = "completed"
            logger.info(f"✅ 算法单完成: {algo.algo_id} 均价={algo.avg_price:.3f}")
            return

        elapsed = time.time() - algo.start_time
        if elapsed > algo.duration_seconds:
            # 超时: 剩余部分一次性下单
            self._send_child_order(algo, algo.remaining, aggressive=True)
            return

        if algo.algo_type == AlgoType.TWAP:
            self._twap_slice(algo, elapsed)
        elif algo.algo_type == AlgoType.VWAP:
            self._vwap_slice(algo, elapsed)
        elif algo.algo_type == AlgoType.ICEBERG:
            self._iceberg_slice(algo)

    # ==================== TWAP ====================

    def _twap_slice(self, algo: AlgoOrder, elapsed: float):
        """TWAP 切片逻辑"""
        total_slices = max(algo.duration_seconds // 10, 1)  # 每10秒一个切片
        current_slice = int(elapsed // 10)

        # 计算理论已完成量
        target_filled = int(algo.total_volume * (current_slice + 1) / total_slices)
        target_filled = min(target_filled, algo.total_volume)

        # 需要补单的量
        need_volume = target_filled - algo.filled_volume
        if need_volume < 100:  # 最小下单100股
            return

        # 取整到100的倍数
        need_volume = (need_volume // 100) * 100
        if need_volume <= 0:
            return

        self._send_child_order(algo, need_volume)

    # ==================== VWAP ====================

    def _vwap_slice(self, algo: AlgoOrder, elapsed: float):
        """VWAP 切片逻辑 (基于历史成交量分布)"""
        progress = elapsed / algo.duration_seconds

        # 获取历史成交量分布曲线 (简化版: 使用U型分布)
        vol_weight = self._get_volume_profile(progress)

        target_filled = int(algo.total_volume * vol_weight)
        need_volume = target_filled - algo.filled_volume

        if need_volume < 100:
            return

        need_volume = (need_volume // 100) * 100
        self._send_child_order(algo, need_volume)

    @staticmethod
    def _get_volume_profile(progress: float) -> float:
        """
        A股日内成交量分布 (U型)
        开盘和收盘量大, 午间量小
        """
        # 简化的U型分布
        x = progress
        # 双峰: 开盘高 -> 下降 -> 午间低 -> 上升 -> 收盘高
        weight = 0.15 * np.exp(-10 * (x - 0.0)**2) + \
                 0.10 * np.exp(-10 * (x - 0.15)**2) + \
                 0.05 + \
                 0.10 * np.exp(-10 * (x - 0.85)**2) + \
                 0.15 * np.exp(-10 * (x - 1.0)**2)

        # 累积分布
        steps = np.linspace(0, progress, 100)
        weights = [0.15 * np.exp(-10 * (s - 0.0)**2) +
                   0.10 * np.exp(-10 * (s - 0.15)**2) +
                   0.05 +
                   0.10 * np.exp(-10 * (s - 0.85)**2) +
                   0.15 * np.exp(-10 * (s - 1.0)**2)
                   for s in steps]

        total_steps = np.linspace(0, 1.0, 100)
        total_weights = [0.15 * np.exp(-10 * (s - 0.0)**2) +
                         0.10 * np.exp(-10 * (s - 0.15)**2) +
                         0.05 +
                         0.10 * np.exp(-10 * (s - 0.85)**2) +
                         0.15 * np.exp(-10 * (s - 1.0)**2)
                         for s in total_steps]

        return np.sum(weights) / np.sum(total_weights)

    # ==================== 冰山单 ====================

    def _iceberg_slice(self, algo: AlgoOrder):
        """冰山单: 每次只露出一小部分"""
        # 检查是否有活跃子单
        active_child = [
            o for o in self.om.get_active_orders()
            if o.request.symbol == algo.symbol
        ]
        if active_child:
            return  # 有活跃子单, 等待成交

        # 显示量: 总量的 5%~15%
        show_pct = 0.05 + 0.10 * algo.urgency
        show_volume = max(100, int(algo.remaining * show_pct))
        show_volume = min(show_volume, algo.remaining)
        show_volume = (show_volume // 100) * 100

        if show_volume > 0:
            self._send_child_order(algo, show_volume)

    # ==================== 子单下发 ====================

    def _send_child_order(self, algo: AlgoOrder, volume: int,
                          aggressive: bool = False):
        """下发子单"""
        # 获取当前行情 (从缓存)
        # current_price = self._get_current_price(algo.symbol)
        current_price = algo.price_limit  # 简化

        if aggressive:
            # 激进: 对手价
            if algo.side == OrderSide.BUY:
                price = round(current_price * 1.002, 2)  # 卖一价附近
            else:
                price = round(current_price * 0.998, 2)
            order_type = 24  # 最优五档即时成交
        else:
            # 被动: 挂单价
            if algo.side == OrderSide.BUY:
                price = round(current_price * 0.999, 2)  # 略低于现价
            else:
                price = round(current_price * 1.001, 2)
            order_type = 23  # 限价单

        request = OrderRequest(
            symbol=algo.symbol,
            side=algo.side,
            price=price,
            volume=volume,
            order_type=order_type,
            strategy_name=f"algo_{algo.algo_id}",
        )

        order_id = self.om.place_order(request)
        if order_id:
            algo.child_orders.append(order_id)

    # ==================== 成交回调 ====================

    def _on_child_fill(self, trade):
        """子单成交回调"""
        with self._lock:
            for algo in self._active_algos.values():
                if trade.order_id in algo.child_orders:
                    algo.filled_volume += trade.traded_volume
                    algo.filled_amount += trade.traded_price * trade.traded_volume
                    logger.info(
                        f"算法单进度: {algo.algo_id} "
                        f"{algo.filled_volume}/{algo.total_volume} "
                        f"({algo.completion_pct:.1%})"
                    )
                    break
