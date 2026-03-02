"""
订单流因子
- 基于逐笔委托和成交数据
- 衡量买卖双方力量对比
"""
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class OrderFlowFactors:
    """订单流因子计算器"""

    def __init__(self, use_gpu: bool = False):
        try:
            if use_gpu:
                import cupy as cp
                self.xp = cp
            else:
                self.xp = np
        except ImportError:
            self.xp = np

    def trade_flow_imbalance(self, buy_volumes: np.ndarray,
                              sell_volumes: np.ndarray,
                              window: int = 100) -> float:
        """逐笔成交流不平衡"""
        xp = self.xp
        bv = xp.asarray(buy_volumes[-window:])
        sv = xp.asarray(sell_volumes[-window:])
        total = xp.sum(bv) + xp.sum(sv)
        if float(total) < 1e-8:
            return 0.0
        return float((xp.sum(bv) - xp.sum(sv)) / total)

    def order_arrival_rate(self, timestamps: np.ndarray,
                            directions: np.ndarray,
                            window: int = 200) -> Dict[str, float]:
        """
        委托到达率
        分别计算买/卖委托的到达速率
        """
        xp = self.xp
        ts = xp.asarray(timestamps[-window:])
        dirs = xp.asarray(directions[-window:])

        if len(ts) < 10:
            return {"buy_rate": 0, "sell_rate": 0, "rate_imbalance": 0}

        duration = float(ts[-1] - ts[0])
        if duration < 1e-6:
            return {"buy_rate": 0, "sell_rate": 0, "rate_imbalance": 0}

        buy_count = float(xp.sum(dirs == 1))
        sell_count = float(xp.sum(dirs == -1))

        buy_rate = buy_count / duration
        sell_rate = sell_count / duration
        total_rate = buy_rate + sell_rate

        return {
            "buy_rate": buy_rate,
            "sell_rate": sell_rate,
            "rate_imbalance": (
                (buy_rate - sell_rate) / total_rate
                if total_rate > 0 else 0
            ),
        }

    def cancel_rate(self, total_orders: int,
                     cancelled_orders: int) -> float:
        """撤单率"""
        if total_orders <= 0:
            return 0.0
        return cancelled_orders / total_orders

    def effective_spread(self, trade_prices: np.ndarray,
                          mid_prices: np.ndarray,
                          directions: np.ndarray) -> float:
        """
        有效价差
        衡量实际交易成本
        """
        xp = self.xp
        tp = xp.asarray(trade_prices)
        mp = xp.asarray(mid_prices)
        d = xp.asarray(directions)

        if len(tp) < 5:
            return 0.0

        # 有效价差 = 2 * direction * (trade_price - mid_price) / mid_price
        spreads = 2 * d * (tp - mp) / (mp + 1e-8)
        return float(xp.mean(spreads))

    def kyle_lambda(self, returns: np.ndarray,
                     signed_volumes: np.ndarray,
                     window: int = 100) -> float:
        """
        Kyle's Lambda
        价格冲击系数: return = lambda * signed_volume + epsilon
        """
        xp = self.xp
        r = xp.asarray(returns[-window:])
        sv = xp.asarray(signed_volumes[-window:])

        if len(r) < 20:
            return 0.0

        sv_mean = xp.mean(sv)
        r_mean = xp.mean(r)

        cov = xp.mean((sv - sv_mean) * (r - r_mean))
        var = xp.mean((sv - sv_mean) ** 2)

        if float(var) < 1e-12:
            return 0.0

        return float(cov / var)

    def amihud_illiquidity(self, returns: np.ndarray,
                            volumes: np.ndarray,
                            window: int = 20) -> float:
        """
        Amihud非流动性指标
        |return| / volume
        """
        xp = self.xp
        r = xp.asarray(returns[-window:])
        v = xp.asarray(volumes[-window:])

        mask = v > 0
        if xp.sum(mask) < 5:
            return 0.0

        illiq = xp.mean(xp.abs(r[mask]) / v[mask])
        return float(illiq)

    def toxicity_index(self, buy_volumes: np.ndarray,
                        sell_volumes: np.ndarray,
                        bucket_size: int = 50,
                        n_buckets: int = 30) -> float:
        """
        毒性指标 (类VPIN)
        衡量逆向选择风险
        """
        xp = self.xp
        bv = xp.asarray(buy_volumes)
        sv = xp.asarray(sell_volumes)

        total = bv + sv
        cum_vol = xp.cumsum(total)

        bucket_indices = (cum_vol // bucket_size).astype(int)
        max_bucket = int(bucket_indices[-1]) + 1

        if max_bucket < n_buckets:
            return 0.5

        bucket_imbalance = xp.zeros(max_bucket)
        bucket_total = xp.zeros(max_bucket)

        for i in range(max_bucket):
            mask = bucket_indices == i
            bucket_imbalance[i] = xp.abs(xp.sum(bv[mask]) - xp.sum(sv[mask]))
            bucket_total[i] = xp.sum(total[mask])

        recent = bucket_imbalance[-n_buckets:]
        recent_total = bucket_total[-n_buckets:]

        denom = xp.sum(recent_total)
        if float(denom) < 1e-8:
            return 0.5

        return float(xp.sum(recent) / denom)
