"""
高频微观结构因子
- 基于Level-2逐笔和盘口数据计算
- 支持GPU加速 (CuPy)
"""
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np


@dataclass
class FactorResult:
    name: str
    value: float
    timestamp: int
    symbol: str


class MicroStructureFactors:
    """微观结构因子计算器"""
    
    def __init__(self, use_gpu: bool = True):
        self.xp = cp if (use_gpu and GPU_AVAILABLE) else np
        self._cache = {}
        
    # ==================== 订单流因子 ====================
    
    def order_flow_imbalance(self, bid_volumes: np.ndarray, 
                              ask_volumes: np.ndarray,
                              bid_prices: np.ndarray,
                              ask_prices: np.ndarray,
                              window: int = 100) -> float:
        """
        订单流不平衡 (OFI)
        衡量买卖压力差异, 是最核心的L2因子之一
        """
        xp = self.xp
        bid_v = xp.asarray(bid_volumes[-window:])
        ask_v = xp.asarray(ask_volumes[-window:])
        bid_p = xp.asarray(bid_prices[-window:])
        ask_p = xp.asarray(ask_prices[-window:])
        
        # 计算各时刻的OFI增量
        delta_bid_v = xp.diff(bid_v)
        delta_ask_v = xp.diff(ask_v)
        delta_bid_p = xp.diff(bid_p)
        delta_ask_p = xp.diff(ask_p)
        
        # 买方OFI
        ofi_bid = xp.where(delta_bid_p > 0, bid_v[1:],
                  xp.where(delta_bid_p == 0, delta_bid_v, 0))
        
        # 卖方OFI
        ofi_ask = xp.where(delta_ask_p < 0, ask_v[1:],
                  xp.where(delta_ask_p == 0, -delta_ask_v, 0))
        
        ofi = float(xp.sum(ofi_bid - ofi_ask))
        
        # 标准化
        total = float(xp.sum(bid_v) + xp.sum(ask_v))
        return ofi / total if total > 0 else 0.0
    
    def volume_weighted_ofi(self, ticks: list, window: int = 200) -> float:
        """成交量加权订单流不平衡"""
        xp = self.xp
        prices = xp.array([t.price for t in ticks[-window:]])
        volumes = xp.array([t.volume for t in ticks[-window:]])
        directions = xp.array([t.direction for t in ticks[-window:]])
        
        buy_vol = xp.sum(volumes * (directions == 1))
        sell_vol = xp.sum(volumes * (directions == -1))
        total_vol = buy_vol + sell_vol
        
        if total_vol == 0:
            return 0.0
        return float((buy_vol - sell_vol) / total_vol)
    
    # ==================== 价格冲击因子 ====================
    
    def price_impact(self, prices: np.ndarray, volumes: np.ndarray,
                     directions: np.ndarray, window: int = 100) -> float:
        """
        价格冲击因子
        衡量单位成交量对价格的影响程度
        """
        xp = self.xp
        p = xp.asarray(prices[-window:])
        v = xp.asarray(volumes[-window:])
        d = xp.asarray(directions[-window:])
        
        returns = xp.diff(xp.log(p))
        signed_vol = (v[1:] * d[1:]).astype(float)
        
        # 回归: return = alpha + beta * signed_volume
        if len(returns) < 10:
            return 0.0
            
        sv_mean = xp.mean(signed_vol)
        r_mean = xp.mean(returns)
        
        cov = xp.mean((signed_vol - sv_mean) * (returns - r_mean))
        var = xp.mean((signed_vol - sv_mean) ** 2)
        
        beta = float(cov / var) if float(var) > 1e-12 else 0.0
        return beta
    
    # ==================== VPIN因子 ====================
    
    def vpin(self, prices: np.ndarray, volumes: np.ndarray,
             bucket_size: int = 50000, n_buckets: int = 50) -> float:
        """
        Volume-Synchronized Probability of Informed Trading (VPIN)
        知情交易概率, 用于检测异常交易行为
        """
        xp = self.xp
        p = xp.asarray(prices)
        v = xp.asarray(volumes)
        
        # 使用BVC方法分类买卖
        returns = xp.diff(xp.log(p))
        # 正收益 -> 买方主导, 负收益 -> 卖方主导
        buy_pct = xp.where(returns > 0, 1.0,
                  xp.where(returns < 0, 0.0, 0.5))
        
        vol = v[1:]
        buy_vol = vol * buy_pct
        sell_vol = vol * (1 - buy_pct)
        
        # 按成交量分桶
        cum_vol = xp.cumsum(vol)
        bucket_indices = (cum_vol // bucket_size).astype(int)
        
        max_bucket = int(bucket_indices[-1]) + 1
        if max_bucket < n_buckets:
            return 0.5  # 数据不足
            
        bucket_buy = xp.zeros(max_bucket)
        bucket_sell = xp.zeros(max_bucket)
        
        for i in range(max_bucket):
            mask = bucket_indices == i
            bucket_buy[i] = xp.sum(buy_vol[mask])
            bucket_sell[i] = xp.sum(sell_vol[mask])
        
        # 取最近n_buckets个桶
        recent_buy = bucket_buy[-n_buckets:]
        recent_sell = bucket_sell[-n_buckets:]
        
        vpin_val = float(xp.mean(xp.abs(recent_buy - recent_sell)) / bucket_size)
        return min(vpin_val, 1.0)
    
    # ==================== 盘口因子 ====================
    
    def bid_ask_spread(self, ask1: float, bid1: float, mid: float) -> float:
        """相对买卖价差"""
        if mid <= 0:
            return 0.0
        return (ask1 - bid1) / mid
    
    def depth_imbalance(self, bid_volumes: np.ndarray, 
                         ask_volumes: np.ndarray, levels: int = 5) -> float:
        """
        多档深度不平衡
        正值 -> 买方力量强, 负值 -> 卖方力量强
        """
        xp = self.xp
        bv = xp.asarray(bid_volumes[:levels])
        av = xp.asarray(ask_volumes[:levels])
        
        # 距离加权 (近档权重大)
        weights = 1.0 / xp.arange(1, levels + 1, dtype=float)
        
        weighted_bid = float(xp.sum(bv * weights))
        weighted_ask = float(xp.sum(av * weights))
        total = weighted_bid + weighted_ask
        
        return (weighted_bid - weighted_ask) / total if total > 0 else 0.0
    
    def order_book_slope(self, prices: np.ndarray, 
                          volumes: np.ndarray, side: str = 'bid') -> float:
        """
        盘口斜率
        衡量盘口深度的分布特征
        """
        xp = self.xp
        p = xp.asarray(prices[:10])
        v = xp.asarray(volumes[:10])
        
        if len(p) < 3:
            return 0.0
            
        mid = (p[0] + p[-1]) / 2
        price_dist = xp.abs(p - p[0]) / mid  # 相对价格距离
        cum_vol = xp.cumsum(v).astype(float)
        
        # 线性回归斜率
        n = len(price_dist)
        x_mean = xp.mean(price_dist)
        y_mean = xp.mean(cum_vol)
        
        slope = float(
            xp.sum((price_dist - x_mean) * (cum_vol - y_mean)) /
            (xp.sum((price_dist - x_mean) ** 2) + 1e-12)
        )
        return slope
    
    # ==================== 波动率因子 ====================
    
    def realized_volatility(self, prices: np.ndarray, 
                             window: int = 300) -> float:
        """已实现波动率 (基于逐笔数据)"""
        xp = self.xp
        p = xp.asarray(prices[-window:])
        log_returns = xp.diff(xp.log(p))
        return float(xp.sqrt(xp.sum(log_returns ** 2)))
    
    def bipower_variation(self, prices: np.ndarray, 
                           window: int = 300) -> float:
        """双幂次变差 (对跳跃更稳健)"""
        xp = self.xp
        p = xp.asarray(prices[-window:])
        log_returns = xp.abs(xp.diff(xp.log(p)))
        
        if len(log_returns) < 2:
            return 0.0
            
        bpv = float(
            (np.pi / 2) * xp.mean(log_returns[:-1] * log_returns[1:])
        )
        return bpv
    
    def jump_indicator(self, prices: np.ndarray, window: int = 300) -> float:
        """跳跃指标 = (RV - BPV) / RV"""
        rv = self.realized_volatility(prices, window)
        bpv = self.bipower_variation(prices, window)
        
        if rv < 1e-12:
            return 0.0
        return max(0, (rv - bpv) / rv)
    
    # ==================== 资金流因子 ====================
    
    def smart_money_flow(self, prices: np.ndarray, volumes: np.ndarray,
                          amounts: np.ndarray, directions: np.ndarray,
                          large_threshold: float = 500000) -> float:
        """
        聪明资金流向
        大单净流入 / 总成交额
        """
        xp = self.xp
        a = xp.asarray(amounts)
        d = xp.asarray(directions)
        
        large_mask = a >= large_threshold
        large_buy = xp.sum(a[large_mask & (d == 1)])
        large_sell = xp.sum(a[large_mask & (d == -1)])
        total = xp.sum(a)
        
        if float(total) < 1e-6:
            return 0.0
        return float((large_buy - large_sell) / total)
    
    # ==================== 批量计算 ====================
    
    def compute_all(self, symbol: str, orderbook: 'OrderBookSnapshot',
                    recent_ticks: list, timestamp: int) -> Dict[str, float]:
        """一次性计算所有因子"""
        
        results = {}
        
        try:
            # 盘口因子
            results['bid_ask_spread'] = self.bid_ask_spread(
                orderbook.ask_prices[0], orderbook.bid_prices[0],
                orderbook.last_price
            )
            results['depth_imbalance_5'] = self.depth_imbalance(
                orderbook.bid_volumes, orderbook.ask_volumes, levels=5
            )
            results['depth_imbalance_10'] = self.depth_imbalance(
                orderbook.bid_volumes, orderbook.ask_volumes, levels=10
            )
            results['ask_slope'] = self.order_book_slope(
                orderbook.ask_prices, orderbook.ask_volumes, 'ask'
            )
            results['bid_slope'] = self.order_book_slope(
                orderbook.bid_prices, orderbook.bid_volumes, 'bid'
            )
            
            if len(recent_ticks) > 50:
                prices = np.array([t.price for t in recent_ticks])
                volumes = np.array([t.volume for t in recent_ticks])
                directions = np.array([t.direction for t in recent_ticks])
                amounts = np.array([t.amount for t in recent_ticks])
                
                # 订单流因子
                results['volume_weighted_ofi'] = self.volume_weighted_ofi(
                    recent_ticks
                )
                results['price_impact'] = self.price_impact(
                    prices, volumes, directions
                )
                results['vpin'] = self.vpin(prices, volumes)
                
                # 波动率因子
                results['realized_vol'] = self.realized_volatility(prices)
                results['jump_indicator'] = self.jump_indicator(prices)
                
                # 资金流因子
                results['smart_money_flow'] = self.smart_money_flow(
                    prices, volumes, amounts, directions
                )
                
        except Exception as e:
            logger.error(f"因子计算异常 {symbol}: {e}")
            
        return results
