"""
盘中流计算引擎
- Tick级特征实时更新
- 滑动窗口聚合
- 多时间粒度特征 (tick / 1s / 3s / 1min)
- 事件驱动架构
"""
import time
import threading
import numpy as np
from typing import Dict, List, Callable, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """流事件"""
    event_type: str          # tick / orderbook / factor_update / signal
    symbol: str
    timestamp: float
    data: Any


@dataclass
class WindowState:
    """滑动窗口状态 (每只股票独立维护)"""
    # 价格序列
    prices: deque = field(default_factory=lambda: deque(maxlen=10000))
    volumes: deque = field(default_factory=lambda: deque(maxlen=10000))
    amounts: deque = field(default_factory=lambda: deque(maxlen=10000))
    directions: deque = field(default_factory=lambda: deque(maxlen=10000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=10000))

    # 聚合统计 (增量更新)
    tick_count: int = 0
    total_volume: int = 0
    total_amount: float = 0.0
    buy_volume: int = 0
    sell_volume: int = 0
    vwap: float = 0.0
    high: float = 0.0
    low: float = float("inf")
    open_price: float = 0.0
    last_price: float = 0.0

    # 1分钟K线缓存
    minute_bars: deque = field(default_factory=lambda: deque(maxlen=300))
    current_minute: int = -1
    minute_open: float = 0.0
    minute_high: float = 0.0
    minute_low: float = float("inf")
    minute_volume: int = 0


class StreamComputeEngine:
    """流计算引擎"""

    def __init__(self, config: dict, redis_cache=None):
        self.config = config
        self.redis = redis_cache

        # 每只股票的窗口状态
        self._states: Dict[str, WindowState] = defaultdict(WindowState)

        # 事件处理器注册表
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)

        # 因子计算器
        self._factor_calculators: List[Callable] = []

        # 运行状态
        self._running = False
        self._event_queue: deque = deque(maxlen=100000)
        self._lock = threading.Lock()

        # 性能统计
        self._stats = {
            "events_processed": 0,
            "avg_latency_us": 0,
            "max_latency_us": 0,
        }

    def register_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        self._handlers[event_type].append(handler)

    def register_factor_calculator(self, calculator: Callable):
        """注册因子计算器"""
        self._factor_calculators.append(calculator)

    def start(self):
        self._running = True
        self._process_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="stream_engine"
        )
        self._process_thread.start()
        logger.info("⚡ 流计算引擎已启动")

    def stop(self):
        self._running = False
        logger.info("流计算引擎已停止")

    # ==================== 数据输入 ====================

    def on_tick(self, symbol: str, price: float, volume: int,
                amount: float, direction: int, timestamp: float):
        """
        接收Tick数据 (由L2 Collector调用)
        这是最高频的入口, 必须极低延迟
        """
        t0 = time.perf_counter_ns()

        state = self._states[symbol]

        # 更新窗口状态 (增量)
        state.prices.append(price)
        state.volumes.append(volume)
        state.amounts.append(amount)
        state.directions.append(direction)
        state.timestamps.append(timestamp)

        state.tick_count += 1
        state.total_volume += volume
        state.total_amount += amount
        state.last_price = price

        if state.open_price == 0:
            state.open_price = price
        if price > state.high:
            state.high = price
        if price < state.low:
            state.low = price

        # VWAP增量更新
        if state.total_volume > 0:
            state.vwap = state.total_amount / state.total_volume

        # 买卖量统计
        if direction == 1:
            state.buy_volume += volume
        elif direction == -1:
            state.sell_volume += volume

        # 1分钟K线聚合
        self._update_minute_bar(state, price, volume, timestamp)

        # 触发Tick级因子计算
        self._compute_tick_features(symbol, state)

        # 性能统计
        latency_us = (time.perf_counter_ns() - t0) / 1000
        self._stats["events_processed"] += 1
        self._stats["max_latency_us"] = max(
            self._stats["max_latency_us"], latency_us
        )

    def on_orderbook(self, symbol: str, snapshot: dict):
        """接收盘口快照"""
        event = StreamEvent(
            event_type="orderbook",
            symbol=symbol,
            timestamp=time.time(),
            data=snapshot,
        )
        self._event_queue.append(event)

    # ==================== Tick级特征计算 ====================

    def _compute_tick_features(self, symbol: str, state: WindowState):
        """
        Tick级实时特征计算
        每个Tick到达时立即更新
        """
        features = {}

        n = len(state.prices)
        if n < 20:
            return

        prices = np.array(state.prices, dtype=np.float64)
        volumes = np.array(state.volumes, dtype=np.float64)
        directions = np.array(state.directions, dtype=np.float64)

        # ---- 实时VWAP偏离 ----
        features["vwap_bias"] = (
            (state.last_price - state.vwap) / state.vwap
            if state.vwap > 0 else 0.0
        )

        # ---- 买卖力量比 ----
        total_vol = state.buy_volume + state.sell_volume
        features["buy_sell_ratio"] = (
            state.buy_volume / total_vol if total_vol > 0 else 0.5
        )

        # ---- 短期动量 (最近100tick) ----
        window = min(100, n)

        recent_prices = prices[-window:]
        features["tick_momentum_100"] = (
            (recent_prices[-1] - recent_prices[0]) / (recent_prices[0] + 1e-8)
        )

        # ---- 短期波动率 (最近200tick) ----
        window_vol = min(200, n)
        if window_vol > 10:
            recent_p = prices[-window_vol:]
            log_ret = np.diff(np.log(recent_p + 1e-8))
            features["tick_volatility_200"] = float(np.std(log_ret))
        else:
            features["tick_volatility_200"] = 0.0

        # ---- 成交加速度 (最近50tick vs 前50tick) ----
        if n >= 100:
            recent_vol = np.sum(volumes[-50:])
            prev_vol = np.sum(volumes[-100:-50])
            features["volume_acceleration"] = (
                (recent_vol - prev_vol) / (prev_vol + 1e-8)
            )
        else:
            features["volume_acceleration"] = 0.0

        # ---- 大单占比 (单笔 > 均值*3) ----
        if n >= 50:
            recent_amounts = np.array(state.amounts)[-200:]
            mean_amt = np.mean(recent_amounts)
            large_mask = recent_amounts > mean_amt * 3
            features["large_order_ratio"] = (
                float(np.sum(recent_amounts[large_mask])) /
                (float(np.sum(recent_amounts)) + 1e-8)
            )
        else:
            features["large_order_ratio"] = 0.0

        # ---- 净主动买入比例 (最近500tick) ----
        window_flow = min(500, n)
        recent_dir = directions[-window_flow:]
        recent_vol = volumes[-window_flow:]
        net_buy = np.sum(recent_vol[recent_dir == 1])
        net_sell = np.sum(recent_vol[recent_dir == -1])
        features["net_buy_ratio"] = (
            (net_buy - net_sell) / (net_buy + net_sell + 1e-8)
        )

        # ---- 价格分布偏度 (最近300tick) ----
        if n >= 50:
            window_skew = min(300, n)
            rp = prices[-window_skew:]
            mean_p = np.mean(rp)
            std_p = np.std(rp)
            if std_p > 1e-8:
                features["price_skewness"] = float(
                    np.mean(((rp - mean_p) / std_p) ** 3)
                )
            else:
                features["price_skewness"] = 0.0
        else:
            features["price_skewness"] = 0.0

        # ---- 成交节奏 (tick间隔标准差) ----
        if n >= 20:
            ts = np.array(state.timestamps)[-100:]
            intervals = np.diff(ts)
            if len(intervals) > 5:
                features["tick_interval_std"] = float(np.std(intervals))
                features["tick_interval_mean"] = float(np.mean(intervals))
            else:
                features["tick_interval_std"] = 0.0
                features["tick_interval_mean"] = 0.0
        else:
            features["tick_interval_std"] = 0.0
            features["tick_interval_mean"] = 0.0

        # ---- 1分钟K线特征 ----
        if len(state.minute_bars) >= 5:
            bars = list(state.minute_bars)
            bar_closes = np.array([b["close"] for b in bars[-20:]])
            bar_volumes = np.array([b["volume"] for b in bars[-20:]])

            features["bar_momentum_5m"] = (
                (bar_closes[-1] - bar_closes[-min(5, len(bar_closes))]) /
                (bar_closes[-min(5, len(bar_closes))] + 1e-8)
            )
            features["bar_vol_ratio"] = (
                bar_volumes[-1] / (np.mean(bar_volumes[:-1]) + 1e-8)
                if len(bar_volumes) > 1 else 1.0
            )

        # 写入Redis
        if self.redis:
            self.redis.set_factors(symbol, features, expire=60)

        # 追加到因子历史 (供Transformer使用)
        if self.redis and state.tick_count % 10 == 0:  # 每10个tick追加一次
            factor_array = np.array(
                list(features.values()), dtype=np.float32
            )
            self.redis.append_factor_history(symbol, factor_array)

        # 触发外部因子计算器
        for calc in self._factor_calculators:
            try:
                extra_features = calc(symbol, state, features)
                if extra_features:
                    features.update(extra_features)
            except Exception as e:
                logger.error(f"外部因子计算异常: {e}")

        # 分发事件
        for handler in self._handlers.get("factor_update", []):
            try:
                handler(StreamEvent(
                    event_type="factor_update",
                    symbol=symbol,
                    timestamp=time.time(),
                    data=features,
                ))
            except Exception as e:
                logger.error(f"因子事件处理异常: {e}")

    # ==================== 1分钟K线聚合 ====================

    def _update_minute_bar(self, state: WindowState, price: float,
                            volume: int, timestamp: float):
        """增量更新1分钟K线"""
        dt = datetime.fromtimestamp(timestamp)
        current_minute = dt.hour * 60 + dt.minute

        if current_minute != state.current_minute:
            # 新的一分钟, 保存上一根K线
            if state.current_minute >= 0 and state.minute_volume > 0:
                bar = {
                    "minute": state.current_minute,
                    "open": state.minute_open,
                    "high": state.minute_high,
                    "low": state.minute_low,
                    "close": state.last_price,
                    "volume": state.minute_volume,
                    "timestamp": timestamp,
                }
                state.minute_bars.append(bar)

            # 重置
            state.current_minute = current_minute
            state.minute_open = price
            state.minute_high = price
            state.minute_low = price
            state.minute_volume = volume
        else:
            # 同一分钟, 更新
            if price > state.minute_high:
                state.minute_high = price
            if price < state.minute_low:
                state.minute_low = price
            state.minute_volume += volume

    # ==================== 事件处理循环 ====================

    def _process_loop(self):
        """处理非Tick事件 (盘口快照等, 频率较低)"""
        while self._running:
            try:
                if self._event_queue:
                    event = self._event_queue.popleft()
                    handlers = self._handlers.get(event.event_type, [])
                    for handler in handlers:
                        handler(event)
                else:
                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"事件处理异常: {e}")
                time.sleep(0.01)

    # ==================== 状态查询 ====================

    def get_state(self, symbol: str) -> Optional[WindowState]:
        return self._states.get(symbol)

    def get_all_features(self, symbol: str) -> Dict[str, float]:
        """获取某只股票的全部实时特征"""
        if self.redis:
            return self.redis.get_factors(symbol)
        return {}

    def get_stats(self) -> dict:
        return dict(self._stats)

    def reset_daily(self):
        """日内状态重置"""
        self._states.clear()
        self._stats = {
            "events_processed": 0,
            "avg_latency_us": 0,
            "max_latency_us": 0,
        }
        logger.info("流计算引擎日内状态已重置")
