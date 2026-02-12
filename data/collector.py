"""
Level-2 数据采集模块
- 逐笔委托/成交
- 十档盘口快照
- 委托队列
"""
import time
import threading
import numpy as np
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """逐笔成交数据"""
    symbol: str
    timestamp: int          # 微秒时间戳
    price: float
    volume: int
    amount: float
    direction: int          # 1=买, -1=卖, 0=中性
    order_id: int
    trade_type: int         # 0=普通, 1=撤单


@dataclass
class OrderBookSnapshot:
    """十档盘口快照"""
    symbol: str
    timestamp: int
    ask_prices: np.ndarray   # shape=(10,)
    ask_volumes: np.ndarray
    bid_prices: np.ndarray
    bid_volumes: np.ndarray
    ask_orders_count: np.ndarray  # 各档委托笔数
    bid_orders_count: np.ndarray
    total_ask_vol: int
    total_bid_vol: int
    last_price: float
    open_price: float
    high_price: float
    low_price: float
    turnover: float
    volume: int


@dataclass
class OrderQueue:
    """委托队列数据 (买一/卖一前50笔)"""
    symbol: str
    timestamp: int
    ask1_queue: List[int]    # 卖一队列各笔委托量
    bid1_queue: List[int]    # 买一队列各笔委托量


class L2DataCollector:
    """Level-2 数据采集器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.symbols: List[str] = []
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._running = False
        self._buffer = defaultdict(list)
        self._lock = threading.Lock()
        
    def subscribe(self, symbols: List[str]):
        """订阅股票列表"""
        self.symbols = symbols
        logger.info(f"订阅 {len(symbols)} 只股票的L2数据")
        
    def register_callback(self, event_type: str, callback: Callable):
        """
        注册回调函数
        event_type: 'tick' | 'orderbook' | 'order_queue'
        """
        self._callbacks[event_type].append(callback)
        
    def start(self):
        """启动数据采集"""
        self._running = True
        
        # 启动逐笔数据接收线程
        self._tick_thread = threading.Thread(
            target=self._receive_tick_data, daemon=True
        )
        self._tick_thread.start()
        
        # 启动快照数据接收线程
        self._snapshot_thread = threading.Thread(
            target=self._receive_snapshot_data, daemon=True
        )
        self._snapshot_thread.start()
        
        # 启动数据落盘线程
        self._flush_thread = threading.Thread(
            target=self._flush_buffer, daemon=True
        )
        self._flush_thread.start()
        
        logger.info("L2数据采集器已启动")
        
    def stop(self):
        self._running = False
        logger.info("L2数据采集器已停止")
        
    def _receive_tick_data(self):
        """接收逐笔成交数据 (对接实际L2数据源)"""
        # ========================================
        # 这里对接你的L2数据源SDK
        # 常见数据源: 恒生极速行情、华锐ATP、CTP等
        # ========================================
        while self._running:
            try:
                # 示例: 从数据源SDK获取逐笔数据
                # raw_data = self.l2_sdk.recv_tick()
                # tick = self._parse_tick(raw_data)
                
                # 分发到回调
                # for cb in self._callbacks.get('tick', []):
                #     cb(tick)
                
                # 写入缓冲区
                # with self._lock:
                #     self._buffer['tick'].append(tick)
                
                time.sleep(0.0001)  # 占位, 实际由SDK驱动
            except Exception as e:
                logger.error(f"Tick数据接收异常: {e}")
                
    def _receive_snapshot_data(self):
        """接收盘口快照数据"""
        while self._running:
            try:
                # snapshot = self.l2_sdk.recv_snapshot()
                # parsed = self._parse_snapshot(snapshot)
                # for cb in self._callbacks.get('orderbook', []):
                #     cb(parsed)
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"快照数据接收异常: {e}")
    
    def _flush_buffer(self):
        """定期将缓冲区数据落盘到ClickHouse"""
        while self._running:
            time.sleep(1.0)  # 每秒落盘一次
            with self._lock:
                if self._buffer:
                    buffer_copy = dict(self._buffer)
                    self._buffer.clear()
            # 异步写入ClickHouse
            # self.storage.batch_insert(buffer_copy)
            
    @staticmethod
    def _parse_tick(raw) -> TickData:
        """解析原始逐笔数据"""
        return TickData(
            symbol=raw['code'],
            timestamp=raw['time'],
            price=raw['price'],
            volume=raw['vol'],
            amount=raw['amount'],
            direction=raw.get('bsFlag', 0),
            order_id=raw.get('orderId', 0),
            trade_type=raw.get('tradeType', 0),
        )
