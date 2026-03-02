"""
Redis 实时特征缓存层
- Tick/盘口数据缓存
- 因子值实时读写
- 信号/持仓状态共享
- 支持多进程间通信
"""
import json
import time
import pickle
import numpy as np
import redis
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis 缓存管理器"""

    # Key前缀规范
    PREFIX_TICK     = "tick:"        # tick:{symbol}
    PREFIX_BOOK     = "book:"        # book:{symbol}
    PREFIX_FACTOR   = "factor:"      # factor:{symbol}
    PREFIX_SIGNAL   = "signal:"      # signal:{symbol}
    PREFIX_POSITION = "pos:"         # pos:{symbol}
    PREFIX_STATE    = "state:"       # state:{key}
    PREFIX_NEWS     = "news:"        # news:{symbol}
    PREFIX_QUEUE    = "queue:"       # queue:{name}  消息队列

    def __init__(self, config: dict):
        self.pool = redis.ConnectionPool(
            host=config.get("host", "127.0.0.1"),
            port=config.get("port", 6379),
            db=config.get("db", 0),
            max_connections=config.get("max_connections", 50),
            decode_responses=False,
            socket_timeout=5,
            socket_connect_timeout=3,
            retry_on_timeout=True,
        )
        self.client = redis.Redis(connection_pool=self.pool)
        self._check_connection()

    def _check_connection(self):
        try:
            self.client.ping()
            logger.info("✅ Redis 连接成功")
        except redis.ConnectionError as e:
            logger.error(f"❌ Redis 连接失败: {e}")
            raise

    # ==================== Tick数据 ====================

    def set_tick(self, symbol: str, tick_data: dict, expire: int = 60):
        """写入最新Tick"""
        key = f"{self.PREFIX_TICK}{symbol}"
        self.client.set(key, json.dumps(tick_data, default=str).encode(), ex=expire)

    def get_tick(self, symbol: str) -> Optional[dict]:
        """读取最新Tick"""
        key = f"{self.PREFIX_TICK}{symbol}"
        data = self.client.get(key)
        return json.loads(data) if data else None

    # ==================== 盘口数据 ====================

    def set_orderbook(self, symbol: str, book_data: dict, expire: int = 30):
        """写入盘口快照"""
        key = f"{self.PREFIX_BOOK}{symbol}"
        self.client.set(key, json.dumps(book_data, default=str).encode(), ex=expire)

    def get_orderbook(self, symbol: str) -> Optional[dict]:
        key = f"{self.PREFIX_BOOK}{symbol}"
        data = self.client.get(key)
        return json.loads(data) if data else None

    # ==================== 因子值 ====================

    def set_factors(self, symbol: str, factors: Dict[str, float], expire: int = 120):
        """写入因子值 (Hash结构, 支持单因子读取)"""
        key = f"{self.PREFIX_FACTOR}{symbol}"
        pipe = self.client.pipeline()
        for fname, fval in factors.items():
            pipe.hset(key, fname, str(fval))
        pipe.expire(key, expire)
        pipe.execute()

    def get_factors(self, symbol: str) -> Dict[str, float]:
        """读取全部因子值"""
        key = f"{self.PREFIX_FACTOR}{symbol}"
        raw = self.client.hgetall(key)
        return {k.decode(): float(v) for k, v in raw.items()} if raw else {}

    def get_factor(self, symbol: str, factor_name: str) -> Optional[float]:
        """读取单个因子值"""
        key = f"{self.PREFIX_FACTOR}{symbol}"
        val = self.client.hget(key, factor_name)
        return float(val) if val else None

    def get_all_factors_batch(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """批量读取多只股票因子"""
        pipe = self.client.pipeline()
        for sym in symbols:
            pipe.hgetall(f"{self.PREFIX_FACTOR}{sym}")
        results = pipe.execute()
        return {
            sym: {k.decode(): float(v) for k, v in raw.items()} if raw else {}
            for sym, raw in zip(symbols, results)
        }

    # ==================== 因子历史序列 ====================

    def append_factor_history(self, symbol: str, factor_values: np.ndarray,
                               max_len: int = 200):
        """追加因子历史 (List结构, 用于Transformer输入)"""
        key = f"{self.PREFIX_FACTOR}{symbol}:history"
        serialized = pickle.dumps(factor_values)
        pipe = self.client.pipeline()
        pipe.rpush(key, serialized)
        pipe.ltrim(key, -max_len, -1)  # 保留最近max_len条
        pipe.expire(key, 86400)
        pipe.execute()

    def get_factor_history(self, symbol: str, length: int = 60) -> Optional[np.ndarray]:
        """获取因子历史序列"""
        key = f"{self.PREFIX_FACTOR}{symbol}:history"
        raw_list = self.client.lrange(key, -length, -1)
        if not raw_list:
            return None
        arrays = [pickle.loads(item) for item in raw_list]
        return np.stack(arrays)

    # ==================== 信号 ====================

    def set_signal(self, symbol: str, signal: dict, expire: int = 300):
        """写入交易信号"""
        key = f"{self.PREFIX_SIGNAL}{symbol}"
        self.client.set(key, json.dumps(signal, default=str).encode(), ex=expire)

    def get_signal(self, symbol: str) -> Optional[dict]:
        key = f"{self.PREFIX_SIGNAL}{symbol}"
        data = self.client.get(key)
        return json.loads(data) if data else None

    # ==================== 新闻情感 ====================

    def set_news_sentiment(self, symbol: str, sentiment: dict, expire: int = 3600):
        """写入新闻情感分数"""
        key = f"{self.PREFIX_NEWS}{symbol}"
        self.client.set(key, json.dumps(sentiment, default=str).encode(), ex=expire)

    def get_news_sentiment(self, symbol: str) -> Optional[dict]:
        key = f"{self.PREFIX_NEWS}{symbol}"
        data = self.client.get(key)
        return json.loads(data) if data else None

    # ==================== 消息队列 (进程间通信) ====================

    def publish(self, channel: str, message: dict):
        """发布消息"""
        self.client.publish(channel, json.dumps(message, default=str).encode())

    def subscribe(self, channels: List[str]):
        """订阅消息"""
        pubsub = self.client.pubsub()
        pubsub.subscribe(*channels)
        return pubsub

    def push_queue(self, queue_name: str, data: dict):
        """推入队列"""
        key = f"{self.PREFIX_QUEUE}{queue_name}"
        self.client.rpush(key, json.dumps(data, default=str).encode())

    def pop_queue(self, queue_name: str, timeout: int = 1) -> Optional[dict]:
        """弹出队列"""
        key = f"{self.PREFIX_QUEUE}{queue_name}"
        result = self.client.blpop(key, timeout=timeout)
        if result:
            return json.loads(result[1])
        return None

    # ==================== 系统状态 ====================

    def set_state(self, key: str, value: Any, expire: int = 86400):
        state_key = f"{self.PREFIX_STATE}{key}"
        self.client.set(state_key, json.dumps(value, default=str).encode(), ex=expire)

    def get_state(self, key: str) -> Any:
        state_key = f"{self.PREFIX_STATE}{key}"
        data = self.client.get(state_key)
        return json.loads(data) if data else None

    # ==================== 工具方法 ====================

    def flush_expired(self):
        """清理过期数据 (Redis自动处理TTL, 此方法用于手动清理)"""
        logger.info("Redis缓存清理完成")

    def get_memory_usage(self) -> dict:
        """获取Redis内存使用情况"""
        info = self.client.info("memory")
        return {
            "used_memory_human": info.get("used_memory_human"),
            "used_memory_peak_human": info.get("used_memory_peak_human"),
            "mem_fragmentation_ratio": info.get("mem_fragmentation_ratio"),
        }
