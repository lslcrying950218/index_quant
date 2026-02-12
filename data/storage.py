"""
ClickHouse 时序数据存储
- 高性能列式存储
- 支持海量L2数据
"""
from clickhouse_driver import Client
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ClickHouseStorage:
    """ClickHouse 存储管理器"""
    
    def __init__(self, config: dict):
        self.client = Client(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password'],
            settings={'max_block_size': 100000}
        )
        self._init_tables()
        
    def _init_tables(self):
        """初始化数据表"""
        
        # 逐笔成交表
        self.client.execute("""
            CREATE TABLE IF NOT EXISTS tick_data (
                symbol      LowCardinality(String),
                timestamp   DateTime64(6, 'Asia/Shanghai'),
                price       Float64,
                volume      UInt32,
                amount      Float64,
                direction   Int8,
                order_id    UInt64,
                trade_type  UInt8
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMMDD(timestamp)
            ORDER BY (symbol, timestamp)
            TTL timestamp + INTERVAL 90 DAY
            SETTINGS index_granularity = 8192
        """)
        
        # 十档盘口快照表
        self.client.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_snapshot (
                symbol          LowCardinality(String),
                timestamp       DateTime64(6, 'Asia/Shanghai'),
                last_price      Float64,
                ask_price_1     Float64, ask_vol_1   UInt32,
                ask_price_2     Float64, ask_vol_2   UInt32,
                ask_price_3     Float64, ask_vol_3   UInt32,
                ask_price_4     Float64, ask_vol_4   UInt32,
                ask_price_5     Float64, ask_vol_5   UInt32,
                ask_price_6     Float64, ask_vol_6   UInt32,
                ask_price_7     Float64, ask_vol_7   UInt32,
                ask_price_8     Float64, ask_vol_8   UInt32,
                ask_price_9     Float64, ask_vol_9   UInt32,
                ask_price_10    Float64, ask_vol_10  UInt32,
                bid_price_1     Float64, bid_vol_1   UInt32,
                bid_price_2     Float64, bid_vol_2   UInt32,
                bid_price_3     Float64, bid_vol_3   UInt32,
                bid_price_4     Float64, bid_vol_4   UInt32,
                bid_price_5     Float64, bid_vol_5   UInt32,
                bid_price_6     Float64, bid_vol_6   UInt32,
                bid_price_7     Float64, bid_vol_7   UInt32,
                bid_price_8     Float64, bid_vol_8   UInt32,
                bid_price_9     Float64, bid_vol_9   UInt32,
                bid_price_10    Float64, bid_vol_10  UInt32,
                total_ask_vol   UInt64,
                total_bid_vol   UInt64,
                turnover        Float64,
                total_volume    UInt64
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMMDD(timestamp)
            ORDER BY (symbol, timestamp)
            TTL timestamp + INTERVAL 180 DAY
        """)
        
        # 因子值表
        self.client.execute("""
            CREATE TABLE IF NOT EXISTS factor_values (
                symbol      LowCardinality(String),
                timestamp   DateTime64(3, 'Asia/Shanghai'),
                factor_name LowCardinality(String),
                value       Float64
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMMDD(timestamp)
            ORDER BY (symbol, timestamp, factor_name)
        """)
        
        logger.info("ClickHouse 数据表初始化完成")
        
    def batch_insert_ticks(self, ticks: list):
        """批量插入逐笔数据"""
        if not ticks:
            return
        data = [(t.symbol, t.timestamp, t.price, t.volume,
                 t.amount, t.direction, t.order_id, t.trade_type)
                for t in ticks]
        self.client.execute(
            "INSERT INTO tick_data VALUES", data
        )
        
    def query_ticks(self, symbol: str, start_time: str, end_time: str):
        """查询逐笔数据"""
        return self.client.execute(f"""
            SELECT * FROM tick_data
            WHERE symbol = '{symbol}'
              AND timestamp BETWEEN '{start_time}' AND '{end_time}'
            ORDER BY timestamp
        """)
        
    def query_snapshots(self, symbol: str, date: str):
        """查询某日全部快照"""
        return self.client.execute(f"""
            SELECT * FROM orderbook_snapshot
            WHERE symbol = '{symbol}'
              AND toDate(timestamp) = '{date}'
            ORDER BY timestamp
        """)
