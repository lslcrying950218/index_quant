"""
QMT/MiniQMT 交易接口封装
- 连接管理
- 下单/撤单/查询
- 回调处理
"""
import time
import threading
from enum import Enum
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ============ QMT SDK 导入 ============
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount, XtOrder, XtPosition


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class OrderRequest:
    """下单请求"""
    symbol: str
    side: OrderSide
    price: float
    volume: int
    order_type: int = 23       # 23=限价单, 24=市价单(最优五档)
    strategy_name: str = ""
    remark: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrderRecord:
    """订单记录"""
    order_id: int
    request: OrderRequest
    status: OrderStatus = OrderStatus.PENDING
    filled_volume: int = 0
    filled_amount: float = 0.0
    avg_price: float = 0.0
    commission: float = 0.0
    submit_time: float = 0.0
    update_time: float = 0.0
    error_msg: str = ""


class QMTCallback(XtQuantTraderCallback):
    """QMT 回调处理器"""

    def __init__(self, order_manager: 'QMTOrderManager'):
        super().__init__()
        self.order_manager = order_manager

    def on_disconnected(self):
        logger.error("⚠️ QMT 连接断开!")
        self.order_manager._on_disconnect()

    def on_stock_order(self, order: XtOrder):
        """委托回报"""
        logger.info(
            f"委托回报: {order.stock_code} 状态={order.order_status} "
            f"已成={order.traded_volume}/{order.order_volume}"
        )
        self.order_manager._on_order_update(order)

    def on_stock_trade(self, trade):
        """成交回报"""
        logger.info(
            f"成交回报: {trade.stock_code} 价格={trade.traded_price} "
            f"数量={trade.traded_volume}"
        )
        self.order_manager._on_trade(trade)

    def on_order_error(self, order_error):
        """下单失败"""
        logger.error(
            f"下单失败: {order_error.stock_code} "
            f"错误={order_error.error_msg}"
        )
        self.order_manager._on_order_error(order_error)

    def on_order_stock_async_response(self, response):
        """异步下单回报"""
        logger.debug(f"异步回报: order_id={response.order_id}")


class QMTOrderManager:
    """
    QMT 订单管理器
    - 连接维护 & 自动重连
    - 订单状态机
    - 持仓/资金查询
    """

    def __init__(self, config: dict):
        self.config = config
        self.qmt_path = config['path']
        self.account_id = config['account']
        self.session_id = config.get('session_id', int(time.time()))

        self.trader: Optional[XtQuantTrader] = None
        self.account: Optional[StockAccount] = None
        self.callback: Optional[QMTCallback] = None

        # 订单管理
        self._orders: Dict[int, OrderRecord] = {}
        self._lock = threading.Lock()
        self._connected = False

        # 外部回调
        self._fill_callbacks: List[Callable] = []
        self._status_callbacks: List[Callable] = []

    # ==================== 连接管理 ====================

    def connect(self) -> bool:
        """连接QMT"""
        try:
            self.account = StockAccount(self.account_id)
            self.callback = QMTCallback(self)

            self.trader = XtQuantTrader(self.qmt_path, self.session_id)
            self.trader.register_callback(self.callback)
            self.trader.start()

            # 等待连接建立
            retry = 0
            while retry < 30:
                connect_result = self.trader.connect()
                if connect_result == 0:
                    self._connected = True
                    logger.info("✅ QMT 连接成功")

                    # 订阅账户信息
                    self.trader.subscribe(self.account)
                    return True
                retry += 1
                time.sleep(1)

            logger.error("❌ QMT 连接超时")
            return False

        except Exception as e:
            logger.error(f"QMT 连接异常: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.trader:
            self.trader.stop()
            self._connected = False
            logger.info("QMT 已断开")

    def _on_disconnect(self):
        """断线重连"""
        self._connected = False
        logger.warning("触发自动重连...")
        threading.Thread(target=self._reconnect_loop, daemon=True).start()

    def _reconnect_loop(self):
        for attempt in range(10):
            time.sleep(3 * (attempt + 1))
            if self.connect():
                return
        logger.critical("重连失败, 请人工介入!")

    # ==================== 下单接口 ====================

    def place_order(self, request: OrderRequest) -> Optional[int]:
        """
        下单
        返回 order_id, 失败返回 None
        """
        if not self._connected:
            logger.error("QMT 未连接, 无法下单")
            return None

        try:
            if request.side == OrderSide.BUY:
                order_id = self.trader.order_stock(
                    self.account,
                    request.symbol,
                    xtdata.STOCK_BUY,
                    request.volume,
                    request.order_type,
                    request.price,
                    request.strategy_name,
                    request.remark,
                )
            else:
                order_id = self.trader.order_stock(
                    self.account,
                    request.symbol,
                    xtdata.STOCK_SELL,
                    request.volume,
                    request.order_type,
                    request.price,
                    request.strategy_name,
                    request.remark,
                )

            if order_id and order_id > 0:
                record = OrderRecord(
                    order_id=order_id,
                    request=request,
                    status=OrderStatus.SUBMITTED,
                    submit_time=time.time(),
                )
                with self._lock:
                    self._orders[order_id] = record

                logger.info(
                    f"📤 下单成功: {request.symbol} "
                    f"{request.side.value} {request.volume}股 "
                    f"@{request.price} id={order_id}"
                )
                return order_id
            else:
                logger.error(f"下单返回异常 id={order_id}")
                return None

        except Exception as e:
            logger.error(f"下单异常: {e}")
            return None

    def cancel_order(self, order_id: int) -> bool:
        """撤单"""
        if not self._connected:
            return False
        try:
            result = self.trader.cancel_order_stock(self.account, order_id)
            logger.info(f"撤单请求: order_id={order_id}, result={result}")
            return result == 0
        except Exception as e:
            logger.error(f"撤单异常: {e}")
            return False

    def cancel_all(self, symbol: Optional[str] = None):
        """撤销所有挂单"""
        with self._lock:
            for oid, record in self._orders.items():
                if record.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED):
                    if symbol is None or record.request.symbol == symbol:
                        self.cancel_order(oid)

    # ==================== 查询接口 ====================

    def get_positions(self) -> List[XtPosition]:
        """查询持仓"""
        if not self._connected:
            return []
        return self.trader.query_stock_positions(self.account)

    def get_position_dict(self) -> Dict[str, XtPosition]:
        """持仓字典 {symbol: position}"""
        positions = self.get_positions()
        return {p.stock_code: p for p in positions if p.volume > 0}

    def get_balance(self) -> dict:
        """查询资金"""
        if not self._connected:
            return {}
        asset = self.trader.query_stock_asset(self.account)
        if asset:
            return {
                'total_asset': asset.total_asset,
                'cash': asset.cash,
                'market_value': asset.market_value,
                'frozen_cash': asset.frozen_cash,
            }
        return {}

    def get_active_orders(self) -> List[OrderRecord]:
        """获取活跃订单"""
        with self._lock:
            return [
                r for r in self._orders.values()
                if r.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED)
            ]

    # ==================== 回调处理 ====================

    def _on_order_update(self, order: XtOrder):
        with self._lock:
            if order.order_id in self._orders:
                record = self._orders[order.order_id]
                record.filled_volume = order.traded_volume
                record.update_time = time.time()

                if order.order_status == 56:  # 已成
                    record.status = OrderStatus.FILLED
                elif order.order_status == 54:  # 部成
                    record.status = OrderStatus.PARTIAL_FILLED
                elif order.order_status == 50:  # 已报
                    record.status = OrderStatus.SUBMITTED
                elif order.order_status in (51, 52):  # 已撤
                    record.status = OrderStatus.CANCELLED
                elif order.order_status == 57:  # 废单
                    record.status = OrderStatus.REJECTED

        for cb in self._status_callbacks:
            cb(order)

    def _on_trade(self, trade):
        with self._lock:
            if trade.order_id in self._orders:
                record = self._orders[trade.order_id]
                record.filled_volume = trade.traded_volume
                record.avg_price = trade.traded_price
                record.filled_amount += trade.traded_price * trade.traded_volume
                record.update_time = time.time()

        for cb in self._fill_callbacks:
            cb(trade)

    def _on_order_error(self, error):
        with self._lock:
            if error.order_id in self._orders:
                self._orders[error.order_id].status = OrderStatus.FAILED
                self._orders[error.order_id].error_msg = error.error_msg

    # ==================== 回调注册 ====================

    def on_fill(self, callback: Callable):
        self._fill_callbacks.append(callback)

    def on_status_change(self, callback: Callable):
        self._status_callbacks.append(callback)
