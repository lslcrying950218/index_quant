"""
订单管理系统 (OMS)
- 订单全生命周期管理
- 持仓管理 & 成本计算
- 交易记录持久化
- 绩效归因
"""
import time
import uuid
import threading
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from collections import defaultdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OrderState(Enum):
    CREATED = "created"
    PENDING_RISK = "pending_risk"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    ALGO_TWAP = "algo_twap"
    ALGO_VWAP = "algo_vwap"
    ALGO_ICEBERG = "algo_iceberg"


@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    side: str                    # buy / sell
    price: float
    volume: int
    order_type: OrderType
    strategy_name: str = ""
    signal_strength: float = 0.0

    # 状态
    state: OrderState = OrderState.CREATED
    filled_volume: int = 0
    filled_amount: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0

    # 时间戳
    create_time: str = ""
    submit_time: str = ""
    fill_time: str = ""
    cancel_time: str = ""

    # 关联
    parent_order_id: str = ""    # 算法母单ID
    broker_order_id: int = 0     # 券商订单号
    error_msg: str = ""

    # 元数据
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.order_id:
            self.order_id = str(uuid.uuid4())[:12]
        if not self.create_time:
            self.create_time = datetime.now().isoformat()

    @property
    def remaining_volume(self) -> int:
        return self.volume - self.filled_volume

    @property
    def is_active(self) -> bool:
        return self.state in (
            OrderState.SUBMITTED, OrderState.PARTIAL_FILLED
        )

    @property
    def is_done(self) -> bool:
        return self.state in (
            OrderState.FILLED, OrderState.CANCELLED,
            OrderState.REJECTED, OrderState.FAILED, OrderState.EXPIRED,
        )

    @property
    def fill_rate(self) -> float:
        return self.filled_volume / self.volume if self.volume > 0 else 0


@dataclass
class Position:
    """持仓"""
    symbol: str
    volume: int = 0
    available_volume: int = 0     # 可卖数量 (T+1)
    avg_cost: float = 0.0        # 持仓均价
    total_cost: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0    # 已实现盈亏
    buy_time: str = ""           # 首次买入时间
    last_trade_time: str = ""

    def update_price(self, price: float):
        self.current_price = price
        self.market_value = self.volume * price
        if self.total_cost > 0:
            self.unrealized_pnl = self.market_value - self.total_cost
            self.unrealized_pnl_pct = self.unrealized_pnl / self.total_cost
        else:
            self.unrealized_pnl = 0
            self.unrealized_pnl_pct = 0

    def on_buy_fill(self, fill_volume: int, fill_price: float):
        """买入成交更新"""
        new_cost = fill_volume * fill_price
        self.total_cost += new_cost
        self.volume += fill_volume
        self.avg_cost = self.total_cost / self.volume if self.volume > 0 else 0
        self.last_trade_time = datetime.now().isoformat()
        if not self.buy_time:
            self.buy_time = self.last_trade_time

    def on_sell_fill(self, fill_volume: int, fill_price: float):
        """卖出成交更新"""
        sell_cost = fill_volume * self.avg_cost
        sell_revenue = fill_volume * fill_price
        self.realized_pnl += sell_revenue - sell_cost
        self.volume -= fill_volume
        self.total_cost = self.volume * self.avg_cost
        self.last_trade_time = datetime.now().isoformat()

        if self.volume <= 0:
            self.volume = 0
            self.total_cost = 0
            self.avg_cost = 0


@dataclass
class TradeRecord:
    """成交记录"""
    trade_id: str
    order_id: str
    symbol: str
    side: str
    price: float
    volume: int
    amount: float
    commission: float
    timestamp: str
    strategy_name: str = ""


class OrderManagementSystem:
    """订单管理系统"""

    # 交易费率
    COMMISSION_RATE = 0.00025     # 佣金万2.5
    MIN_COMMISSION = 5.0          # 最低5元
    STAMP_TAX_RATE = 0.0005       # 印花税万5 (仅卖出)
    TRANSFER_FEE_RATE = 0.00001   # 过户费万0.1

    def __init__(self, config: dict, storage=None):
        self.config = config
        self.storage = storage

        # 订单簿
        self._orders: Dict[str, Order] = {}
        # 持仓簿
        self._positions: Dict[str, Position] = {}
        # 成交记录
        self._trades: List[TradeRecord] = []
        # 日统计
        self._daily_stats = {
            "trade_count": 0,
            "buy_amount": 0.0,
            "sell_amount": 0.0,
            "commission_total": 0.0,
            "realized_pnl": 0.0,
        }

        self._lock = threading.Lock()
        self._trade_log_path = Path("logs/trades")
        self._trade_log_path.mkdir(parents=True, exist_ok=True)

    # ==================== 订单管理 ====================

    def create_order(self, symbol: str, side: str, price: float,
                     volume: int, order_type: OrderType = OrderType.LIMIT,
                     strategy_name: str = "",
                     signal_strength: float = 0.0,
                     **kwargs) -> Order:
        """创建订单"""
        order = Order(
            order_id="",
            symbol=symbol,
            side=side,
            price=price,
            volume=volume,
            order_type=order_type,
            strategy_name=strategy_name,
            signal_strength=signal_strength,
            tags=kwargs,
        )

        with self._lock:
            self._orders[order.order_id] = order

        logger.info(
            f"📝 订单创建: {order.order_id} {symbol} "
            f"{side} {volume}股 @{price:.2f} [{order_type.value}]"
        )
        return order

    def update_order_state(self, order_id: str, new_state: OrderState,
                            broker_order_id: int = 0, error_msg: str = ""):
        """更新订单状态"""
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.warning(f"订单不存在: {order_id}")
                return

            old_state = order.state
            order.state = new_state

            if new_state == OrderState.SUBMITTED:
                order.submit_time = datetime.now().isoformat()
                if broker_order_id:
                    order.broker_order_id = broker_order_id
            elif new_state in (OrderState.CANCELLED, OrderState.EXPIRED):
                order.cancel_time = datetime.now().isoformat()
            elif new_state in (OrderState.REJECTED, OrderState.FAILED):
                order.error_msg = error_msg

            logger.debug(
                f"订单状态更新: {order_id} {old_state.value} -> {new_state.value}"
            )

    def on_fill(self, order_id: str, fill_price: float, fill_volume: int):
        """
        成交回报处理

        核心方法: 更新订单、持仓、费用、统计
        """
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                logger.warning(f"成交回报: 订单不存在 {order_id}")
                return

            symbol = order.symbol
            side = order.side

            # 1. 更新订单
            order.filled_volume += fill_volume
            fill_amount = fill_price * fill_volume
            order.filled_amount += fill_amount

            if order.filled_volume >= order.volume:
                order.state = OrderState.FILLED
                order.fill_time = datetime.now().isoformat()
            else:
                order.state = OrderState.PARTIAL_FILLED

            order.avg_fill_price = (
                order.filled_amount / order.filled_volume
                if order.filled_volume > 0 else 0
            )

            # 2. 计算费用
            commission = max(fill_amount * self.COMMISSION_RATE, self.MIN_COMMISSION)
            transfer_fee = fill_amount * self.TRANSFER_FEE_RATE
            stamp_tax = fill_amount * self.STAMP_TAX_RATE if side == "sell" else 0
            total_fee = commission + transfer_fee + stamp_tax
            order.commission += total_fee

            # 3. 计算滑点
            if order.price > 0:
                if side == "buy":
                    order.slippage = (fill_price - order.price) / order.price
                else:
                    order.slippage = (order.price - fill_price) / order.price

            # 4. 更新持仓
            if symbol not in self._positions:
                self._positions[symbol] = Position(symbol=symbol)

            pos = self._positions[symbol]
            if side == "buy":
                pos.on_buy_fill(fill_volume, fill_price)
            else:
                pos.on_sell_fill(fill_volume, fill_price)

            # 清理空仓
            if pos.volume <= 0:
                del self._positions[symbol]

            # 5. 记录成交
            trade = TradeRecord(
                trade_id=str(uuid.uuid4())[:12],
                order_id=order_id,
                symbol=symbol,
                side=side,
                price=fill_price,
                volume=fill_volume,
                amount=fill_amount,
                commission=total_fee,
                timestamp=datetime.now().isoformat(),
                strategy_name=order.strategy_name,
            )
            self._trades.append(trade)

            # 6. 更新日统计
            self._daily_stats["trade_count"] += 1
            self._daily_stats["commission_total"] += total_fee
            if side == "buy":
                self._daily_stats["buy_amount"] += fill_amount
            else:
                self._daily_stats["sell_amount"] += fill_amount
                self._daily_stats["realized_pnl"] += (
                    fill_price - pos.avg_cost if symbol in self._positions else 0
                ) * fill_volume

        logger.info(
            f"💰 成交: {symbol} {side} {fill_volume}股 @{fill_price:.2f} "
            f"费用={total_fee:.2f} 滑点={order.slippage:.4%}"
        )

    # ==================== 持仓查询 ====================

    def get_positions(self) -> Dict[str, Position]:
        with self._lock:
            return dict(self._positions)

    def get_position(self, symbol: str) -> Optional[Position]:
        with self._lock:
            return self._positions.get(symbol)

    def update_market_prices(self, prices: Dict[str, float]):
        """批量更新持仓市值"""
        with self._lock:
            for sym, pos in self._positions.items():
                if sym in prices:
                    pos.update_price(prices[sym])

    def get_portfolio_summary(self) -> dict:
        """获取组合摘要"""
        with self._lock:
            positions = list(self._positions.values())

        total_market_value = sum(p.market_value for p in positions)
        total_cost = sum(p.total_cost for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_realized_pnl = sum(p.realized_pnl for p in positions)

        # 集中度
        if positions and total_market_value > 0:
            weights = [p.market_value / total_market_value for p in positions]
            max_weight = max(weights)
            hhi = sum(w ** 2 for w in weights)  # 赫芬达尔指数
        else:
            max_weight = 0
            hhi = 0

        return {
            "position_count": len(positions),
            "total_market_value": round(total_market_value, 2),
            "total_cost": round(total_cost, 2),
            "unrealized_pnl": round(total_unrealized_pnl, 2),
            "unrealized_pnl_pct": (
                round(total_unrealized_pnl / total_cost * 100, 2)
                if total_cost > 0 else 0
            ),
            "realized_pnl": round(total_realized_pnl, 2),
            "max_single_weight": round(max_weight * 100, 2),
            "hhi_concentration": round(hhi, 4),
            "daily_stats": dict(self._daily_stats),
        }

    # ==================== 订单查询 ====================

    def get_order(self, order_id: str) -> Optional[Order]:
        with self._lock:
            return self._orders.get(order_id)

    def get_active_orders(self) -> List[Order]:
        with self._lock:
            return [o for o in self._orders.values() if o.is_active]

    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        with self._lock:
            return [o for o in self._orders.values() if o.symbol == symbol]

    def get_today_trades(self) -> List[TradeRecord]:
        today_str = date.today().isoformat()
        with self._lock:
            return [
                t for t in self._trades
                if t.timestamp.startswith(today_str)
            ]

    # ==================== T+1可卖量管理 ====================

    def update_available_volumes(self):
        """
        盘前更新可卖数量
        A股T+1: 昨日及之前买入的才可卖
        """
        today_str = date.today().isoformat()
        with self._lock:
            for sym, pos in self._positions.items():
                # 今日买入量
                today_buy = sum(
                    t.volume for t in self._trades
                    if t.symbol == sym
                    and t.side == "buy"
                    and t.timestamp.startswith(today_str)
                )
                pos.available_volume = max(0, pos.volume - today_buy)

    # ==================== 持久化 ====================

    def save_trades_to_file(self):
        """保存今日成交记录到文件"""
        today = date.today().isoformat()
        trades = self.get_today_trades()
        if not trades:
            return

        filepath = self._trade_log_path / f"trades_{today}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            for trade in trades:
                f.write(json.dumps(asdict(trade), ensure_ascii=False) + "\n")

        logger.info(f"成交记录已保存: {filepath} ({len(trades)} 条)")

    def save_positions_snapshot(self):
        """保存持仓快照"""
        today = date.today().isoformat()
        positions = self.get_positions()

        filepath = self._trade_log_path / f"positions_{today}.json"
        data = {
            sym: {
                "volume": pos.volume,
                "avg_cost": pos.avg_cost,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "realized_pnl": pos.realized_pnl,
            }
            for sym, pos in positions.items()
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"持仓快照已保存: {filepath}")

    # ==================== 日终重置 ====================

    def end_of_day_reset(self):
        """日终处理"""
        # 保存数据
        self.save_trades_to_file()
        self.save_positions_snapshot()

        # 重置日统计
        self._daily_stats = {
            "trade_count": 0,
            "buy_amount": 0.0,
            "sell_amount": 0.0,
            "commission_total": 0.0,
            "realized_pnl": 0.0,
        }

        # 清理已完成订单 (保留最近500条)
        with self._lock:
            done_orders = {
                k: v for k, v in self._orders.items() if v.is_done
            }
            if len(done_orders) > 500:
                sorted_orders = sorted(
                    done_orders.items(),
                    key=lambda x: x[1].create_time,
                )
                for k, _ in sorted_orders[:-500]:
                    del self._orders[k]

            # 清理成交记录 (保留最近2000条)
            if len(self._trades) > 2000:
                self._trades = self._trades[-1500:]

        logger.info("OMS 日终重置完成")

    # ==================== 绩效归因 ====================

    def get_strategy_attribution(self) -> Dict[str, dict]:
        """按策略归因"""
        attribution = defaultdict(lambda: {
            "trade_count": 0,
            "buy_amount": 0.0,
            "sell_amount": 0.0,
            "commission": 0.0,
            "pnl": 0.0,
        })

        with self._lock:
            for trade in self._trades:
                strat = trade.strategy_name or "unknown"
                attr = attribution[strat]
                attr["trade_count"] += 1
                attr["commission"] += trade.commission
                if trade.side == "buy":
                    attr["buy_amount"] += trade.amount
                else:
                    attr["sell_amount"] += trade.amount

        return dict(attribution)

    def get_slippage_report(self) -> dict:
        """滑点分析报告"""
        with self._lock:
            filled_orders = [
                o for o in self._orders.values()
                if o.state == OrderState.FILLED and o.slippage != 0
            ]

        if not filled_orders:
            return {"avg_slippage_bps": 0, "count": 0}

        slippages = [o.slippage for o in filled_orders]
        return {
            "avg_slippage_bps": round(np.mean(slippages) * 10000, 2),
            "median_slippage_bps": round(np.median(slippages) * 10000, 2),
            "max_slippage_bps": round(max(slippages) * 10000, 2),
            "total_slippage_cost": round(
                sum(o.filled_amount * abs(o.slippage) for o in filled_orders), 2
            ),
            "count": len(filled_orders),
        }
