"""
实时风控引擎
- 盘前检查
- 盘中实时监控
- 熔断机制
"""
import time
import threading
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    HALT = "halt"


@dataclass
class RiskState:
    """风控状态"""
    level: RiskLevel = RiskLevel.NORMAL
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_value: float = 0.0
    total_trades_today: int = 0
    trades_per_minute: int = 0
    position_concentration: float = 0.0
    total_position_pct: float = 0.0
    violations: List[str] = field(default_factory=list)
    halted: bool = False
    last_update: float = 0.0


class RealTimeRiskEngine:
    """实时风控引擎"""

    def __init__(self, config: dict, order_manager=None, alert_manager=None):
        self.config = config
        self.om = order_manager
        self.alert = alert_manager

        # 风控参数
        self.total_capital = config['total_capital']
        self.max_position_pct = config['max_position_pct']
        self.max_total_position_pct = config['max_total_position_pct']
        self.max_daily_loss_pct = config['max_daily_loss_pct']
        self.max_total_drawdown_pct = config['max_total_drawdown_pct']
        self.max_order_per_minute = config['max_order_per_minute']
        self.min_holding_seconds = config['min_holding_seconds']
        self.stock_count_range = config['stock_count_range']
        self.banned_stocks = set(config.get('banned_stocks', []))

        # 状态
        self.state = RiskState(peak_value=self.total_capital)
        self._trade_timestamps: List[float] = []
        self._holding_start: Dict[str, float] = {}  # symbol -> 建仓时间
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self._monitor_thread.start()
        logger.info("🛡️ 实时风控引擎已启动")

    def stop(self):
        self._running = False

    # ==================== 盘前风控 ====================

    def pre_trade_check(self, symbol: str, side: str,
                        volume: int, price: float) -> Tuple[bool, str]:
        """
        盘前风控检查 (下单前必须通过)
        返回: (是否通过, 原因)
        """
        checks = [
            self._check_halted,
            lambda: self._check_banned(symbol),
            lambda: self._check_position_limit(symbol, side, volume, price),
            lambda: self._check_total_position(side, volume, price),
            lambda: self._check_order_frequency(),
            lambda: self._check_min_holding(symbol, side),
            lambda: self._check_daily_loss(),
            lambda: self._check_price_limit(symbol, price),
            lambda: self._check_volume_valid(volume),
        ]

        for check in checks:
            passed, reason = check() if callable(check) else check
            if not passed:
                self.state.violations.append(
                    f"{datetime.now():%H:%M:%S} {symbol} {reason}"
                )
                logger.warning(f"🚫 风控拦截: {symbol} {reason}")
                return False, reason

        return True, "OK"

    def _check_halted(self) -> Tuple[bool, str]:
        if self.state.halted:
            return False, "系统已熔断, 禁止交易"
        return True, ""

    def _check_banned(self, symbol: str) -> Tuple[bool, str]:
        if symbol in self.banned_stocks:
            return False, f"{symbol} 在禁止交易名单中"
        # ST股检查
        # if symbol_is_st(symbol):
        #     return False, f"{symbol} 为ST股, 禁止交易"
        return True, ""

    def _check_position_limit(self, symbol: str, side: str,
                               volume: int, price: float) -> Tuple[bool, str]:
        """单票仓位限制"""
        if side == "sell":
            return True, ""

        order_value = volume * price
        positions = self.om.get_position_dict() if self.om else {}

        current_value = 0
        if symbol in positions:
            pos = positions[symbol]
            current_value = pos.volume * pos.open_price  # 近似

        new_value = current_value + order_value
        max_value = self.total_capital * self.max_position_pct

        if new_value > max_value:
            return False, (
                f"单票仓位超限: {new_value:.0f} > {max_value:.0f} "
                f"({self.max_position_pct:.0%})"
            )
        return True, ""

    def _check_total_position(self, side: str, volume: int,
                               price: float) -> Tuple[bool, str]:
        """总仓位限制"""
        if side == "sell":
            return True, ""

        balance = self.om.get_balance() if self.om else {}
        market_value = balance.get('market_value', 0)
        new_total = market_value + volume * price
        max_total = self.total_capital * self.max_total_position_pct

        if new_total > max_total:
            return False, (
                f"总仓位超限: {new_total:.0f} > {max_total:.0f} "
                f"({self.max_total_position_pct:.0%})"
            )
        return True, ""

    def _check_order_frequency(self) -> Tuple[bool, str]:
        """下单频率限制"""
        now = time.time()
        # 清理1分钟前的记录
        self._trade_timestamps = [
            t for t in self._trade_timestamps if now - t < 60
        ]
        if len(self._trade_timestamps) >= self.max_order_per_minute:
            return False, (
                f"下单频率超限: {len(self._trade_timestamps)} >= "
                f"{self.max_order_per_minute}/min"
            )
        self._trade_timestamps.append(now)
        return True, ""

    def _check_min_holding(self, symbol: str, side: str) -> Tuple[bool, str]:
        """最短持仓时间"""
        if side != "sell":
            return True, ""
        if symbol in self._holding_start:
            held = time.time() - self._holding_start[symbol]
            if held < self.min_holding_seconds:
                return False, (
                    f"持仓时间不足: {held:.0f}s < {self.min_holding_seconds}s"
                )
        return True, ""

    def _check_daily_loss(self) -> Tuple[bool, str]:
        """日亏损限制"""
        if self.state.daily_pnl_pct < -self.max_daily_loss_pct:
            return False, (
                f"日亏损超限: {self.state.daily_pnl_pct:.2%} < "
                f"-{self.max_daily_loss_pct:.2%}"
            )
        return True, ""

    def _check_price_limit(self, symbol: str, price: float) -> Tuple[bool, str]:
        """价格合理性检查"""
        if price <= 0:
            return False, "价格不合法"
        if price > 9999:
            return False, "价格异常偏高"
        return True, ""

    def _check_volume_valid(self, volume: int) -> Tuple[bool, str]:
        """数量合法性"""
        if volume <= 0:
            return False, "数量不合法"
        if volume % 100 != 0:
            return False, "数量必须为100的整数倍"
        if volume > 100000:
            return False, "单笔数量超过10万股上限"
        return True, ""

    # ==================== 盘中监控 ====================

    def _monitor_loop(self):
        """盘中实时监控循环"""
        while self._running:
            try:
                self._update_risk_state()
                self._check_circuit_breaker()
                time.sleep(1)  # 每秒检查一次
            except Exception as e:
                logger.error(f"风控监控异常: {e}")
                time.sleep(3)

    def _update_risk_state(self):
        """更新风控状态"""
        if not self.om:
            return

        balance = self.om.get_balance()
        if not balance:
            return

        total_asset = balance.get('total_asset', self.total_capital)

        with self._lock:
            # 日盈亏
            self.state.daily_pnl = total_asset - self.total_capital
            self.state.daily_pnl_pct = self.state.daily_pnl / self.total_capital

            # 最大回撤
            if total_asset > self.state.peak_value:
                self.state.peak_value = total_asset
            drawdown = self.state.peak_value - total_asset
            self.state.max_drawdown = max(self.state.max_drawdown, drawdown)
            self.state.max_drawdown_pct = (
                self.state.max_drawdown / self.state.peak_value
            )

            # 仓位集中度
            positions = self.om.get_position_dict()
            if positions:
                values = [
                    p.volume * p.open_price for p in positions.values()
                ]
                total_pos = sum(values)
                max_pos = max(values) if values else 0
                self.state.position_concentration = (
                    max_pos / total_pos if total_pos > 0 else 0
                )
                self.state.total_position_pct = total_pos / self.total_capital

            # 交易频率
            now = time.time()
            self.state.trades_per_minute = len([
                t for t in self._trade_timestamps if now - t < 60
            ])

            self.state.last_update = now

    def _check_circuit_breaker(self):
        """熔断检查"""
        with self._lock:
            # 日亏损熔断
            if self.state.daily_pnl_pct < -self.max_daily_loss_pct:
                self._trigger_halt(
                    f"日亏损触发熔断: {self.state.daily_pnl_pct:.2%}"
                )
                return

            # 总回撤熔断
            if self.state.max_drawdown_pct > self.max_total_drawdown_pct:
                self._trigger_halt(
                    f"总回撤触发熔断: {self.state.max_drawdown_pct:.2%}"
                )
                return

            # 更新风险等级
            if self.state.daily_pnl_pct < -self.max_daily_loss_pct * 0.5:
                self.state.level = RiskLevel.CRITICAL
            elif self.state.daily_pnl_pct < -self.max_daily_loss_pct * 0.3:
                self.state.level = RiskLevel.WARNING
            else:
                self.state.level = RiskLevel.NORMAL

    def _trigger_halt(self, reason: str):
        """触发熔断"""
        self.state.halted = True
        self.state.level = RiskLevel.HALT
        logger.critical(f"🚨 系统熔断: {reason}")

        # 撤销所有挂单
        if self.om:
            self.om.cancel_all()

        # 发送告警
        if self.alert:
            self.alert.send_critical(f"🚨 量化系统熔断\n原因: {reason}")

    # ==================== 状态查询 ====================

    def get_risk_report(self) -> dict:
        """获取风控报告"""
        with self._lock:
            return {
                'level': self.state.level.value,
                'halted': self.state.halted,
                'daily_pnl': self.state.daily_pnl,
                'daily_pnl_pct': f"{self.state.daily_pnl_pct:.2%}",
                'max_drawdown': self.state.max_drawdown,
                'max_drawdown_pct': f"{self.state.max_drawdown_pct:.2%}",
                'total_position_pct': f"{self.state.total_position_pct:.2%}",
                'trades_per_minute': self.state.trades_per_minute,
                'violations_today': len(self.state.violations),
                'last_violations': self.state.violations[-5:],
            }
