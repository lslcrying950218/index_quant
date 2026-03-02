"""
Prometheus 指标导出
- 系统运行指标
- 交易绩效指标
- 因子质量指标
- 风控状态指标
- 供 Grafana 可视化
"""
import time
import threading
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        start_http_server,
        Gauge, Counter, Histogram, Summary, Info,
    )
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False
    logger.warning("prometheus_client 未安装, 指标导出功能不可用")


class MetricsExporter:
    """Prometheus 指标导出器"""

    def __init__(self, config: dict):
        self.config = config
        self.port = config.get("prometheus_port", 9090)
        self._running = False

        if not PROM_AVAILABLE:
            return

        # ============ 系统指标 ============
        self.system_info = Info(
            "quant_system", "量化交易系统信息"
        )
        self.uptime = Gauge(
            "quant_uptime_seconds", "系统运行时间"
        )
        self.heartbeat_ts = Gauge(
            "quant_heartbeat_timestamp", "最近心跳时间戳"
        )

        # ============ 数据指标 ============
        self.symbols_with_data = Gauge(
            "quant_symbols_with_data", "有行情数据的股票数"
        )
        self.symbols_with_factors = Gauge(
            "quant_symbols_with_factors", "有因子数据的股票数"
        )
        self.tick_buffer_size = Gauge(
            "quant_tick_buffer_total", "Tick缓冲区总条数"
        )
        self.data_latency = Histogram(
            "quant_data_latency_ms", "数据延迟(ms)",
            buckets=[0.1, 0.5, 1, 2, 5, 10, 50, 100]
        )

        # ============ 策略指标 ============
        self.strategy_cycle_count = Counter(
            "quant_strategy_cycles_total", "策略执行周期总数"
        )
        self.strategy_cycle_latency = Histogram(
            "quant_strategy_cycle_latency_ms", "策略周期延迟(ms)",
            buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000]
        )
        self.model_inference_latency = Histogram(
            "quant_model_inference_latency_ms", "模型推理延迟(ms)",
            buckets=[1, 2, 5, 10, 20, 50]
        )
        self.signals_generated = Counter(
            "quant_signals_total", "生成信号总数",
            ["action"]  # buy / sell / hold
        )

        # ============ 交易指标 ============
        self.orders_submitted = Counter(
            "quant_orders_submitted_total", "提交订单总数",
            ["side"]  # buy / sell
        )
        self.orders_filled = Counter(
            "quant_orders_filled_total", "成交订单总数",
            ["side"]
        )
        self.orders_rejected = Counter(
            "quant_orders_rejected_total", "拒绝订单总数"
        )
        self.order_fill_latency = Histogram(
            "quant_order_fill_latency_ms", "订单成交延迟(ms)",
            buckets=[10, 50, 100, 500, 1000, 5000]
        )
        self.slippage_bps = Histogram(
            "quant_slippage_bps", "滑点(bps)",
            buckets=[0.5, 1, 2, 5, 10, 20, 50]
        )
        self.commission_total = Gauge(
            "quant_commission_today", "今日佣金总额"
        )

        # ============ 持仓指标 ============
        self.total_asset = Gauge(
            "quant_total_asset", "总资产"
        )
        self.cash_balance = Gauge(
            "quant_cash_balance", "现金余额"
        )
        self.market_value = Gauge(
            "quant_market_value", "持仓市值"
        )
        self.position_count = Gauge(
            "quant_position_count", "持仓数量"
        )
        self.position_pct = Gauge(
            "quant_position_pct", "仓位百分比"
        )

        # ============ 盈亏指标 ============
        self.daily_pnl = Gauge(
            "quant_daily_pnl", "日盈亏"
        )
        self.daily_pnl_pct = Gauge(
            "quant_daily_pnl_pct", "日盈亏百分比"
        )
        self.total_pnl = Gauge(
            "quant_total_pnl", "累计盈亏"
        )
        self.max_drawdown_pct = Gauge(
            "quant_max_drawdown_pct", "最大回撤百分比"
        )
        self.nav = Gauge(
            "quant_nav", "净值"
        )

        # ============ 风控指标 ============
        self.risk_level = Gauge(
            "quant_risk_level", "风控等级 (0=normal,1=warning,2=critical,3=halt)"
        )
        self.risk_violations = Counter(
            "quant_risk_violations_total", "风控违规总次数"
        )
        self.circuit_breaker_triggered = Counter(
            "quant_circuit_breaker_total", "熔断触发次数"
        )

        # ============ 因子指标 ============
        self.factor_ic = Gauge(
            "quant_factor_ic", "因子IC值",
            ["factor_name"]
        )

        # ============ NLP指标 ============
        self.news_count = Counter(
            "quant_news_processed_total", "处理新闻总数"
        )
        self.sentiment_avg = Gauge(
            "quant_sentiment_avg", "平均情感分数"
        )

        self._start_time = time.time()

    def start(self):
        """启动Prometheus HTTP服务"""
        if not PROM_AVAILABLE:
            logger.warning("Prometheus不可用, 跳过指标导出")
            return

        self._running = True
        try:
            start_http_server(self.port)
            logger.info(f"✅ Prometheus指标服务启动: http://0.0.0.0:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Prometheus启动失败: {e}")

        # 系统信息
        self.system_info.info({
            "version": "1.0.0",
            "mode": self.config.get("mode", "production"),
        })

        # 启动定时更新线程
        self._update_thread = threading.Thread(
            target=self._periodic_update, daemon=True, name="metrics_updater"
        )
        self._update_thread.start()

    def stop(self):
        self._running = False

    def _periodic_update(self):
        """定期更新基础指标"""
        while self._running:
            try:
                self.uptime.set(time.time() - self._start_time)
                self.heartbeat_ts.set(time.time())
            except Exception:
                pass
            time.sleep(5)

    # ==================== 上报方法 ====================

    def report_cycle(self, cycle_id: str, n_stocks: int,
                     n_buy: int, n_sell: int, latency_ms: float):
        """上报策略周期指标"""
        if not PROM_AVAILABLE:
            return
        self.strategy_cycle_count.inc()
        self.strategy_cycle_latency.observe(latency_ms)
        self.signals_generated.labels(action="buy").inc(n_buy)
        self.signals_generated.labels(action="sell").inc(n_sell)

    def report_order(self, side: str, status: str,
                     fill_latency_ms: float = 0, slippage: float = 0):
        """上报订单指标"""
        if not PROM_AVAILABLE:
            return
        self.orders_submitted.labels(side=side).inc()
        if status == "filled":
            self.orders_filled.labels(side=side).inc()
            if fill_latency_ms > 0:
                self.order_fill_latency.observe(fill_latency_ms)
            if slippage != 0:
                self.slippage_bps.observe(abs(slippage) * 10000)
        elif status == "rejected":
            self.orders_rejected.inc()

    def report_portfolio(self, total_asset: float, cash: float,
                          market_value: float, position_count: int,
                          daily_pnl: float, daily_pnl_pct: float,
                          max_dd_pct: float, nav: float):
        """上报持仓/盈亏指标"""
        if not PROM_AVAILABLE:
            return
        self.total_asset.set(total_asset)
        self.cash_balance.set(cash)
        self.market_value.set(market_value)
        self.position_count.set(position_count)
        self.daily_pnl.set(daily_pnl)
        self.daily_pnl_pct.set(daily_pnl_pct)
        self.max_drawdown_pct.set(max_dd_pct)
        self.nav.set(nav)

        total_capital = self.config.get("total_capital", 1000000)
        self.position_pct.set(market_value / total_capital * 100 if total_capital > 0 else 0)

    def report_risk(self, level: str, violations: int = 0,
                     halted: bool = False):
        """上报风控指标"""
        if not PROM_AVAILABLE:
            return
        level_map = {"normal": 0, "warning": 1, "critical": 2, "halt": 3}
        self.risk_level.set(level_map.get(level, 0))
        if violations > 0:
            self.risk_violations.inc(violations)
        if halted:
            self.circuit_breaker_triggered.inc()

    def report_factors(self, symbol: str, factors: Dict[str, float]):
        """上报因子值 (采样, 避免指标爆炸)"""
        if not PROM_AVAILABLE:
            return
        # 只上报关键因子
        key_factors = [
            "depth_imbalance_5", "volume_weighted_ofi",
            "realized_vol", "smart_money_flow", "vpin"
        ]
        for fname in key_factors:
            if fname in factors:
                self.factor_ic.labels(factor_name=fname).set(factors[fname])

    def report_heartbeat(self, status: dict):
        """上报心跳状态"""
        if not PROM_AVAILABLE:
            return
        self.symbols_with_data.set(status.get("symbols_with_data", 0))
        self.symbols_with_factors.set(status.get("symbols_with_factors", 0))
        self.tick_buffer_size.set(status.get("tick_buffer_total", 0))
