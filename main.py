"""
量化交易系统 - 主入口
全链路: 数据采集 → 因子计算 → 模型推理 → 组合优化 → 算法执行 → 风控监控

启动方式:
    python main.py                    # 默认配置启动
    python main.py --config custom.yaml  # 自定义配置
    python main.py --mode paper       # 模拟盘模式
"""
import os
import sys
import time
import yaml
import signal
import argparse
import threading
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime, date, time as dtime
from collections import defaultdict, deque
from typing import Dict, List, Optional
import logging

# ============================================================
#  日志配置
# ============================================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / f"quant_{datetime.now():%Y%m%d}.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("main")

# ============================================================
#  模块导入
# ============================================================
from data.collector import L2DataCollector, TickData, OrderBookSnapshot
from data.storage import ClickHouseStorage
from data.cache import RedisCache
from factor.microstructure import MicroStructureFactors
from factor.manager import FactorManager
from model.transformer_model import ModelManager
from model.lgb_model import LGBModel
from model.ensemble import EnsembleModel
from execution.qmt_api import QMTOrderManager, OrderRequest, OrderSide
from execution.algo_exec import (
    AlgoExecutionEngine, AlgoOrder, AlgoType,
)
from execution.portfolio import PortfolioOptimizer, TargetPosition
from risk.realtime import RealTimeRiskEngine
from risk.post_trade import PostTradeAnalyzer
from monitor.alert import AlertManager
from monitor.dashboard import MetricsExporter


# ============================================================
#  常量
# ============================================================
MORNING_OPEN   = dtime(9, 30)
MORNING_CLOSE  = dtime(11, 30)
AFTERNOON_OPEN = dtime(13, 0)
AFTERNOON_CLOSE = dtime(15, 0)
PRE_OPEN_START  = dtime(9, 15)
STRATEGY_INTERVAL = 30        # 策略主循环间隔(秒)
FACTOR_HISTORY_LEN = 120      # 因子历史序列长度
TICK_BUFFER_MAX = 8000        # 每只股票最大缓存tick数
TICK_BUFFER_TRIM = 5000       # 裁剪后保留数


# ============================================================
#  主系统类
# ============================================================
class QuantTradingSystem:
    """
    全链路量化交易系统主控

    生命周期:
        __init__  → start → [运行中] → stop
                              ↑
                     _strategy_loop (核心循环)
                     _scheduler_loop (定时任务)
                     _heartbeat_loop (心跳监控)
    """

    # ----------------------------------------------------------
    #  初始化
    # ----------------------------------------------------------
    def __init__(self, config_path: str = "config/settings.yaml"):
        self._print_banner()

        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.mode = self.config["system"].get("mode", "production")
        logger.info(f"运行模式: {self.mode}")

        # ---------- 初始化各子系统 ----------
        # 告警 (最先初始化, 其他模块可能依赖)
        self.alert = AlertManager(self.config.get("alert", {}))

        # 数据层
        self.storage = ClickHouseStorage(self.config["data"]["clickhouse"])
        self.redis = RedisCache(self.config["data"]["redis"])
        self.collector = L2DataCollector(self.config["data"])

        # 因子层
        self.factor_engine = MicroStructureFactors(use_gpu=True)
        self.factor_manager = FactorManager(self.config.get("factor", {}))

        # 模型层
        self.model_manager = ModelManager(self.config["model"])
        self.lgb_model = LGBModel(self.config["model"].get("lgb", {}))
        self.ensemble = EnsembleModel(self.config["model"].get("ensemble", {}))

        # 交易层
        self.order_manager = QMTOrderManager(self.config["qmt"])
        self.algo_engine = AlgoExecutionEngine(
            self.order_manager, self.config
        )
        self.portfolio_optimizer = PortfolioOptimizer(self.config["risk"])

        # 风控层
        self.risk_engine = RealTimeRiskEngine(
            self.config["risk"],
            order_manager=self.order_manager,
            alert_manager=self.alert,
        )

        # 监控层
        self.post_analyzer = PostTradeAnalyzer(self.config)
        self.metrics = MetricsExporter(self.config.get("monitor", {}))

        # ---------- 运行时状态 ----------
        self._running = False
        self._threads: List[threading.Thread] = []

        # 数据缓存
        self._tick_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=TICK_BUFFER_MAX)
        )
        self._orderbook_cache: Dict[str, OrderBookSnapshot] = {}
        self._price_cache: Dict[str, float] = {}
        self._factor_cache: Dict[str, Dict[str, float]] = {}
        self._factor_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=FACTOR_HISTORY_LEN)
        )

        # 交易日状态
        self._today = date.today()
        self._cycle_count = 0
        self._last_rebalance_time = 0.0
        self._daily_trade_count = 0

        # 优雅退出
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("✅ 所有模块初始化完成")

    # ----------------------------------------------------------
    #  启动 / 停止
    # ----------------------------------------------------------
    def start(self):
        """启动全链路系统"""
        self._running = True
        logger.info("=" * 60)
        logger.info("  系统启动中...")
        logger.info("=" * 60)

        # Step 1: 连接QMT
        if self.mode != "backtest":
            if not self.order_manager.connect():
                logger.critical("❌ QMT连接失败, 系统退出")
                self.alert.send_critical("QMT连接失败, 系统无法启动")
                sys.exit(1)
            logger.info("✅ QMT 连接成功")

        # Step 2: 加载模型权重
        self._load_models()

        # Step 3: 加载股票池 & 订阅行情
        symbols = self._load_stock_pool()
        self.collector.subscribe(symbols)
        logger.info(f"✅ 股票池加载完成: {len(symbols)} 只")

        # Step 4: 注册数据回调
        self.collector.register_callback("tick", self._on_tick)
        self.collector.register_callback("orderbook", self._on_orderbook)
        self.collector.register_callback("order_queue", self._on_order_queue)

        # Step 5: 启动子系统
        self.collector.start()
        self.algo_engine.start()
        self.risk_engine.start()
        self.metrics.start()

        # Step 6: 启动后台线程
        thread_targets = [
            ("strategy_loop", self._strategy_loop),
            ("scheduler_loop", self._scheduler_loop),
            ("heartbeat_loop", self._heartbeat_loop),
            ("factor_flush_loop", self._factor_flush_loop),
        ]
        for name, target in thread_targets:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
            logger.info(f"  线程启动: {name}")

        self.alert.send_info(
            f"🚀 量化交易系统已启动\n"
            f"模式: {self.mode}\n"
            f"股票池: {len(symbols)} 只\n"
            f"资金: {self.config['risk']['total_capital']:,.0f}"
        )
        logger.info("🚀 系统启动完成, 进入主循环")

        # 主线程阻塞等待
        self._main_loop()

    def stop(self):
        """优雅停止系统"""
        if not self._running:
            return
        self._running = False
        logger.info("系统停止中...")

        # 1. 停止算法引擎 (会撤销所有子单)
        try:
            self.algo_engine.stop()
        except Exception as e:
            logger.error(f"算法引擎停止异常: {e}")

        # 2. 停止数据采集
        try:
            self.collector.stop()
        except Exception as e:
            logger.error(f"数据采集停止异常: {e}")

        # 3. 停止风控
        try:
            self.risk_engine.stop()
        except Exception as e:
            logger.error(f"风控引擎停止异常: {e}")

        # 4. 盘后分析
        try:
            self._run_post_trade()
        except Exception as e:
            logger.error(f"盘后分析异常: {e}")

        # 5. 断开QMT
        try:
            self.order_manager.disconnect()
        except Exception as e:
            logger.error(f"QMT断开异常: {e}")

        # 6. 停止监控
        try:
            self.metrics.stop()
        except Exception as e:
            logger.error(f"监控停止异常: {e}")

        # 7. 等待线程退出
        for t in self._threads:
            t.join(timeout=5)

        self.alert.send_info("🛑 量化交易系统已安全停止")
        logger.info("🛑 系统已安全停止")

    def _main_loop(self):
        """主线程阻塞循环"""
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _signal_handler(self, signum, frame):
        """信号处理 (Ctrl+C / kill)"""
        logger.info(f"收到信号 {signum}, 准备退出...")
        self._running = False

    # ----------------------------------------------------------
    #  数据回调 (由 L2DataCollector 驱动)
    # ----------------------------------------------------------
    def _on_tick(self, tick: TickData):
        """逐笔成交回调 - 高频, 需极低延迟"""
        sym = tick.symbol
        self._tick_buffer[sym].append(tick)
        self._price_cache[sym] = tick.price

        # 写入Redis供其他进程读取
        # self.redis.set_tick(sym, tick)

    def _on_orderbook(self, snapshot: OrderBookSnapshot):
        """盘口快照回调 - 触发因子计算"""
        sym = snapshot.symbol
        self._orderbook_cache[sym] = snapshot
        self._price_cache[sym] = snapshot.last_price

        # 实时因子计算
        ticks = list(self._tick_buffer.get(sym, []))
        if len(ticks) >= 50:
            try:
                factors = self.factor_engine.compute_all(
                    sym, snapshot, ticks, snapshot.timestamp
                )
                self._factor_cache[sym] = factors

                # 追加到历史序列 (用于Transformer输入)
                factor_values = np.array(
                    list(factors.values()), dtype=np.float32
                )
                self._factor_history[sym].append(factor_values)

                # 上报因子指标
                self.metrics.report_factors(sym, factors)

            except Exception as e:
                logger.error(f"因子计算异常 {sym}: {e}")

    def _on_order_queue(self, queue_data):
        """委托队列回调"""
        # 可用于更精细的因子计算
        pass

    # ----------------------------------------------------------
    #  策略主循环
    # ----------------------------------------------------------
    def _strategy_loop(self):
        """
        策略主循环
        每 STRATEGY_INTERVAL 秒执行一次完整的:
        因子汇总 → 模型推理 → 组合优化 → 风控检查 → 算法下单
        """
        logger.info("策略循环已启动")

        while self._running:
            try:
                # 非交易时段休眠
                if not self._is_trading_time():
                    time.sleep(3)
                    continue

                # 检查风控状态
                if self.risk_engine.state.halted:
                    logger.warning("系统已熔断, 策略暂停")
                    time.sleep(10)
                    continue

                # 执行策略周期
                self._run_strategy_cycle()

                # 等待下一周期
                time.sleep(STRATEGY_INTERVAL)

            except Exception as e:
                logger.error(f"策略循环异常: {e}\n{traceback.format_exc()}")
                self.alert.send_warning(f"策略循环异常: {e}")
                time.sleep(10)

    def _run_strategy_cycle(self):
        """
        单次策略执行周期
        
        Pipeline:
        ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ 因子汇总 │ → │ 模型推理  │ → │ 组合优化  │ → │ 风控检查  │ → │ 算法下单  │
        └─────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
        """
        self._cycle_count += 1
        t_start = time.time()
        cycle_id = f"C{self._cycle_count:06d}"

        logger.info(f"{'='*50}")
        logger.info(f"[{cycle_id}] 策略周期开始")

        # ============ Step 1: 准备因子矩阵 ============
        t1 = time.time()

        symbols_with_factors = [
            sym for sym, hist in self._factor_history.items()
            if len(hist) >= 10  # 至少10个时间步
        ]

        if len(symbols_with_factors) < 5:
            logger.info(f"[{cycle_id}] 因子数据不足 ({len(symbols_with_factors)}只), 跳过")
            return

        seq_len = self.config["model"]["transformer"]["seq_len"]
        n_factors = len(self._factor_cache[symbols_with_factors[0]])
        n_stocks = len(symbols_with_factors)

        # 构建 (n_stocks, seq_len, n_factors) 张量
        factor_matrix = np.zeros(
            (n_stocks, seq_len, n_factors), dtype=np.float32
        )
        for i, sym in enumerate(symbols_with_factors):
            hist = list(self._factor_history[sym])
            arr = np.array(hist[-seq_len:])  # (T, F)
            actual_len = arr.shape[0]

            # 不足seq_len的部分用最早值填充
            if actual_len < seq_len:
                pad = np.tile(arr[0], (seq_len - actual_len, 1))
                arr = np.vstack([pad, arr])

            factor_matrix[i] = arr[:seq_len, :n_factors]

        # NaN处理
        factor_matrix = np.nan_to_num(factor_matrix, nan=0.0)

        dt_factor = time.time() - t1
        logger.info(
            f"[{cycle_id}] Step1 因子准备: {n_stocks}只 x {seq_len}步 x {n_factors}因子, "
            f"耗时 {dt_factor*1000:.1f}ms"
        )

        # ============ Step 2: 模型推理 ============
        t2 = time.time()

        # Transformer 预测
        tf_returns, tf_vols = self.model_manager.predict(factor_matrix)

        # LightGBM 预测 (使用最新截面因子)
        lgb_features = factor_matrix[:, -1, :]  # 最新一步
        lgb_returns = self.lgb_model.predict(lgb_features)

        # 模型集成
        pred_returns, pred_vols = self.ensemble.combine(
            predictions={
                "transformer": (tf_returns, tf_vols),
                "lightgbm": (lgb_returns, None),
            },
            symbols=symbols_with_factors,
        )

        dt_model = time.time() - t2
        logger.info(
            f"[{cycle_id}] Step2 模型推理: "
            f"TF={dt_model*1000:.1f}ms, "
            f"Top3预测: {sorted(pred_returns.items(), key=lambda x: x[1], reverse=True)[:3]}"
        )

        # ============ Step 3: 组合优化 ============
        t3 = time.time()

        # 获取当前持仓
        current_positions = {}
        try:
            pos_dict = self.order_manager.get_position_dict()
            for sym, pos in pos_dict.items():
                current_positions[sym] = pos.volume
        except Exception as e:
            logger.error(f"查询持仓异常: {e}")
            return

        # 优化
        targets = self.portfolio_optimizer.optimize(
            pred_returns=pred_returns,
            pred_vols=pred_vols,
            current_prices=dict(self._price_cache),
            current_positions=current_positions,
        )

        dt_portfolio = time.time() - t3

        # 统计
        n_buy = sum(1 for t in targets if t.side == "buy")
        n_sell = sum(1 for t in targets if t.side == "sell")
        n_hold = sum(1 for t in targets if t.side == "hold")

        logger.info(
            f"[{cycle_id}] Step3 组合优化: "
            f"买入={n_buy}, 卖出={n_sell}, 持有={n_hold}, "
            f"耗时 {dt_portfolio*1000:.1f}ms"
        )

        # ============ Step 4: 风控检查 + 下单 ============
        t4 = time.time()
        orders_submitted = 0
        orders_blocked = 0

        for target in targets:
            if target.side == "hold" or target.delta_volume == 0:
                continue

            symbol = target.symbol
            side = target.side
            volume = abs(target.delta_volume)
            price = self._price_cache.get(symbol, 0)

            if price <= 0:
                logger.warning(f"[{cycle_id}] {symbol} 价格异常({price}), 跳过")
                continue

            # ---- 风控检查 ----
            passed, reason = self.risk_engine.pre_trade_check(
                symbol=symbol,
                side=side,
                volume=volume,
                price=price,
            )

            if not passed:
                logger.warning(
                    f"[{cycle_id}] 🚫 风控拦截 {symbol} "
                    f"{side} {volume}股: {reason}"
                )
                orders_blocked += 1
                continue

            # ---- 选择算法类型 ----
            algo_type, duration = self._select_algo(
                symbol, side, volume, price
            )

            # ---- 提交算法单 ----
            algo_id = f"{symbol}_{cycle_id}_{int(time.time())}"
            algo_order = AlgoOrder(
                algo_id=algo_id,
                symbol=symbol,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                total_volume=volume,
                algo_type=algo_type,
                duration_seconds=duration,
                price_limit=price,
                urgency=min(abs(pred_returns.get(symbol, 0)) * 10, 1.0),
            )
            self.algo_engine.submit(algo_order)
            orders_submitted += 1

            logger.info(
                f"[{cycle_id}] 📤 {side.upper()} {symbol} "
                f"{volume}股 @{price:.2f} "
                f"algo={algo_type.value} duration={duration}s"
            )

        dt_execution = time.time() - t4
        dt_total = time.time() - t_start

        logger.info(
            f"[{cycle_id}] Step4 执行: "
            f"提交={orders_submitted}, 拦截={orders_blocked}, "
            f"耗时 {dt_execution*1000:.1f}ms"
        )
        logger.info(
            f"[{cycle_id}] ✅ 策略周期完成, "
            f"总耗时 {dt_total*1000:.1f}ms "
            f"(因子={dt_factor*1000:.0f} + 模型={dt_model*1000:.0f} + "
            f"优化={dt_portfolio*1000:.0f} + 执行={dt_execution*1000:.0f})"
        )

        # 上报指标
        self.metrics.report_cycle(
            cycle_id=cycle_id,
            n_stocks=n_stocks,
            n_buy=n_buy,
            n_sell=n_sell,
            latency_ms=dt_total * 1000,
        )

        self._daily_trade_count += orders_submitted

    # ----------------------------------------------------------
    #  算法选择
    # ----------------------------------------------------------
    def _select_algo(self, symbol: str, side: str,
                     volume: int, price: float) -> tuple:
        """
        根据订单特征选择最优算法

        Returns:
            (AlgoType, duration_seconds)
        """
        order_value = volume * price

        # 小单 (< 2万): 直接下单, 短时间TWAP
        if order_value < 20000:
            return AlgoType.TWAP, 30

        # 中单 (2万~10万): TWAP 1-2分钟
        if order_value < 100000:
            return AlgoType.TWAP, 90

        # 大单 (> 10万): VWAP 跟随市场节奏
        # 或冰山单隐藏意图
        if side == "buy":
            return AlgoType.VWAP, 180
        else:
            # 卖出用冰山单, 减少冲击
            return AlgoType.ICEBERG, 240

    # ----------------------------------------------------------
    #  定时任务
    # ----------------------------------------------------------
    def _scheduler_loop(self):
        """定时任务调度"""
        logger.info("定时任务调度已启动")

        while self._running:
            try:
                now = datetime.now()
                current_time = now.time()

                # ---- 09:15 盘前准备 ----
                if current_time.hour == 9 and current_time.minute == 15:
                    if self._today != date.today():
                        self._today = date.today()
                        self._on_new_trading_day()

                # ---- 09:25 集合竞价结束, 更新股票池 ----
                if current_time.hour == 9 and current_time.minute == 25:
                    self._refresh_stock_pool()

                # ---- 11:30 午间暂停 ----
                if current_time.hour == 11 and current_time.minute == 30:
                    self._on_morning_close()

                # ---- 14:50 尾盘处理 ----
                if current_time.hour == 14 and current_time.minute == 50:
                    self._on_near_close()

                # ---- 15:05 盘后处理 ----
                if current_time.hour == 15 and current_time.minute == 5:
                    self._run_post_trade()

                # ---- 15:30 模型增量训练 ----
                if current_time.hour == 15 and current_time.minute == 30:
                    self._run_incremental_training()

                # ---- 每小时: 保存状态 ----
                if current_time.minute == 0:
                    self._save_state()

                time.sleep(30)  # 30秒检查一次

            except Exception as e:
                logger.error(f"定时任务异常: {e}\n{traceback.format_exc()}")
                time.sleep(60)

    def _on_new_trading_day(self):
        """新交易日初始化"""
        logger.info("📅 新交易日初始化")

        # 重置日内状态
        self._cycle_count = 0
        self._daily_trade_count = 0
        self._last_rebalance_time = 0.0

        # 清空因子缓存
        self._factor_cache.clear()
        self._factor_history.clear()
        self._tick_buffer.clear()

        # 重置风控日内计数
        self.risk_engine.state.daily_pnl = 0.0
        self.risk_engine.state.daily_pnl_pct = 0.0
        self.risk_engine.state.halted = False
        self.risk_engine.state.violations.clear()
        self.risk_engine.state.total_trades_today = 0

        # 查询账户状态
        balance = self.order_manager.get_balance()
        positions = self.order_manager.get_position_dict()
        logger.info(
            f"账户状态: 总资产={balance.get('total_asset', 0):,.0f}, "
            f"现金={balance.get('cash', 0):,.0f}, "
            f"持仓={len(positions)}只"
        )

        self.alert.send_info(
            f"📅 新交易日 {self._today}\n"
            f"总资产: {balance.get('total_asset', 0):,.0f}\n"
            f"持仓: {len(positions)} 只"
        )

    def _refresh_stock_pool(self):
        """刷新股票池"""
        try:
            new_symbols = self._load_stock_pool()
            self.collector.subscribe(new_symbols)
            logger.info(f"股票池已刷新: {len(new_symbols)} 只")
        except Exception as e:
            logger.error(f"股票池刷新失败: {e}")

    def _on_morning_close(self):
        """午间收盘处理"""
        logger.info("🕐 午间休市")
        # 可在此做中间统计
        risk_report = self.risk_engine.get_risk_report()
        logger.info(f"午间风控报告: {risk_report}")

    def _on_near_close(self):
        """尾盘处理 (14:50)"""
        logger.info("⏰ 尾盘处理开始")

        # 检查是否需要尾盘减仓
        risk_report = self.risk_engine.get_risk_report()

        # 如果日亏损超过1.5%, 尾盘减仓至50%
        if self.risk_engine.state.daily_pnl_pct < -0.015:
            logger.warning("日亏损超1.5%, 尾盘减仓")
            self.alert.send_warning(
                f"⚠️ 尾盘减仓触发\n"
                f"日亏损: {self.risk_engine.state.daily_pnl_pct:.2%}"
            )
            self._reduce_positions(target_pct=0.5)

    def _reduce_positions(self, target_pct: float):
        """减仓到目标比例"""
        positions = self.order_manager.get_position_dict()
        for sym, pos in positions.items():
            if pos.volume <= 0:
                continue
            sell_volume = int(pos.volume * (1 - target_pct))
            sell_volume = (sell_volume // 100) * 100
            if sell_volume >= 100:
                price = self._price_cache.get(sym, 0)
                passed, reason = self.risk_engine.pre_trade_check(
                    sym, "sell", sell_volume, price
                )
                if passed:
                    algo = AlgoOrder(
                        algo_id=f"reduce_{sym}_{int(time.time())}",
                        symbol=sym,
                        side=OrderSide.SELL,
                        total_volume=sell_volume,
                        algo_type=AlgoType.TWAP,
                        duration_seconds=60,
                        price_limit=price,
                        urgency=0.9,
                    )
                    self.algo_engine.submit(algo)

    def _run_post_trade(self):
        """盘后分析"""
        logger.info("📊 盘后分析开始")
        try:
            # 获取今日交易记录
            balance = self.order_manager.get_balance()
            positions = self.order_manager.get_position_dict()
            risk_report = self.risk_engine.get_risk_report()

            report = {
                "date": str(self._today),
                "total_asset": balance.get("total_asset", 0),
                "cash": balance.get("cash", 0),
                "daily_pnl": self.risk_engine.state.daily_pnl,
                "daily_pnl_pct": f"{self.risk_engine.state.daily_pnl_pct:.2%}",
                "max_drawdown_pct": risk_report.get("max_drawdown_pct", "0%"),
                "trade_count": self._daily_trade_count,
                "position_count": len(positions),
                "total_position_pct": risk_report.get("total_position_pct", "0%"),
                "strategy_cycles": self._cycle_count,
                "risk_violations": risk_report.get("violations_today", 0),
            }

            # 发送日报
            self.alert.send_trade_report(report)

            # 保存到数据库
            self.post_analyzer.save_daily_report(report)

            # 因子IC分析
            self.factor_manager.analyze_daily_ic()

            logger.info(f"📊 盘后分析完成: {report}")

        except Exception as e:
            logger.error(f"盘后分析异常: {e}\n{traceback.format_exc()}")

    def _run_incremental_training(self):
        """增量训练 (盘后)"""
        retrain_days = self.config["model"]["ensemble"].get("retrain_days", 5)

        # 检查是否到了重训周期
        # 简化: 每 retrain_days 天训练一次
        day_of_year = self._today.timetuple().tm_yday
        if day_of_year % retrain_days != 0:
            return

        logger.info("🧠 开始增量训练...")
        try:
            # 从ClickHouse加载最近N天的因子和收益数据
            # train_data = self.storage.query_training_data(days=30)
            # metrics = self.model_manager.train_epoch(train_data)
            # logger.info(f"训练完成: {metrics}")

            # 保存模型
            model_path = f"models/model_{self._today}.pt"
            self.model_manager.save(model_path)
            self.model_manager.save("models/latest_model.pt")
            logger.info(f"模型已保存: {model_path}")

            self.alert.send_info(f"🧠 模型增量训练完成")

        except Exception as e:
            logger.error(f"增量训练异常: {e}\n{traceback.format_exc()}")

    # ----------------------------------------------------------
    #  心跳监控
    # ----------------------------------------------------------
    def _heartbeat_loop(self):
        """心跳监控 - 检测各模块健康状态"""
        logger.info("心跳监控已启动")

        while self._running:
            try:
                status = {
                    "qmt_connected": self.order_manager._connected,
                    "collector_running": self.collector._running,
                    "risk_halted": self.risk_engine.state.halted,
                    "risk_level": self.risk_engine.state.level.value,
                    "symbols_with_data": len(self._price_cache),
                    "symbols_with_factors": len(self._factor_cache),
                    "active_algos": len(self.algo_engine._active_algos),
                    "tick_buffer_total": sum(
                        len(v) for v in self._tick_buffer.values()
                    ),
                    "cycle_count": self._cycle_count,
                    "daily_trades": self._daily_trade_count,
                }

                # 检查异常
                if not status["qmt_connected"]:
                    self.alert.send_critical("QMT 连接断开!")

                if status["symbols_with_data"] == 0 and self._is_trading_time():
                    self.alert.send_warning("无行情数据!")

                # 上报心跳指标
                self.metrics.report_heartbeat(status)

                # 每5分钟输出一次状态
                if self._cycle_count % 10 == 0:
                    logger.info(f"💓 心跳: {status}")

                time.sleep(30)

            except Exception as e:
                logger.error(f"心跳监控异常: {e}")
                time.sleep(60)

    # ----------------------------------------------------------
    #  因子落盘
    # ----------------------------------------------------------
    def _factor_flush_loop(self):
        """定期将因子值落盘到ClickHouse"""
        logger.info("因子落盘线程已启动")

        while self._running:
            try:
                if self._factor_cache and self._is_trading_time():
                    batch = []
                    now = datetime.now()
                    for sym, factors in self._factor_cache.items():
                        for fname, fval in factors.items():
                            batch.append((sym, now, fname, float(fval)))

                    if batch:
                        self.storage.client.execute(
                            "INSERT INTO factor_values VALUES", batch
                        )
                        logger.debug(f"因子落盘: {len(batch)} 条")

                time.sleep(10)  # 每10秒落盘一次

            except Exception as e:
                logger.error(f"因子落盘异常: {e}")
                time.sleep(30)

    # ----------------------------------------------------------
    #  辅助方法
    # ----------------------------------------------------------
    def _is_trading_time(self) -> bool:
        """判断当前是否为交易时段"""
        now = datetime.now().time()
        morning = MORNING_OPEN <= now <= MORNING_CLOSE
        afternoon = AFTERNOON_OPEN <= now <= AFTERNOON_CLOSE
        return morning or afternoon

    def _is_pre_market(self) -> bool:
        """判断是否为盘前"""
        now = datetime.now().time()
        return PRE_OPEN_START <= now < MORNING_OPEN

    def _load_stock_pool(self) -> List[str]:
        """
        加载股票池

        筛选逻辑:
        1. 剔除ST / *ST
        2. 剔除上市不足60天
        3. 剔除停牌
        4. 剔除涨跌停
        5. 流动性筛选 (日均成交额 > 2000万)
        6. 市值筛选 (20亿 ~ 500亿)
        """
        try:
            # 实际实现: 从数据源获取全A股列表并筛选
            # symbols = xtdata.get_stock_list_in_sector('沪深A股')
            # filtered = self._filter_symbols(symbols)

            # 示例: 从文件加载
            pool_file = Path("config/stock_pool.txt")
            if pool_file.exists():
                symbols = [
                    line.strip()
                    for line in pool_file.read_text().splitlines()
                    if line.strip() and not line.startswith("#")
                ]
                return symbols

            # 默认返回沪深300成分股
            logger.warning("未找到股票池文件, 使用默认池")
            return []

        except Exception as e:
            logger.error(f"加载股票池异常: {e}")
            return []

    def _load_models(self):
        """加载模型权重"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        # Transformer模型
        tf_path = model_dir / "latest_model.pt"
        if tf_path.exists():
            self.model_manager.load(str(tf_path))
            logger.info(f"✅ Transformer模型已加载: {tf_path}")
        else:
            logger.warning("⚠️ 未找到Transformer模型, 使用随机初始化")

        # LightGBM模型
        lgb_path = model_dir / "latest_lgb.txt"
        if lgb_path.exists():
            self.lgb_model.load(str(lgb_path))
            logger.info(f"✅ LightGBM模型已加载: {lgb_path}")
        else:
            logger.warning("⚠️ 未找到LightGBM模型")

    def _save_state(self):
        """保存系统状态 (用于恢复)"""
        try:
            state = {
                "date": str(self._today),
                "cycle_count": self._cycle_count,
                "daily_trade_count": self._daily_trade_count,
                "risk_state": self.risk_engine.get_risk_report(),
                "timestamp": datetime.now().isoformat(),
            }
            state_path = Path("data/system_state.yaml")
            state_path.parent.mkdir(exist_ok=True)
            with open(state_path, "w") as f:
                yaml.dump(state, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"状态保存异常: {e}")

    @staticmethod
    def _print_banner():
        banner = r"""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     ██████  ██    ██  █████  ███    ██ ████████          ║
    ║    ██    ██ ██    ██ ██   ██ ████   ██    ██             ║
    ║    ██    ██ ██    ██ ███████ ██ ██  ██    ██             ║
    ║    ██ ▄▄ ██ ██    ██ ██   ██ ██  ██ ██    ██             ║
    ║     ██████   ██████  ██   ██ ██   ████    ██             ║
    ║        ▀▀                                                ║
    ║          Quantitative Trading System v1.0                ║
    ║          L2 Data | GPU Model | QMT Execution             ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
        """
        print(banner)


# ============================================================
#  入口
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="量化交易系统")
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["production", "paper", "backtest"],
        default=None,
        help="运行模式 (覆盖配置文件)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 创建必要目录
    for d in ["logs", "models", "data", "config"]:
        Path(d).mkdir(exist_ok=True)

    # 初始化系统
    system = QuantTradingSystem(config_path=args.config)

    # 覆盖运行模式
    if args.mode:
        system.config["system"]["mode"] = args.mode
        system.mode = args.mode

    # 启动 (阻塞)
    system.start()


if __name__ == "__main__":
    main()
