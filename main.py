"""
量化交易系统 - 主入口 (完整版)
整合: L2数据 → NLP → 流计算 → ONNX推理 → 规则引擎 → 信号融合 → OMS → 算法执行 → 风控 → 监控

启动:
    python main.py
    python main.py --config custom.yaml --mode paper
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
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"quant_{datetime.now():%Y%m%d}.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

# ============================================================
#  模块导入
# ============================================================
from data.collector import L2DataCollector, TickData, OrderBookSnapshot
from data.storage import ClickHouseStorage
from data.cache import RedisCache
from data.news_crawler import NewsCrawler
from data.nlp_pipeline import NLPPipeline

from compute.batch_processor import BatchProcessor
from compute.stream_engine import StreamComputeEngine
from compute.onnx_inference import ONNXInferenceEngine

from factor.microstructure import MicroStructureFactors

from model.transformer_model import ModelManager
from model.lgb_model import LGBModel
from model.ensemble import EnsembleModel

from decision.rule_engine import RuleEngine
from decision.signal_generator import SignalGenerator
from decision.portfolio import PortfolioOptimizer

from execution.qmt_api import QMTOrderManager, OrderSide
from execution.oms import OrderManagementSystem, OrderType
from execution.algo_exec import AlgoExecutionEngine, AlgoOrder, AlgoType

from risk.realtime import RealTimeRiskEngine

from monitor.metrics_exporter import MetricsExporter
from monitor.dashboard import GrafanaDashboardGenerator
from monitor.alert import AlertManager
from monitor.report_generator import ReportGenerator
from risk.post_trade import PostTradeAnalyzer
# ============================================================
#  常量
# ============================================================
MORNING_OPEN    = dtime(9, 30)
MORNING_CLOSE   = dtime(11, 30)
AFTERNOON_OPEN  = dtime(13, 0)
AFTERNOON_CLOSE = dtime(15, 0)
PRE_OPEN_START  = dtime(9, 15)

STRATEGY_INTERVAL   = 30      # 策略主循环间隔(秒)
TICK_BUFFER_MAX     = 8000
FACTOR_HISTORY_LEN  = 120


# ============================================================
#  主系统类
# ============================================================
class QuantTradingSystem:
    """
    全链路量化交易系统

    架构层次:
    ┌─────────────────────────────────────────────────────────┐
    │  数据层: L2行情 + 新闻爬虫 + NLP + Redis缓存            │
    ├─────────────────────────────────────────────────────────┤
    │  计算层: 盘前批处理 + 盘中流计算 + ONNX推理              │
    ├─────────────────────────────────────────────────────────┤
    │  决策层: 规则引擎 + 信号融合 + 组合优化                  │
    ├─────────────────────────────────────────────────────────┤
    │  交易层: QMT接口 + OMS + 算法执行                       │
    ├─────────────────────────────────────────────────────────┤
    │  风控层: 盘前/盘中/盘后 三级风控                         │
    ├─────────────────────────────────────────────────────────┤
    │  监控层: Prometheus + Grafana + 告警 + 日报              │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self._print_banner()

        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.mode = self.config["system"].get("mode", "production")
        logger.info(f"运行模式: {self.mode}")

        # ========================================
        #  初始化所有子系统 (按依赖顺序)
        # ========================================

        # --- 监控 & 告警 (最先, 其他模块依赖) ---
        self.alert = AlertManager(self.config.get("alert", {}))
        self.metrics = MetricsExporter(self.config.get("monitor", {}))

        # --- 数据层 ---
        self.storage = ClickHouseStorage(self.config["data"]["clickhouse"])
        self.redis = RedisCache(self.config["data"]["redis"])
        self.collector = L2DataCollector(self.config["data"])
        self.news_crawler = NewsCrawler(
            self.config.get("news", {}), redis_cache=self.redis
        )
        self.nlp = NLPPipeline(
            self.config.get("nlp", {}), redis_cache=self.redis
        )

        # --- 计算层 ---
        self.batch_processor = BatchProcessor(
            self.config.get("model", {}),
            storage=self.storage,
            redis_cache=self.redis,
        )
        self.stream_engine = StreamComputeEngine(
            self.config.get("stream", {}), redis_cache=self.redis
        )
        self.onnx_engine = ONNXInferenceEngine(self.config.get("model", {}))

        # --- 因子层 ---
        self.factor_engine = MicroStructureFactors(use_gpu=True)

        # --- 模型层 ---
        self.model_manager = ModelManager(self.config["model"])
        self.lgb_model = LGBModel(self.config["model"].get("lgb", {}))
        self.ensemble = EnsembleModel(self.config["model"].get("ensemble", {}))

        # --- 决策层 ---
        self.rule_engine = RuleEngine(self.config.get("rules", {"rules_dir": "config/rules"}))
        self.signal_generator = SignalGenerator(
            self.config.get("signal", {}),
            rule_engine=self.rule_engine,
            nlp_pipeline=self.nlp,
            redis_cache=self.redis,
        )
        self.portfolio_optimizer = PortfolioOptimizer(self.config["risk"])

        # --- 交易层 ---
        self.order_manager = QMTOrderManager(self.config["qmt"])
        self.oms = OrderManagementSystem(self.config, storage=self.storage)
        self.algo_engine = AlgoExecutionEngine(self.order_manager, self.config)

        # --- 风控层 ---
        self.risk_engine = RealTimeRiskEngine(
            self.config["risk"],
            order_manager=self.order_manager,
            alert_manager=self.alert,
        )

        # --- 监控层 (补充) ---
        self.post_analyzer = PostTradeAnalyzer(self.config, storage=self.storage)
        self.report_generator = ReportGenerator(self.config)
        self.grafana_gen = GrafanaDashboardGenerator(self.config.get("monitor", {}))

        # ========================================
        #  运行时状态
        # ========================================
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
        self._daily_trade_count = 0
        self._batch_predictions: Dict[str, float] = {}
        self._stock_pool: List[str] = []

        # 优雅退出
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("✅ 所有模块初始化完成")

    # ==============================================================
    #  启动 / 停止
    # ==============================================================

    def start(self):
        """启动全链路系统"""
        self._running = True
        logger.info("=" * 60)
        logger.info("  系统启动中...")
        logger.info("=" * 60)

        # Step 1: 启动基础设施
        self.metrics.start()
        self.grafana_gen.generate_main_dashboard()

        # Step 2: 连接QMT
        if self.mode != "backtest":
            if not self.order_manager.connect():
                logger.critical("❌ QMT连接失败")
                self.alert.send_critical("QMT连接失败, 系统无法启动")
                sys.exit(1)
            logger.info("✅ QMT 连接成功")

            # 注册QMT成交回调 -> OMS
            self.order_manager.on_fill(self._on_qmt_fill)

        # Step 3: 加载模型
        self._load_models()

        # Step 4: 加载股票池 & 订阅
        self._stock_pool = self._load_stock_pool()
        self.collector.subscribe(self._stock_pool)
        logger.info(f"✅ 股票池: {len(self._stock_pool)} 只")

        # Step 5: 注册数据回调
        self.collector.register_callback("tick", self._on_tick)
        self.collector.register_callback("orderbook", self._on_orderbook)

        # Step 6: 注册新闻回调 -> NLP
        self.news_crawler.register_callback(self._on_news)

        # Step 7: 启动所有子系统
        self.collector.start()
        self.stream_engine.start()
        self.news_crawler.start()
        self.algo_engine.start()
        self.risk_engine.start()

        # Step 8: 启动后台线程
        thread_defs = [
            ("strategy_loop",     self._strategy_loop),
            ("scheduler_loop",    self._scheduler_loop),
            ("heartbeat_loop",    self._heartbeat_loop),
            ("factor_flush_loop", self._factor_flush_loop),
            ("portfolio_sync",    self._portfolio_sync_loop),
        ]
        for name, target in thread_defs:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
            logger.info(f"  线程启动: {name}")

        self.alert.send_info(
            f"🚀 量化系统已启动\n"
            f"模式: {self.mode}\n"
            f"股票池: {len(self._stock_pool)} 只\n"
            f"资金: {self.config['risk']['total_capital']:,.0f}"
        )
        logger.info("🚀 系统启动完成")

        # 主线程阻塞
        self._main_loop()

    def stop(self):
        """优雅停止"""
        if not self._running:
            return
        self._running = False
        logger.info("系统停止中...")

        # 按逆序停止
        stop_sequence = [
            ("算法引擎",   lambda: self.algo_engine.stop()),
            ("新闻爬虫",   lambda: self.news_crawler.stop()),
            ("流计算引擎", lambda: self.stream_engine.stop()),
            ("数据采集",   lambda: self.collector.stop()),
            ("风控引擎",   lambda: self.risk_engine.stop()),
            ("盘后分析",   lambda: self._run_post_trade()),
            ("OMS日终",    lambda: self.oms.end_of_day_reset()),
            ("QMT断开",    lambda: self.order_manager.disconnect()),
            ("监控停止",   lambda: self.metrics.stop()),
        ]

        for name, fn in stop_sequence:
            try:
                fn()
                logger.info(f"  ✓ {name} 已停止")
            except Exception as e:
                logger.error(f"  ✗ {name} 停止异常: {e}")

        # 等待线程退出
        for t in self._threads:
            t.join(timeout=5)

        self.alert.send_info("🛑 量化系统已安全停止")
        logger.info("🛑 系统已安全停止")

    def _main_loop(self):
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _signal_handler(self, signum, frame):
        logger.info(f"收到信号 {signum}, 准备退出...")
        self._running = False

    # ==============================================================
    #  数据回调
    # ==============================================================

    def _on_tick(self, tick: TickData):
        """L2逐笔成交回调"""
        sym = tick.symbol
        self._tick_buffer[sym].append(tick)
        self._price_cache[sym] = tick.price

        # 转发到流计算引擎
        self.stream_engine.on_tick(
            symbol=sym,
            price=tick.price,
            volume=tick.volume,
            amount=tick.amount,
            direction=tick.direction,
            timestamp=tick.timestamp / 1e6,  # 微秒 -> 秒
        )

    def _on_orderbook(self, snapshot: OrderBookSnapshot):
        """盘口快照回调"""
        sym = snapshot.symbol
        self._orderbook_cache[sym] = snapshot
        self._price_cache[sym] = snapshot.last_price

        # 微观结构因子计算
        ticks = list(self._tick_buffer.get(sym, []))
        if len(ticks) >= 50:
            try:
                factors = self.factor_engine.compute_all(
                    sym, snapshot, ticks, snapshot.timestamp
                )
                self._factor_cache[sym] = factors

                # 追加历史
                factor_arr = np.array(list(factors.values()), dtype=np.float32)
                self._factor_history[sym].append(factor_arr)

                # 写入Redis
                self.redis.set_factors(sym, factors, expire=60)
                self.redis.append_factor_history(sym, factor_arr)

                # 上报指标
                self.metrics.report_factors(sym, factors)

            except Exception as e:
                logger.error(f"因子计算异常 {sym}: {e}")

        # 转发到流计算引擎
        self.stream_engine.on_orderbook(sym, {
            "ask_prices": snapshot.ask_prices.tolist() if hasattr(snapshot.ask_prices, 'tolist') else [],
            "bid_prices": snapshot.bid_prices.tolist() if hasattr(snapshot.bid_prices, 'tolist') else [],
            "ask_volumes": snapshot.ask_volumes.tolist() if hasattr(snapshot.ask_volumes, 'tolist') else [],
            "bid_volumes": snapshot.bid_volumes.tolist() if hasattr(snapshot.bid_volumes, 'tolist') else [],
            "last_price": snapshot.last_price,
        })

    def _on_news(self, news_item):
        """新闻到达回调 -> NLP分析"""
        try:
            result = self.nlp.analyze(
                text=f"{news_item.title} {news_item.content}",
                symbol=news_item.symbols[0] if news_item.symbols else "",
            )

            # 重大事件告警
            if result.event_type in ("major_positive", "major_negative", "warning"):
                self.alert.send_warning(
                    f"📰 {result.event_type.upper()}\n"
                    f"标题: {news_item.title}\n"
                    f"情感: {result.score:+.2f} ({result.label})\n"
                    f"关联: {', '.join(news_item.symbols[:5])}"
                )

            self.metrics.news_count.inc() if hasattr(self.metrics, 'news_count') else None

        except Exception as e:
            logger.error(f"NLP处理异常: {e}")

    def _on_qmt_fill(self, trade):
        """QMT成交回报 -> OMS"""
        try:
            # 查找OMS中的订单
            # trade.order_id 是券商订单号, 需要映射
            self.oms.on_fill(
                order_id=str(trade.order_id),
                fill_price=trade.traded_price,
                fill_volume=trade.traded_volume,
            )

            # 上报指标
            self.metrics.report_order(
                side="buy" if trade.order_sysid else "sell",
                status="filled",
                slippage=0,
            )

            self._daily_trade_count += 1

        except Exception as e:
            logger.error(f"成交回报处理异常: {e}")

    # ==============================================================
    #  策略主循环
    # ==============================================================

    def _strategy_loop(self):
        """
        策略主循环 (每30秒)

        Pipeline:
        ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ 因子汇总  │ → │ 模型推理  │ → │ 信号融合  │ → │ 组合优化  │ → │ 风控检查  │ → │ 算法下单  │
        └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                        ↑ 规则引擎
                                        ↑ NLP情感
                                        ↑ 盘前预测
        """
        logger.info("策略循环已启动")

        while self._running:
            try:
                if not self._is_trading_time():
                    time.sleep(3)
                    continue

                if self.risk_engine.state.halted:
                    logger.warning("系统已熔断, 策略暂停")
                    time.sleep(10)
                    continue

                self._run_strategy_cycle()
                time.sleep(STRATEGY_INTERVAL)

            except Exception as e:
                logger.error(f"策略循环异常: {e}\n{traceback.format_exc()}")
                self.alert.send_warning(f"策略循环异常: {e}")
                time.sleep(10)

    def _run_strategy_cycle(self):
        """单次策略周期"""
        self._cycle_count += 1
        t_start = time.time()
        cid = f"C{self._cycle_count:06d}"

        logger.info(f"{'='*55}")
        logger.info(f"[{cid}] 策略周期开始")

        # ============ Step 1: 因子矩阵准备 ============
        t1 = time.time()

        # 合并: 微观结构因子 + 流计算因子
        all_factors = {}
        for sym in self._stock_pool:
            factors = {}

            # 微观结构因子 (from factor_engine)
            if sym in self._factor_cache:
                factors.update(self._factor_cache[sym])

            # 流计算因子 (from stream_engine via Redis)
            stream_factors = self.stream_engine.get_all_features(sym)
            if stream_factors:
                factors.update(stream_factors)

            # NLP情感因子
            if self.nlp:
                sentiment = self.nlp.get_aggregate_sentiment(sym, hours=12)
                factors["news_sentiment_score"] = sentiment.get("avg_score", 0)
                factors["news_sentiment_momentum"] = sentiment.get("momentum", 0)
                factors["news_count_24h"] = sentiment.get("count", 0)

            # 盘前批处理因子
            if sym in self._batch_predictions:
                factors["batch_pred_score"] = self._batch_predictions[sym]

            if factors:
                all_factors[sym] = factors

        symbols_ready = [s for s in self._stock_pool if s in all_factors and len(all_factors[s]) >= 5]

        if len(symbols_ready) < 5:
            logger.info(f"[{cid}] 因子数据不足 ({len(symbols_ready)}只), 跳过")
            return

        # 构建因子矩阵 (n_stocks, seq_len, n_factors)
        seq_len = self.config["model"]["transformer"]["seq_len"]
        sample_factors = all_factors[symbols_ready[0]]
        n_factors = len(sample_factors)
        n_stocks = len(symbols_ready)

        factor_matrix = np.zeros((n_stocks, seq_len, n_factors), dtype=np.float32)
        for i, sym in enumerate(symbols_ready):
            # 尝试从Redis获取历史序列
            hist = self.redis.get_factor_history(sym, length=seq_len)
            if hist is not None and len(hist) >= 5:
                actual_len = min(hist.shape[0], seq_len)
                actual_features = min(hist.shape[1], n_factors)
                if actual_len < seq_len:
                    pad = np.tile(hist[0:1], (seq_len - actual_len, 1))
                    hist = np.vstack([pad, hist[-actual_len:]])
                factor_matrix[i, :, :actual_features] = hist[-seq_len:, :actual_features]
            else:
                # 回退: 用当前截面填充
                current = np.array(list(all_factors[sym].values())[:n_factors], dtype=np.float32)
                factor_matrix[i, :, :len(current)] = current

        factor_matrix = np.nan_to_num(factor_matrix, nan=0.0)
        dt_factor = time.time() - t1
        logger.info(f"[{cid}] Step1 因子: {n_stocks}只 x {seq_len}步 x {n_factors}因子, {dt_factor*1000:.0f}ms")

        # ============ Step 2: 模型推理 ============
        t2 = time.time()

        # 优先使用ONNX推理
        onnx_result = self.onnx_engine.predict_alpha(factor_matrix)
        if onnx_result is not None:
            tf_returns, tf_vols = onnx_result
            inference_source = "ONNX"
        else:
            # 回退到PyTorch
            tf_returns, tf_vols = self.model_manager.predict(factor_matrix)
            inference_source = "PyTorch"

        # LightGBM (截面因子)
        lgb_features = factor_matrix[:, -1, :]
        try:
            lgb_returns = self.lgb_model.predict(lgb_features)
        except Exception:
            lgb_returns = np.zeros(n_stocks)

        # 模型集成
        pred_returns_dict, pred_vols_dict = self.ensemble.combine(
            predictions={
                "transformer": (tf_returns, tf_vols),
                "lightgbm": (lgb_returns, None),
            },
            symbols=symbols_ready,
        )

        dt_model = time.time() - t2
        self.metrics.model_inference_latency.observe(dt_model * 1000) if hasattr(self.metrics, 'model_inference_latency') else None
        logger.info(f"[{cid}] Step2 推理({inference_source}): {dt_model*1000:.1f}ms")

        # ============ Step 3: 信号融合 (模型 + 规则 + NLP) ============
        t3 = time.time()

        # 获取当前持仓信息
        current_positions = {}
        position_info = {}
        try:
            pos_dict = self.order_manager.get_position_dict()
            for sym, pos in pos_dict.items():
                current_positions[sym] = pos.volume
                position_info[sym] = {
                    "position_volume": pos.volume,
                    "position_pnl_pct": (
                        (self._price_cache.get(sym, 0) - pos.open_price) / pos.open_price
                        if pos.open_price > 0 else 0
                    ),
                    "position_pnl_pct_today": 0,  # 简化
                }
        except Exception as e:
            logger.error(f"查询持仓异常: {e}")
            return

        # 为每只股票生成融合信号
        signals = {}
        for i, sym in enumerate(symbols_ready):
            pred_ret = pred_returns_dict.get(sym, 0)
            pred_vol = pred_vols_dict.get(sym, 0.02)

            sig = self.signal_generator.generate(
                symbol=sym,
                model_pred_return=pred_ret,
                model_pred_vol=pred_vol,
                realtime_factors=all_factors.get(sym, {}),
                position_info=position_info.get(sym),
            )
            signals[sym] = sig

        dt_signal = time.time() - t3

        n_buy = sum(1 for s in signals.values() if s.action == "buy")
        n_sell = sum(1 for s in signals.values() if s.action == "sell")
        n_hold = sum(1 for s in signals.values() if s.action == "hold")
        logger.info(
            f"[{cid}] Step3 信号融合: buy={n_buy} sell={n_sell} hold={n_hold}, {dt_signal*1000:.0f}ms"
        )

        # ============ Step 4: 组合优化 ============
        t4 = time.time()

        # 将信号转换为组合优化器的输入
        signal_returns = {
            sym: sig.strength * 0.01  # 信号强度 -> 预期收益
            for sym, sig in signals.items()
            if sig.action != "hold"
        }
        # 补充hold的股票 (当前持仓)
        for sym in current_positions:
            if sym not in signal_returns:
                sig = signals.get(sym)
                if sig and sig.action == "hold":
                    signal_returns[sym] = 0.001  # 微正, 倾向持有

        targets = self.portfolio_optimizer.optimize(
            pred_returns=signal_returns if signal_returns else pred_returns_dict,
            pred_vols=pred_vols_dict,
            current_prices=dict(self._price_cache),
            current_positions=current_positions,
        )

        dt_portfolio = time.time() - t4
        logger.info(f"[{cid}] Step4 组合优化: {len(targets)}个目标, {dt_portfolio*1000:.0f}ms")

        # ============ Step 5: 风控检查 + 下单 ============
        t5 = time.time()
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
                continue

            # 风控检查
            passed, reason = self.risk_engine.pre_trade_check(
                symbol=symbol, side=side, volume=volume, price=price,
            )

            if not passed:
                logger.warning(f"[{cid}] 🚫 风控拦截 {symbol} {side} {volume}股: {reason}")
                orders_blocked += 1
                self.metrics.report_order(side=side, status="rejected")
                continue

            # 在OMS中创建订单
            algo_type, duration = self._select_algo(symbol, side, volume, price)
            oms_order = self.oms.create_order(
                symbol=symbol,
                side=side,
                price=price,
                volume=volume,
                order_type=OrderType(f"algo_{algo_type.value}"),
                strategy_name=f"main_{cid}",
                signal_strength=signals.get(symbol, None).strength if symbol in signals else 0,
            )

            # 提交算法单
            algo_order = AlgoOrder(
                algo_id=oms_order.order_id,
                symbol=symbol,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                total_volume=volume,
                algo_type=algo_type,
                duration_seconds=duration,
                price_limit=price,
                urgency=signals[symbol].urgency if symbol in signals else 0.5,
            )
            self.algo_engine.submit(algo_order)
            orders_submitted += 1

            self.metrics.report_order(side=side, status="submitted")

            logger.info(
                f"[{cid}] 📤 {side.upper()} {symbol} {volume}股 @{price:.2f} "
                f"algo={algo_type.value} signal={signals.get(symbol, None).strength if symbol in signals else 0:.3f}"
            )

        dt_exec = time.time() - t5
        dt_total = time.time() - t_start

        logger.info(
            f"[{cid}] Step5 执行: 提交={orders_submitted} 拦截={orders_blocked}, {dt_exec*1000:.0f}ms"
        )
        logger.info(
            f"[{cid}] ✅ 周期完成 | 总耗时={dt_total*1000:.0f}ms "
            f"(因子={dt_factor*1000:.0f} 推理={dt_model*1000:.0f} "
            f"信号={dt_signal*1000:.0f} 优化={dt_portfolio*1000:.0f} 执行={dt_exec*1000:.0f})"
        )

        # 上报指标
        self.metrics.report_cycle(cid, n_stocks, n_buy, n_sell, dt_total * 1000)

    # ==============================================================
    #  算法选择
    # ==============================================================

    def _select_algo(self, symbol: str, side: str,
                     volume: int, price: float) -> tuple:
        """根据订单特征选择算法"""
        order_value = volume * price

        if order_value < 20000:
            return AlgoType.TWAP, 30
        elif order_value < 50000:
            return AlgoType.TWAP, 60
        elif order_value < 100000:
            return AlgoType.TWAP, 120
        else:
            if side == "buy":
                return AlgoType.VWAP, 180
            else:
                return AlgoType.ICEBERG, 240

    # ==============================================================
    #  定时任务
    # ==============================================================

    def _scheduler_loop(self):
        """定时任务调度"""
        logger.info("定时任务调度已启动")

        # 记录已执行的任务 (防止同一分钟重复执行)
        executed_today: set = set()

        while self._running:
            try:
                now = datetime.now()
                ct = now.time()
                task_key = f"{now.date()}_{ct.hour:02d}{ct.minute:02d}"

                # ---- 09:00 盘前批处理结果加载 ----
                if ct.hour == 9 and ct.minute == 0 and f"{task_key}_batch" not in executed_today:
                    executed_today.add(f"{task_key}_batch")
                    self._load_batch_results()

                # ---- 09:15 新交易日初始化 ----
                if ct.hour == 9 and ct.minute == 15 and f"{task_key}_init" not in executed_today:
                    executed_today.add(f"{task_key}_init")
                    if self._today != date.today():
                        self._today = date.today()
                        executed_today.clear()
                        self._on_new_trading_day()

                # ---- 09:25 刷新股票池 ----
                if ct.hour == 9 and ct.minute == 25 and f"{task_key}_pool" not in executed_today:
                    executed_today.add(f"{task_key}_pool")
                    self._refresh_stock_pool()

                # ---- 11:30 午间统计 ----
                if ct.hour == 11 and ct.minute == 30 and f"{task_key}_noon" not in executed_today:
                    executed_today.add(f"{task_key}_noon")
                    self._on_noon_break()

                # ---- 14:50 尾盘处理 ----
                if ct.hour == 14 and ct.minute == 50 and f"{task_key}_close" not in executed_today:
                    executed_today.add(f"{task_key}_close")
                    self._on_near_close()

                # ---- 15:05 盘后分析 ----
                if ct.hour == 15 and ct.minute == 5 and f"{task_key}_post" not in executed_today:
                    executed_today.add(f"{task_key}_post")
                    self._run_post_trade()

                # ---- 15:30 模型增量训练 ----
                if ct.hour == 15 and ct.minute == 30 and f"{task_key}_train" not in executed_today:
                    executed_today.add(f"{task_key}_train")
                    self._run_incremental_training()

                # ---- 20:00 盘前批处理 (T+1) ----
                if ct.hour == 20 and ct.minute == 0 and f"{task_key}_nightly" not in executed_today:
                    executed_today.add(f"{task_key}_nightly")
                    self._run_nightly_batch()

                # ---- 每小时保存状态 ----
                if ct.minute == 0 and f"{task_key}_save" not in executed_today:
                    executed_today.add(f"{task_key}_save")
                    self._save_state()

                time.sleep(20)

            except Exception as e:
                logger.error(f"定时任务异常: {e}\n{traceback.format_exc()}")
                time.sleep(60)

    def _on_new_trading_day(self):
        """新交易日初始化"""
        logger.info("📅 新交易日初始化")

        self._cycle_count = 0
        self._daily_trade_count = 0
        self._factor_cache.clear()
        self._factor_history.clear()
        self._tick_buffer.clear()

        # 重置风控
        self.risk_engine.state.daily_pnl = 0.0
        self.risk_engine.state.daily_pnl_pct = 0.0
        self.risk_engine.state.halted = False
        self.risk_engine.state.violations.clear()

        # 重置流计算
        self.stream_engine.reset_daily()

        # 更新OMS可卖量
        self.oms.update_available_volumes()

        # 查询账户
        balance = self.order_manager.get_balance()
        positions = self.order_manager.get_position_dict()
        logger.info(
            f"账户: 总资产={balance.get('total_asset', 0):,.0f} "
            f"现金={balance.get('cash', 0):,.0f} 持仓={len(positions)}只"
        )

        self.alert.send_info(
            f"📅 新交易日 {self._today}\n"
            f"总资产: {balance.get('total_asset', 0):,.0f}\n"
            f"持仓: {len(positions)} 只"
        )

    def _load_batch_results(self):
        """加载盘前批处理结果"""
        try:
            batch_preds = self.redis.get_state("batch_predictions")
            if batch_preds:
                self._batch_predictions = batch_preds
                logger.info(f"盘前预测已加载: {len(batch_preds)} 只")
            else:
                logger.warning("未找到盘前预测结果")
        except Exception as e:
            logger.error(f"加载盘前结果异常: {e}")

    def _refresh_stock_pool(self):
        """刷新股票池"""
        try:
            # 优先使用盘前批处理的股票池
            cached_pool = self.redis.get_state("stock_pool")
            if cached_pool:
                self._stock_pool = cached_pool
            else:
                self._stock_pool = self._load_stock_pool()

            self.collector.subscribe(self._stock_pool)
            logger.info(f"股票池已刷新: {len(self._stock_pool)} 只")
        except Exception as e:
            logger.error(f"股票池刷新失败: {e}")

    def _on_noon_break(self):
        """午间休市"""
        logger.info("🕐 午间休市")
        risk_report = self.risk_engine.get_risk_report()
        portfolio = self.oms.get_portfolio_summary()
        logger.info(f"午间风控: {risk_report}")
        logger.info(f"午间持仓: {portfolio}")

    def _on_near_close(self):
        """尾盘处理 (14:50)"""
        logger.info("⏰ 尾盘处理")

        if self.risk_engine.state.daily_pnl_pct < -0.015:
            logger.warning("日亏损超1.5%, 尾盘减仓")
            self.alert.send_warning(
                f"⚠️ 尾盘减仓触发\n日亏损: {self.risk_engine.state.daily_pnl_pct:.2%}"
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
                passed, _ = self.risk_engine.pre_trade_check(sym, "sell", sell_volume, price)
                if passed and price > 0:
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
        """盘后分析 + 日报"""
        logger.info("📊 盘后分析开始")
        try:
            # 1. 获取数据
            balance = self.order_manager.get_balance()
            positions_raw = self.order_manager.get_position_dict()
            risk_report = self.risk_engine.get_risk_report()
            trades = [
                {
                    "timestamp": t.timestamp,
                    "symbol": t.symbol,
                    "side": t.side,
                    "volume": t.volume,
                    "price": t.price,
                    "amount": t.amount,
                    "commission": t.commission,
                }
                for t in self.oms.get_today_trades()
            ]

            # 2. 保存日报到PostTradeAnalyzer
            report_data = {
                "date": str(self._today),
                "total_asset": balance.get("total_asset", 0),
                "cash": balance.get("cash", 0),
                "daily_pnl": self.risk_engine.state.daily_pnl,
                "daily_pnl_pct": f"{self.risk_engine.state.daily_pnl_pct:.2%}",
                "max_drawdown_pct": risk_report.get("max_drawdown_pct", "0%"),
                "trade_count": self._daily_trade_count,
                "position_count": len(positions_raw),
                "total_position_pct": risk_report.get("total_position_pct", "0%"),
                "strategy_cycles": self._cycle_count,
            }
            self.post_analyzer.save_daily_report(report_data)

            # 3. 计算绩效
            performance = self.post_analyzer.compute_performance()

            # 4. 持仓明细
            positions_detail = {}
            for sym, pos in positions_raw.items():
                price = self._price_cache.get(sym, pos.open_price)
                positions_detail[sym] = {
                    "volume": pos.volume,
                    "avg_cost": pos.open_price,
                    "current_price": price,
                    "market_value": pos.volume * price,
                    "unrealized_pnl": (price - pos.open_price) * pos.volume,
                    "unrealized_pnl_pct": (
                        (price - pos.open_price) / pos.open_price
                        if pos.open_price > 0 else 0
                    ),
                }

            # 5. 滑点分析
            slippage_report = self.oms.get_slippage_report()

            # 6. 生成HTML报告
            report_path = self.report_generator.generate_daily_report(
                performance=performance,
                positions=positions_detail,
                trades=trades,
                risk_report=risk_report,
                slippage_report=slippage_report,
            )

            # 7. 推送日报
            self.alert.send_trade_report({
                **report_data,
                "sharpe": performance.get("sharpe_ratio", 0),
                "total_return": performance.get("total_return", 0),
            })

            # 8. 上报Prometheus
            total_capital = self.config["risk"]["total_capital"]
            self.metrics.report_portfolio(
                total_asset=balance.get("total_asset", 0),
                cash=balance.get("cash", 0),
                market_value=balance.get("market_value", 0),
                position_count=len(positions_raw),
                daily_pnl=self.risk_engine.state.daily_pnl,
                daily_pnl_pct=self.risk_engine.state.daily_pnl_pct,
                max_dd_pct=self.risk_engine.state.max_drawdown_pct,
                nav=balance.get("total_asset", total_capital) / total_capital,
            )

            logger.info(f"📊 盘后分析完成, 报告: {report_path}")

        except Exception as e:
            logger.error(f"盘后分析异常: {e}\n{traceback.format_exc()}")

    def _run_incremental_training(self):
        """增量训练"""
        retrain_interval = self.config["model"]["ensemble"].get("retrain_days", 5)
        day_of_year = self._today.timetuple().tm_yday
        if day_of_year % retrain_interval != 0:
            return

        logger.info("🧠 增量训练开始...")
        try:
            # 训练逻辑 (简化)
            model_path = f"models/model_{self._today}.pt"
            self.model_manager.save(model_path)
            self.model_manager.save("models/latest_model.pt")

            # 导出ONNX
            import torch
            dummy = torch.randn(
                1,
                self.config["model"]["transformer"]["seq_len"],
                50,  # n_factors
            ).to(self.model_manager.device)

            onnx_path = f"models/alpha_transformer_{self._today}.onnx"
            self.onnx_engine.export_pytorch_to_onnx(
                self.model_manager.model, dummy, onnx_path
            )

            # 热更新ONNX
            self.onnx_engine.hot_reload("alpha_transformer", onnx_path)

            self.alert.send_info("🧠 模型增量训练完成")
            logger.info(f"模型已保存并热更新: {onnx_path}")

        except Exception as e:
            logger.error(f"增量训练异常: {e}\n{traceback.format_exc()}")

    def _run_nightly_batch(self):
        """晚间批处理 (T+1准备)"""
        logger.info("🌙 晚间批处理开始...")
        try:
            result = self.batch_processor.run_nightly()
            self._batch_predictions = result.get("predictions", {})
            logger.info(
                f"批处理完成: 股票池={len(result.get('stock_pool', []))} "
                f"预测={len(self._batch_predictions)} "
                f"IC={result.get('model_metrics', {}).get('val_ic', 'N/A')}"
            )
            self.alert.send_info(
                f"🌙 晚间批处理完成\n"
                f"股票池: {len(result.get('stock_pool', []))} 只\n"
                f"验证IC: {result.get('model_metrics', {}).get('val_ic', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"晚间批处理异常: {e}\n{traceback.format_exc()}")

    # ==============================================================
    #  心跳监控
    # ==============================================================

    def _heartbeat_loop(self):
        """心跳监控"""
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
                    "active_orders": len(self.oms.get_active_orders()),
                    "tick_buffer_total": sum(len(v) for v in self._tick_buffer.values()),
                    "stream_events": self.stream_engine.get_stats().get("events_processed", 0),
                    "cycle_count": self._cycle_count,
                    "daily_trades": self._daily_trade_count,
                }

                # 异常检测
                if not status["qmt_connected"] and self._is_trading_time():
                    self.alert.send_critical("QMT 连接断开!")

                if status["symbols_with_data"] == 0 and self._is_trading_time():
                    self.alert.send_warning("无行情数据!")

                # 上报
                self.metrics.report_heartbeat(status)
                self.metrics.report_risk(
                    level=status["risk_level"],
                    halted=status["risk_halted"],
                )

                # 每5分钟日志
                if self._cycle_count % 10 == 0:
                    logger.info(f"💓 心跳: {status}")

                time.sleep(30)

            except Exception as e:
                logger.error(f"心跳异常: {e}")
                time.sleep(60)

    # ==============================================================
    #  因子落盘
    # ==============================================================

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
                            if not np.isnan(fval) and not np.isinf(fval):
                                batch.append((sym, now, fname, float(fval)))

                    if batch:
                        try:
                            self.storage.client.execute(
                                "INSERT INTO factor_values VALUES", batch
                            )
                        except Exception as e:
                            logger.error(f"因子落盘写入异常: {e}")

                time.sleep(10)

            except Exception as e:
                logger.error(f"因子落盘异常: {e}")
                time.sleep(30)

    # ==============================================================
    #  持仓同步
    # ==============================================================

    def _portfolio_sync_loop(self):
        """定期同步持仓市值到OMS和Prometheus"""
        logger.info("持仓同步线程已启动")

        while self._running:
            try:
                if self._is_trading_time():
                    # 更新OMS持仓市值
                    self.oms.update_market_prices(dict(self._price_cache))

                    # 上报Prometheus
                    balance = self.order_manager.get_balance()
                    if balance:
                        total_capital = self.config["risk"]["total_capital"]
                        self.metrics.report_portfolio(
                            total_asset=balance.get("total_asset", 0),
                            cash=balance.get("cash", 0),
                            market_value=balance.get("market_value", 0),
                            position_count=len(self.order_manager.get_position_dict()),
                            daily_pnl=self.risk_engine.state.daily_pnl,
                            daily_pnl_pct=self.risk_engine.state.daily_pnl_pct,
                            max_dd_pct=self.risk_engine.state.max_drawdown_pct,
                            nav=balance.get("total_asset", total_capital) / total_capital,
                        )

                time.sleep(5)

            except Exception as e:
                logger.error(f"持仓同步异常: {e}")
                time.sleep(15)

    # ==============================================================
    #  辅助方法
    # ==============================================================

    def _is_trading_time(self) -> bool:
        now = datetime.now().time()
        return (MORNING_OPEN <= now <= MORNING_CLOSE) or \
               (AFTERNOON_OPEN <= now <= AFTERNOON_CLOSE)

    def _load_stock_pool(self) -> List[str]:
        pool_file = Path("config/stock_pool.txt")
        if pool_file.exists():
            symbols = [
                line.strip()
                for line in pool_file.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
            return symbols
        logger.warning("未找到股票池文件")
        return []

    def _load_models(self):
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        # Transformer
        tf_path = model_dir / "latest_model.pt"
        if tf_path.exists():
            self.model_manager.load(str(tf_path))
            logger.info(f"✅ Transformer模型: {tf_path}")

        # LightGBM
        lgb_path = model_dir / "latest_lgb.txt"
        if lgb_path.exists():
            self.lgb_model.load(str(lgb_path))
            logger.info(f"✅ LightGBM模型: {lgb_path}")

        # ONNX
        onnx_path = model_dir / "alpha_transformer.onnx"
        if onnx_path.exists():
            self.onnx_engine.load_model("alpha_transformer", str(onnx_path))
            logger.info(f"✅ ONNX模型: {onnx_path}")

    def _save_state(self):
        try:
            state = {
                "date": str(self._today),
                "cycle_count": self._cycle_count,
                "daily_trade_count": self._daily_trade_count,
                "risk_state": self.risk_engine.get_risk_report(),
                "portfolio": self.oms.get_portfolio_summary(),
                "stream_stats": self.stream_engine.get_stats(),
                "onnx_stats": self.onnx_engine.get_stats(),
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
    ║       Quantitative Trading System v1.0                   ║
    ║       L2 + NLP + GPU + ONNX + QMT + Grafana              ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
        """
        print(banner)


# ============================================================
#  入口
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="量化交易系统")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="配置文件")
    parser.add_argument("--mode", type=str, choices=["production", "paper", "backtest"], default=None, help="运行模式")
    return parser.parse_args()


def main():
    args = parse_args()

    # 创建目录
    for d in ["logs", "models", "data", "config", "config/rules", "reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    system = QuantTradingSystem(config_path=args.config)

    if args.mode:
        system.config["system"]["mode"] = args.mode
        system.mode = args.mode

    system.start()


if __name__ == "__main__":
    main()

