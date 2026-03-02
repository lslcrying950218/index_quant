"""
盘前批处理引擎
- T-1晚间运行
- 日频因子计算 + 特征工程
- LightGBM模型训练/预测
- 股票池筛选
- 生成次日初始信号
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    logger.warning("lightgbm 未安装")


class BatchProcessor:
    """盘前批处理引擎"""

    def __init__(self, config: dict, storage=None, redis_cache=None):
        self.config = config
        self.storage = storage
        self.redis = redis_cache

        # 模型配置
        self.lgb_params = config.get("lgb", {})
        self.lookback_days = config.get("lookback_days", 60)
        self.predict_horizon = config.get("predict_horizon", 1)  # 预测T+1

        # 因子列表
        self.daily_factors = [
            # 量价因子
            "ret_1d", "ret_5d", "ret_20d",
            "vol_ratio_5d", "vol_ratio_20d",
            "turnover_rate", "turnover_rate_5d",
            "amplitude_5d", "amplitude_20d",
            # 动量因子
            "momentum_10d", "momentum_20d", "momentum_60d",
            "rsi_14", "rsi_28",
            # 波动率因子
            "realized_vol_5d", "realized_vol_20d",
            "vol_skew_20d",
            # 资金流因子
            "money_flow_20d", "large_order_ratio_5d",
            "north_flow_5d",
            # 基本面因子
            "pe_ttm", "pb", "ps_ttm",
            "roe_ttm", "revenue_growth_yoy",
            "market_cap_log",
            # 技术因子
            "ma5_bias", "ma20_bias", "ma60_bias",
            "macd_diff", "macd_dea",
            "boll_width", "boll_position",
            # 行业/板块
            "industry_momentum", "sector_rotation_score",
        ]

    def run_nightly(self, target_date: date = None) -> dict:
        """
        T-1晚间批处理主流程

        Returns:
            {
                "stock_pool": [...],
                "predictions": {symbol: score},
                "model_metrics": {...},
            }
        """
        if target_date is None:
            target_date = date.today()

        t0 = time.time()
        logger.info(f"{'='*60}")
        logger.info(f"盘前批处理开始: 目标日期 {target_date}")
        logger.info(f"{'='*60}")

        # Step 1: 数据准备
        logger.info("[Step 1/6] 数据准备...")
        raw_data = self._load_daily_data(target_date)
        logger.info(f"  加载 {len(raw_data)} 只股票, {self.lookback_days} 天数据")

        # Step 2: 因子计算
        logger.info("[Step 2/6] 日频因子计算...")
        factor_df = self._compute_daily_factors(raw_data)
        logger.info(f"  计算 {len(self.daily_factors)} 个因子")

        # Step 3: 股票池筛选
        logger.info("[Step 3/6] 股票池筛选...")
        stock_pool = self._filter_stock_pool(factor_df, target_date)
        logger.info(f"  筛选后 {len(stock_pool)} 只股票")

        # Step 4: 标签构建 & 训练
        logger.info("[Step 4/6] 模型训练...")
        model, metrics = self._train_model(factor_df, target_date)
        logger.info(f"  训练指标: {metrics}")

        # Step 5: 次日预测
        logger.info("[Step 5/6] 生成预测...")
        predictions = self._predict(model, factor_df, stock_pool, target_date)
        logger.info(f"  Top5预测: {sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]}")

        # Step 6: 写入缓存
        logger.info("[Step 6/6] 写入缓存...")
        self._write_to_cache(stock_pool, predictions, target_date)

        elapsed = time.time() - t0
        logger.info(f"✅ 盘前批处理完成, 耗时 {elapsed:.1f}s")

        return {
            "stock_pool": stock_pool,
            "predictions": predictions,
            "model_metrics": metrics,
            "elapsed_seconds": elapsed,
        }

    # ==================== Step 1: 数据加载 ====================

    def _load_daily_data(self, target_date: date) -> pd.DataFrame:
        """从ClickHouse加载日频数据"""
        start_date = target_date - timedelta(days=self.lookback_days + 30)

        if self.storage:
            query = f"""
                SELECT symbol, trade_date, open, high, low, close,
                       volume, amount, turnover_rate, pe_ttm, pb
                FROM daily_quotes
                WHERE trade_date BETWEEN '{start_date}' AND '{target_date}'
                ORDER BY symbol, trade_date
            """
            data = self.storage.client.execute(query)
            df = pd.DataFrame(data, columns=[
                "symbol", "trade_date", "open", "high", "low", "close",
                "volume", "amount", "turnover_rate", "pe_ttm", "pb"
            ])
            return df

        # 模拟数据 (开发测试用)
        logger.warning("使用模拟数据")
        return pd.DataFrame()

    # ==================== Step 2: 因子计算 ====================

    def _compute_daily_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算日频因子"""
        if df.empty:
            return df

        results = []
        for symbol, group in df.groupby("symbol"):
            group = group.sort_values("trade_date").copy()
            factors = self._calc_single_stock_factors(group)
            factors["symbol"] = symbol
            results.append(factors)

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def _calc_single_stock_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算单只股票的因子"""
        close = df["close"].values
        volume = df["volume"].values
        high = df["high"].values
        low = df["low"].values
        amount = df["amount"].values

        factors = df[["trade_date"]].copy()

        # 收益率
        factors["ret_1d"] = pd.Series(close).pct_change(1).values
        factors["ret_5d"] = pd.Series(close).pct_change(5).values
        factors["ret_20d"] = pd.Series(close).pct_change(20).values

        # 成交量比
        vol_ma5 = pd.Series(volume).rolling(5).mean()
        vol_ma20 = pd.Series(volume).rolling(20).mean()
        factors["vol_ratio_5d"] = (volume / vol_ma5.values)
        factors["vol_ratio_20d"] = (volume / vol_ma20.values)

        # 动量
        factors["momentum_10d"] = pd.Series(close).pct_change(10).values
        factors["momentum_20d"] = pd.Series(close).pct_change(20).values
        factors["momentum_60d"] = pd.Series(close).pct_change(60).values

        # RSI
        factors["rsi_14"] = self._calc_rsi(close, 14)
        factors["rsi_28"] = self._calc_rsi(close, 28)

        # 波动率
        log_ret = np.log(close[1:] / close[:-1])
        log_ret = np.concatenate([[0], log_ret])
        factors["realized_vol_5d"] = pd.Series(log_ret).rolling(5).std().values
        factors["realized_vol_20d"] = pd.Series(log_ret).rolling(20).std().values

        # 均线偏离
        ma5 = pd.Series(close).rolling(5).mean().values
        ma20 = pd.Series(close).rolling(20).mean().values
        ma60 = pd.Series(close).rolling(60).mean().values
        factors["ma5_bias"] = (close - ma5) / (ma5 + 1e-8)
        factors["ma20_bias"] = (close - ma20) / (ma20 + 1e-8)
        factors["ma60_bias"] = (close - ma60) / (ma60 + 1e-8)

        # 振幅
        factors["amplitude_5d"] = (
            pd.Series(high).rolling(5).max().values -
            pd.Series(low).rolling(5).min().values
        ) / (close + 1e-8)

        # 布林带
        ma20_series = pd.Series(close).rolling(20).mean()
        std20 = pd.Series(close).rolling(20).std()
        upper = ma20_series + 2 * std20
        lower = ma20_series - 2 * std20
        factors["boll_width"] = ((upper - lower) / ma20_series).values
        factors["boll_position"] = (
            (close - lower.values) / (upper.values - lower.values + 1e-8)
        )

        return factors

    @staticmethod
    def _calc_rsi(prices: np.ndarray, period: int) -> np.ndarray:
        """计算RSI"""
        deltas = np.diff(prices)
        deltas = np.concatenate([[0], deltas])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = pd.Series(gains).rolling(period).mean().values
        avg_loss = pd.Series(losses).rolling(period).mean().values

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - 100 / (1 + rs)
        return rsi

    # ==================== Step 3: 股票池筛选 ====================

    def _filter_stock_pool(self, factor_df: pd.DataFrame,
                            target_date: date) -> List[str]:
        """股票池筛选"""
        if factor_df.empty:
            return []

        # 取最新一天的数据
        latest = factor_df.groupby("symbol").last().reset_index()

        pool = latest.copy()

        # 剔除条件
        # 1. 流动性: 日均成交额 > 2000万 (需要amount字段)
        # 2. 波动率不能太低 (僵尸股)
        if "realized_vol_20d" in pool.columns:
            pool = pool[pool["realized_vol_20d"] > 0.005]

        # 3. RSI不在极端区域 (避免追涨杀跌)
        if "rsi_14" in pool.columns:
            pool = pool[(pool["rsi_14"] > 15) & (pool["rsi_14"] < 85)]

        return pool["symbol"].tolist()

    # ==================== Step 4: 模型训练 ====================

    def _train_model(self, factor_df: pd.DataFrame,
                      target_date: date) -> Tuple[Optional[object], dict]:
        """训练LightGBM模型"""
        if lgb is None or factor_df.empty:
            return None, {"error": "lgb not available or no data"}

        # 构建标签: T+1收益率
        factor_df = factor_df.copy()
        factor_df["label"] = factor_df.groupby("symbol")["ret_1d"].shift(-1)
        factor_df = factor_df.dropna(subset=["label"])

        # 特征列
        feature_cols = [
            c for c in factor_df.columns
            if c not in ["symbol", "trade_date", "label"]
        ]

        # 训练/验证分割 (时间序列分割)
        dates = sorted(factor_df["trade_date"].unique())
        split_idx = int(len(dates) * 0.8)
        train_dates = dates[:split_idx]
        val_dates = dates[split_idx:]

        train_mask = factor_df["trade_date"].isin(train_dates)
        val_mask = factor_df["trade_date"].isin(val_dates)

        X_train = factor_df.loc[train_mask, feature_cols].values
        y_train = factor_df.loc[train_mask, "label"].values
        X_val = factor_df.loc[val_mask, feature_cols].values
        y_val = factor_df.loc[val_mask, "label"].values

        # 处理NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)

        # 训练
        params = {
            "objective": "regression",
            "metric": "mse",
            "num_leaves": self.lgb_params.get("num_leaves", 63),
            "learning_rate": self.lgb_params.get("learning_rate", 0.05),
            "feature_fraction": self.lgb_params.get("feature_fraction", 0.8),
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
        }

        train_set = lgb.Dataset(X_train, y_train)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=self.lgb_params.get("n_estimators", 500),
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100),
            ],
        )

        # 评估
        val_pred = model.predict(X_val)
        ic = np.corrcoef(val_pred, y_val)[0, 1]
        mse = np.mean((val_pred - y_val) ** 2)

        # 保存模型
        model_path = f"models/lgb_{target_date}.txt"
        model.save_model(model_path)
        model.save_model("models/latest_lgb.txt")

        metrics = {
            "val_ic": round(ic, 4),
            "val_mse": round(mse, 6),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "best_iteration": model.best_iteration,
            "feature_importance_top10": dict(
                sorted(
                    zip(feature_cols, model.feature_importance()),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
        }

        return model, metrics

    # ==================== Step 5: 预测 ====================

    def _predict(self, model, factor_df: pd.DataFrame,
                  stock_pool: List[str], target_date: date) -> Dict[str, float]:
        """生成次日预测"""
        if model is None or factor_df.empty:
            return {}

        # 取最新一天的因子
        latest = factor_df.groupby("symbol").last().reset_index()
        latest = latest[latest["symbol"].isin(stock_pool)]

        feature_cols = [
            c for c in latest.columns
            if c not in ["symbol", "trade_date", "label"]
        ]

        X = latest[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)

        preds = model.predict(X)

        predictions = {
            sym: float(score)
            for sym, score in zip(latest["symbol"].values, preds)
        }

        return predictions

    # ==================== Step 6: 写入缓存 ====================

    def _write_to_cache(self, stock_pool: List[str],
                         predictions: Dict[str, float],
                         target_date: date):
        """将结果写入Redis供盘中使用"""
        if not self.redis:
            return

        self.redis.set_state("stock_pool", stock_pool)
        self.redis.set_state("batch_predictions", predictions)
        self.redis.set_state("batch_date", str(target_date))
        self.redis.set_state("batch_timestamp", datetime.now().isoformat())

        logger.info(f"批处理结果已写入Redis: {len(stock_pool)}只股票, {len(predictions)}个预测")
