"""
盘后分析模块
- 每日绩效统计
- 因子IC/IR分析
- 风险归因
- 历史回顾
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class PostTradeAnalyzer:
    """盘后分析器"""

    def __init__(self, config: dict, storage=None):
        self.config = config
        self.storage = storage
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)

        # 历史净值序列
        self._nav_history: List[dict] = []
        self._load_nav_history()

    def _load_nav_history(self):
        """加载历史净值"""
        nav_file = self.report_dir / "nav_history.json"
        if nav_file.exists():
            with open(nav_file, "r") as f:
                self._nav_history = json.load(f)
            logger.info(f"加载历史净值: {len(self._nav_history)} 天")

    def _save_nav_history(self):
        nav_file = self.report_dir / "nav_history.json"
        with open(nav_file, "w") as f:
            json.dump(self._nav_history, f, indent=2)

    # ==================== 每日报告 ====================

    def save_daily_report(self, report: dict):
        """保存每日报告并更新净值序列"""
        today = report.get("date", str(date.today()))

        # 更新净值序列
        total_asset = report.get("total_asset", 0)
        initial_capital = self.config.get("risk", {}).get("total_capital", 1000000)

        nav = total_asset / initial_capital if initial_capital > 0 else 1.0

        nav_record = {
            "date": today,
            "nav": round(nav, 6),
            "total_asset": total_asset,
            "daily_pnl": report.get("daily_pnl", 0),
            "daily_return": round(float(
                report.get("daily_pnl", 0)
            ) / initial_capital, 6),
            "position_count": report.get("position_count", 0),
            "trade_count": report.get("trade_count", 0),
        }

        # 去重更新
        self._nav_history = [
            n for n in self._nav_history if n["date"] != today
        ]
        self._nav_history.append(nav_record)
        self._nav_history.sort(key=lambda x: x["date"])
        self._save_nav_history()

        # 保存详细日报
        report_file = self.report_dir / f"daily_{today}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"日报已保存: {report_file}")

    # ==================== 绩效分析 ====================

    def compute_performance(self, days: int = None) -> dict:
        """
        计算绩效指标

        Returns:
            {
                total_return, annual_return, sharpe_ratio,
                max_drawdown, calmar_ratio, win_rate,
                avg_daily_return, volatility, ...
            }
        """
        if not self._nav_history:
            return {}

        history = self._nav_history
        if days:
            history = history[-days:]

        navs = np.array([h["nav"] for h in history])
        returns = np.array([h["daily_return"] for h in history])

        if len(navs) < 2:
            return {"error": "数据不足"}

        # 总收益
        total_return = (navs[-1] / navs[0]) - 1

        # 年化收益
        n_days = len(navs)
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        # 波动率
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)

        # 夏普比率 (无风险利率2.5%)
        rf_daily = 0.025 / 252
        excess_returns = returns - rf_daily
        sharpe = (
            np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
            * np.sqrt(252)
        )

        # 最大回撤
        peak = np.maximum.accumulate(navs)
        drawdowns = (navs - peak) / peak
        max_drawdown = float(np.min(drawdowns))
        max_dd_end_idx = np.argmin(drawdowns)
        max_dd_start_idx = np.argmax(navs[:max_dd_end_idx + 1]) if max_dd_end_idx > 0 else 0

        # Calmar比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 胜率
        win_days = np.sum(returns > 0)
        total_days = len(returns)
        win_rate = win_days / total_days if total_days > 0 else 0

        # 盈亏比
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 1e-8
        profit_loss_ratio = avg_win / avg_loss

        # Sortino比率 (只考虑下行波动)
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
        sortino = (np.mean(excess_returns) * 252) / downside_vol

        # 最大连续亏损天数
        is_loss = returns < 0
        max_consecutive_loss = 0
        current_streak = 0
        for loss in is_loss:
            if loss:
                current_streak += 1
                max_consecutive_loss = max(max_consecutive_loss, current_streak)
            else:
                current_streak = 0

        return {
            "period_days": n_days,
            "start_date": history[0]["date"],
            "end_date": history[-1]["date"],
            "total_return": round(total_return * 100, 2),
            "annual_return": round(annual_return * 100, 2),
            "annual_volatility": round(annual_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown": round(max_drawdown * 100, 2),
            "max_dd_start": history[max_dd_start_idx]["date"],
            "max_dd_end": history[max_dd_end_idx]["date"],
            "win_rate": round(win_rate * 100, 2),
            "profit_loss_ratio": round(profit_loss_ratio, 3),
            "avg_daily_return": round(float(np.mean(returns)) * 100, 4),
            "max_daily_return": round(float(np.max(returns)) * 100, 2),
            "min_daily_return": round(float(np.min(returns)) * 100, 2),
            "max_consecutive_loss_days": max_consecutive_loss,
            "current_nav": round(float(navs[-1]), 6),
        }

    # ==================== 因子分析 ====================

    def analyze_factor_ic(self, factor_values: Dict[str, Dict[str, float]],
                           forward_returns: Dict[str, float]) -> Dict[str, dict]:
        """
        因子IC分析

        Args:
            factor_values: {symbol: {factor_name: value}}
            forward_returns: {symbol: T+1 return}

        Returns:
            {factor_name: {ic, rank_ic, t_stat, ...}}
        """
        if not factor_values or not forward_returns:
            return {}

        # 构建DataFrame
        symbols = list(set(factor_values.keys()) & set(forward_returns.keys()))
        if len(symbols) < 20:
            return {"error": "样本不足"}

        returns = np.array([forward_returns[s] for s in symbols])

        # 获取所有因子名
        sample_factors = next(iter(factor_values.values()))
        factor_names = list(sample_factors.keys())

        results = {}
        for fname in factor_names:
            values = np.array([
                factor_values[s].get(fname, np.nan) for s in symbols
            ])

            # 去除NaN
            valid = ~(np.isnan(values) | np.isnan(returns))
            if np.sum(valid) < 20:
                continue

            v = values[valid]
            r = returns[valid]

            # Pearson IC
            ic = float(np.corrcoef(v, r)[0, 1])

            # Rank IC (Spearman)
            from scipy import stats
            rank_ic, p_value = stats.spearmanr(v, r)

            # t统计量
            n = len(v)
            t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic ** 2 + 1e-8)

            results[fname] = {
                "ic": round(ic, 4),
                "rank_ic": round(float(rank_ic), 4),
                "t_stat": round(float(t_stat), 3),
                "p_value": round(float(p_value), 4),
                "ic_abs": round(abs(ic), 4),
                "n_samples": n,
            }

        # 按|IC|排序
        results = dict(
            sorted(results.items(), key=lambda x: x[1]["ic_abs"], reverse=True)
        )

        return results

    # ==================== 风险归因 ====================

    def risk_attribution(self, positions: Dict[str, dict],
                          market_returns: Dict[str, float]) -> dict:
        """
        风险归因分析

        Returns:
            {
                "alpha": 超额收益,
                "beta_contribution": 市场贡献,
                "sector_attribution": 行业归因,
                "stock_attribution": 个股归因 (Top/Bottom 5),
            }
        """
        if not positions or not market_returns:
            return {}

        # 个股归因
        stock_pnl = {}
        total_pnl = 0
        total_value = 0

        for sym, pos in positions.items():
            pnl = pos.get("unrealized_pnl", 0) + pos.get("realized_pnl", 0)
            mv = pos.get("market_value", 0)
            stock_pnl[sym] = pnl
            total_pnl += pnl
            total_value += mv

        # 排序
        sorted_stocks = sorted(stock_pnl.items(), key=lambda x: x[1], reverse=True)
        top5 = sorted_stocks[:5]
        bottom5 = sorted_stocks[-5:]

        # Beta估算 (简化)
        portfolio_return = total_pnl / total_value if total_value > 0 else 0
        market_avg_return = np.mean(list(market_returns.values())) if market_returns else 0
        alpha = portfolio_return - market_avg_return

        return {
            "portfolio_return": round(portfolio_return * 100, 4),
            "market_return": round(market_avg_return * 100, 4),
            "alpha": round(alpha * 100, 4),
            "total_pnl": round(total_pnl, 2),
            "top5_contributors": [
                {"symbol": s, "pnl": round(p, 2)} for s, p in top5
            ],
            "bottom5_contributors": [
                {"symbol": s, "pnl": round(p, 2)} for s, p in bottom5
            ],
        }
