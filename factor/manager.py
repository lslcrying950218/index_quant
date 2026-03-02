"""
因子管理器
- 因子注册 & 生命周期管理
- IC/IR 监控
- 因子筛选 & 自动淘汰
- 因子相关性分析
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from collections import defaultdict, deque
from datetime import datetime, date
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class FactorMeta:
    """因子元信息"""

    def __init__(self, name: str, category: str, description: str = "",
                 compute_fn: Optional[Callable] = None):
        self.name = name
        self.category = category          # microstructure / orderflow / volatility / sentiment / fundamental
        self.description = description
        self.compute_fn = compute_fn
        self.enabled = True
        self.created_at = datetime.now().isoformat()

        # IC监控
        self.ic_history: deque = deque(maxlen=60)     # 最近60天IC
        self.rank_ic_history: deque = deque(maxlen=60)
        self.turnover_history: deque = deque(maxlen=60)  # 因子换手率


class FactorManager:
    """因子管理器"""

    def __init__(self, config: dict):
        self.config = config
        self.ic_monitor_window = config.get("ic_monitor_window", 20)
        self.min_ic_threshold = config.get("min_ic_threshold", 0.02)
        self.max_correlation = config.get("max_correlation", 0.7)

        # 因子注册表
        self._factors: Dict[str, FactorMeta] = {}

        # 因子值缓存: {date: {symbol: {factor_name: value}}}
        self._daily_values: Dict[str, Dict[str, Dict[str, float]]] = {}

        # 收益率缓存: {date: {symbol: forward_return}}
        self._forward_returns: Dict[str, Dict[str, float]] = {}

        # IC汇总
        self._ic_summary: Dict[str, dict] = {}

        self._report_dir = Path("reports/factors")
        self._report_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 因子注册 ====================

    def register(self, name: str, category: str,
                 description: str = "", compute_fn: Callable = None):
        """注册因子"""
        meta = FactorMeta(
            name=name,
            category=category,
            description=description,
            compute_fn=compute_fn,
        )
        self._factors[name] = meta
        logger.debug(f"因子注册: {name} [{category}]")

    def register_batch(self, factor_list: List[dict]):
        """批量注册"""
        for f in factor_list:
            self.register(**f)
        logger.info(f"批量注册 {len(factor_list)} 个因子")

    def enable(self, name: str):
        if name in self._factors:
            self._factors[name].enabled = True

    def disable(self, name: str):
        if name in self._factors:
            self._factors[name].enabled = False
            logger.info(f"因子已禁用: {name}")

    def get_enabled_factors(self) -> List[str]:
        return [n for n, m in self._factors.items() if m.enabled]

    # ==================== IC分析 ====================

    def record_daily_values(self, trade_date: str,
                             factor_values: Dict[str, Dict[str, float]],
                             forward_returns: Dict[str, float]):
        """
        记录每日因子值和前瞻收益 (盘后调用)

        Args:
            trade_date: "2024-01-15"
            factor_values: {symbol: {factor_name: value}}
            forward_returns: {symbol: T+1 return}
        """
        self._daily_values[trade_date] = factor_values
        self._forward_returns[trade_date] = forward_returns

        # 清理过期数据 (保留最近90天)
        dates = sorted(self._daily_values.keys())
        if len(dates) > 90:
            for d in dates[:-90]:
                self._daily_values.pop(d, None)
                self._forward_returns.pop(d, None)

    def compute_daily_ic(self, trade_date: str) -> Dict[str, dict]:
        """
        计算某日所有因子的IC

        Returns:
            {factor_name: {"ic": ..., "rank_ic": ..., "t_stat": ...}}
        """
        factor_values = self._daily_values.get(trade_date, {})
        forward_returns = self._forward_returns.get(trade_date, {})

        if not factor_values or not forward_returns:
            return {}

        symbols = list(set(factor_values.keys()) & set(forward_returns.keys()))
        if len(symbols) < 30:
            return {}

        returns = np.array([forward_returns[s] for s in symbols])
        factor_names = list(next(iter(factor_values.values())).keys())

        results = {}
        for fname in factor_names:
            values = np.array([
                factor_values[s].get(fname, np.nan) for s in symbols
            ])

            valid = ~(np.isnan(values) | np.isnan(returns))
            if np.sum(valid) < 30:
                continue

            v = values[valid]
            r = returns[valid]

            # Pearson IC
            ic = float(np.corrcoef(v, r)[0, 1])

            # Rank IC
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
            }

            # 更新因子元信息
            if fname in self._factors:
                self._factors[fname].ic_history.append(ic)
                self._factors[fname].rank_ic_history.append(float(rank_ic))

        return results

    def analyze_daily_ic(self):
        """分析最近一天的IC (由main.py盘后调用)"""
        dates = sorted(self._daily_values.keys())
        if not dates:
            logger.info("无因子数据, 跳过IC分析")
            return

        latest_date = dates[-1]
        ic_results = self.compute_daily_ic(latest_date)

        if ic_results:
            logger.info(f"因子IC分析 ({latest_date}):")
            sorted_factors = sorted(
                ic_results.items(),
                key=lambda x: abs(x[1]["ic"]),
                reverse=True,
            )
            for fname, metrics in sorted_factors[:10]:
                logger.info(
                    f"  {fname:30s} IC={metrics['ic']:+.4f} "
                    f"RankIC={metrics['rank_ic']:+.4f} "
                    f"t={metrics['t_stat']:+.2f}"
                )

            # 保存
            report_path = self._report_dir / f"ic_{latest_date}.json"
            with open(report_path, "w") as f:
                json.dump(ic_results, f, indent=2)

        # 自动淘汰低质量因子
        self._auto_disable_factors()

    # ==================== IC汇总 ====================

    def get_ic_summary(self) -> Dict[str, dict]:
        """获取所有因子的IC汇总统计"""
        summary = {}

        for fname, meta in self._factors.items():
            if not meta.ic_history:
                continue

            ics = np.array(meta.ic_history)
            rank_ics = np.array(meta.rank_ic_history) if meta.rank_ic_history else ics

            ic_mean = float(np.mean(ics))
            ic_std = float(np.std(ics))
            ir = ic_mean / (ic_std + 1e-8)  # Information Ratio
            ic_positive_pct = float(np.mean(ics > 0))

            summary[fname] = {
                "category": meta.category,
                "enabled": meta.enabled,
                "ic_mean": round(ic_mean, 4),
                "ic_std": round(ic_std, 4),
                "ir": round(ir, 4),
                "rank_ic_mean": round(float(np.mean(rank_ics)), 4),
                "ic_positive_pct": round(ic_positive_pct, 4),
                "n_days": len(ics),
            }

        # 按IR排序
        summary = dict(
            sorted(summary.items(), key=lambda x: abs(x[1]["ir"]), reverse=True)
        )

        self._ic_summary = summary
        return summary

    # ==================== 因子相关性 ====================

    def compute_factor_correlation(self, trade_date: str) -> Optional[pd.DataFrame]:
        """计算因子间相关性矩阵"""
        factor_values = self._daily_values.get(trade_date, {})
        if not factor_values:
            return None

        symbols = list(factor_values.keys())
        factor_names = list(next(iter(factor_values.values())).keys())

        data = {}
        for fname in factor_names:
            data[fname] = [
                factor_values[s].get(fname, np.nan) for s in symbols
            ]

        df = pd.DataFrame(data)
        corr = df.corr()
        return corr

    def get_redundant_factors(self, trade_date: str) -> List[Tuple[str, str, float]]:
        """找出高度相关的冗余因子对"""
        corr = self.compute_factor_correlation(trade_date)
        if corr is None:
            return []

        redundant = []
        factors = corr.columns.tolist()
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                c = abs(corr.iloc[i, j])
                if c > self.max_correlation:
                    redundant.append((factors[i], factors[j], round(c, 4)))

        redundant.sort(key=lambda x: x[2], reverse=True)
        return redundant

    # ==================== 自动淘汰 ====================

    def _auto_disable_factors(self):
        """自动禁用低质量因子"""
        summary = self.get_ic_summary()
        disabled_count = 0

        for fname, metrics in summary.items():
            if not self._factors[fname].enabled:
                continue

            # 条件: IC均值过低 + 足够多的观察天数
            if (metrics["n_days"] >= 20 and
                abs(metrics["ic_mean"]) < self.min_ic_threshold and
                abs(metrics["ir"]) < 0.3):

                self._factors[fname].enabled = False
                disabled_count += 1
                logger.warning(
                    f"因子自动淘汰: {fname} "
                    f"(IC={metrics['ic_mean']:.4f}, IR={metrics['ir']:.4f})"
                )

        if disabled_count > 0:
            logger.info(f"本轮淘汰 {disabled_count} 个低质量因子")

    # ==================== 报告 ====================

    def get_factor_report(self) -> dict:
        """获取因子管理报告"""
        summary = self.get_ic_summary()
        total = len(self._factors)
        enabled = sum(1 for m in self._factors.values() if m.enabled)

        categories = defaultdict(int)
        for m in self._factors.values():
            categories[m.category] += 1

        return {
            "total_factors": total,
            "enabled_factors": enabled,
            "disabled_factors": total - enabled,
            "categories": dict(categories),
            "top10_by_ir": [
                {"name": k, **v}
                for k, v in list(summary.items())[:10]
            ],
            "bottom5_by_ir": [
                {"name": k, **v}
                for k, v in list(summary.items())[-5:]
            ] if len(summary) >= 5 else [],
        }
