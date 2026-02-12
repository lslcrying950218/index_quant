"""
组合优化器
- 基于模型预测 + 风控约束生成目标持仓
- 考虑交易成本的优化
"""
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TargetPosition:
    symbol: str
    target_weight: float    # 目标权重
    target_volume: int      # 目标股数
    target_value: float     # 目标市值
    current_volume: int     # 当前股数
    delta_volume: int       # 需调整股数
    side: str               # buy / sell / hold
    priority: float         # 调仓优先级


class PortfolioOptimizer:
    """组合优化器"""

    def __init__(self, config: dict):
        self.total_capital = config['total_capital']
        self.max_position_pct = config['max_position_pct']
        self.max_total_position_pct = config['max_total_position_pct']
        self.stock_count_range = config['stock_count_range']
        self.transaction_cost = 0.0015  # 单边交易成本 (佣金+印花税+滑点)

    def optimize(
        self,
        pred_returns: Dict[str, float],
        pred_vols: Dict[str, float],
        current_prices: Dict[str, float],
        current_positions: Dict[str, int],
    ) -> List[TargetPosition]:
        """
        生成目标持仓

        Args:
            pred_returns: {symbol: 预期收益}
            pred_vols: {symbol: 预期波动率}
            current_prices: {symbol: 当前价格}
            current_positions: {symbol: 当前持仓股数}

        Returns:
            目标持仓列表 (按调仓优先级排序)
        """
        symbols = list(pred_returns.keys())
        n = len(symbols)

        if n == 0:
            return []

        # ---- Step 1: 计算风险调整后的alpha ----
        alphas = {}
        for sym in symbols:
            ret = pred_returns[sym]
            vol = max(pred_vols.get(sym, 0.02), 0.005)  # 最低波动率

            # 扣除交易成本
            is_new = sym not in current_positions or current_positions[sym] == 0
            cost = self.transaction_cost * 2 if is_new else self.transaction_cost

            risk_adjusted = (ret - cost) / vol  # 类Sharpe
            alphas[sym] = risk_adjusted

        # ---- Step 2: 选股 (Top N) ----
        min_stocks, max_stocks = self.stock_count_range
        sorted_symbols = sorted(alphas.keys(), key=lambda s: alphas[s], reverse=True)

        # 只选alpha > 0的
        selected = [s for s in sorted_symbols if alphas[s] > 0][:max_stocks]

        # 至少保留min_stocks只 (如果有持仓的话)
        if len(selected) < min_stocks:
            selected = sorted_symbols[:min_stocks]

        # ---- Step 3: 权重分配 ----
        if not selected:
            # 全部清仓
            return self._generate_clear_all(current_positions, current_prices)

        # 基于alpha的权重
        raw_weights = {}
        for sym in selected:
            raw_weights[sym] = max(alphas[sym], 0.001)

        total_weight = sum(raw_weights.values())
        weights = {s: w / total_weight for s, w in raw_weights.items()}

        # ---- Step 4: 约束调整 ----
        # 单票上限
        for sym in weights:
            weights[sym] = min(weights[sym], self.max_position_pct)

        # 重新归一化
        total_w = sum(weights.values())
        target_total_pct = min(
            self.max_total_position_pct,
            total_w  # 不强制满仓
        )
        weights = {s: w / total_w * target_total_pct for s, w in weights.items()}

        # ---- Step 5: 转换为目标股数 ----
        targets = []
        for sym in symbols:
            target_weight = weights.get(sym, 0.0)
            target_value = self.total_capital * target_weight
            price = current_prices.get(sym, 0)
            current_vol = current_positions.get(sym, 0)

            if price <= 0:
                continue

            # 取整到100股
            target_vol = int(target_value / price / 100) * 100
            delta = target_vol - current_vol

            if abs(delta) < 100:
                side = "hold"
                delta = 0
            elif delta > 0:
                side = "buy"
            else:
                side = "sell"

            # 调仓优先级: 卖出优先于买入 (先腾资金)
            if side == "sell":
                priority = 100 + abs(delta) * price  # 卖出优先
            elif side == "buy":
                priority = alphas.get(sym, 0)  # alpha高的优先买
            else:
                priority = -1

            targets.append(TargetPosition(
                symbol=sym,
                target_weight=target_weight,
                target_volume=target_vol,
                target_value=target_value,
                current_volume=current_vol,
                delta_volume=delta,
                side=side,
                priority=priority,
            ))

        # 按优先级排序: 先卖后买
        targets.sort(key=lambda t: t.priority, reverse=True)

        logger.info(
            f"组合优化完成: 选股{len(selected)}只, "
            f"买入{sum(1 for t in targets if t.side == 'buy')}只, "
            f"卖出{sum(1 for t in targets if t.side == 'sell')}只, "
            f"持有{sum(1 for t in targets if t.side == 'hold')}只"
        )

        return targets

    def _generate_clear_all(self, current_positions, current_prices):
        """生成全部清仓指令"""
        targets = []
        for sym, vol in current_positions.items():
            if vol > 0:
                targets.append(TargetPosition(
                    symbol=sym,
                    target_weight=0,
                    target_volume=0,
                    target_value=0,
                    current_volume=vol,
                    delta_volume=-vol,
                    side="sell",
                    priority=1000,
                ))
        return targets
