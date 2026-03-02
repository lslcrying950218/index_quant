"""模型集成"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EnsembleModel:
    """模型集成器"""

    def __init__(self, config: dict):
        self.config = config
        self.method = config.get("method", "adaptive_weight")

        # 默认权重
        self.weights = {
            "transformer": 0.6,
            "lightgbm": 0.4,
        }

        # 自适应权重的历史IC
        self._ic_history: Dict[str, List[float]] = {}

    def combine(
        self,
        predictions: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
        symbols: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        融合多个模型的预测

        Args:
            predictions: {
                "model_name": (pred_returns, pred_vols_or_None),
                ...
            }
            symbols: 股票列表

        Returns:
            (pred_returns_dict, pred_vols_dict)
        """
        n = len(symbols)
        combined_returns = np.zeros(n)
        combined_vols = np.ones(n) * 0.02
        total_weight = 0

        vol_count = 0
        vol_sum = np.zeros(n)

        for model_name, (returns, vols) in predictions.items():
            if returns is None:
                continue

            weight = self.weights.get(model_name, 0.5)

            # 确保长度一致
            if len(returns) != n:
                logger.warning(
                    f"模型 {model_name} 输出长度不匹配: {len(returns)} vs {n}"
                )
                continue

            returns = np.nan_to_num(returns, nan=0.0)
            combined_returns += returns * weight
            total_weight += weight

            if vols is not None:
                vols = np.nan_to_num(vols, nan=0.02)
                vols = np.clip(vols, 0.005, 0.5)
                vol_sum += vols
                vol_count += 1

        if total_weight > 0:
            combined_returns /= total_weight

        if vol_count > 0:
            combined_vols = vol_sum / vol_count

        # 转为字典
        ret_dict = {sym: float(combined_returns[i]) for i, sym in enumerate(symbols)}
        vol_dict = {sym: float(combined_vols[i]) for i, sym in enumerate(symbols)}

        return ret_dict, vol_dict

    def update_weights(self, model_ics: Dict[str, float]):
        """
        根据最近IC表现自适应更新权重
        model_ics: {"transformer": 0.05, "lightgbm": 0.03}
        """
        if not model_ics:
            return

        # IC绝对值作为权重
        abs_ics = {k: max(abs(v), 0.001) for k, v in model_ics.items()}
        total = sum(abs_ics.values())

        for model_name in abs_ics:
            self.weights[model_name] = abs_ics[model_name] / total

        logger.info(f"模型权重更新: {self.weights}")
