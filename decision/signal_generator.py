"""
信号生成器
- 融合模型预测 + 规则引擎 + NLP情感
- 生成最终交易信号
- 信号强度量化
"""
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """交易信号"""
    symbol: str
    action: str              # buy / sell / hold
    strength: float          # 信号强度 -1.0 ~ +1.0
    confidence: float        # 置信度 0 ~ 1.0
    target_weight: float     # 建议目标权重
    urgency: float           # 紧急度 0 ~ 1.0
    sources: Dict[str, float]  # 各来源贡献
    reasons: List[str]       # 理由列表
    timestamp: datetime
    ttl_seconds: int = 300   # 信号有效期


class SignalGenerator:
    """信号生成器"""

    def __init__(self, config: dict, rule_engine=None,
                 nlp_pipeline=None, redis_cache=None):
        self.config = config
        self.rule_engine = rule_engine
        self.nlp = nlp_pipeline
        self.redis = redis_cache

        # 信号融合权重
        self.weights = config.get("signal_weights", {
            "model": 0.50,       # 模型预测
            "rules": 0.25,       # 规则引擎
            "sentiment": 0.15,   # NLP情感
            "batch": 0.10,       # 盘前批处理
        })

        # 信号阈值
        self.buy_threshold = config.get("buy_threshold", 0.15)
        self.sell_threshold = config.get("sell_threshold", -0.15)
        self.min_confidence = config.get("min_confidence", 0.3)

    def generate(
        self,
        symbol: str,
        model_pred_return: float,
        model_pred_vol: float,
        realtime_factors: Dict[str, float],
        position_info: Optional[dict] = None,
    ) -> TradeSignal:
        """
        生成单只股票的交易信号

        Args:
            symbol: 股票代码
            model_pred_return: 模型预测收益
            model_pred_vol: 模型预测波动率
            realtime_factors: 实时因子值
            position_info: 当前持仓信息

        Returns:
            TradeSignal
        """
        sources = {}
        reasons = []

        # ========== 1. 模型信号 ==========
        model_score = self._model_signal(model_pred_return, model_pred_vol)
        sources["model"] = model_score

        if model_pred_return > 0.005:
            reasons.append(f"模型看多 (预测+{model_pred_return:.2%})")
        elif model_pred_return < -0.005:
            reasons.append(f"模型看空 (预测{model_pred_return:.2%})")

        # ========== 2. 规则引擎信号 ==========
        rule_score = 0.0
        if self.rule_engine:
            context = {
                **realtime_factors,
                "model_pred_return": model_pred_return,
                "model_pred_vol": model_pred_vol,
            }
            if position_info:
                context.update(position_info)

            # NLP情感注入context
            if self.nlp:
                sentiment = self.nlp.get_aggregate_sentiment(symbol, hours=12)
                context["news_sentiment_score"] = sentiment.get("avg_score", 0)
                context["news_event_type"] = "normal"  # 简化

            triggered_rules = self.rule_engine.evaluate(symbol, context)

            for rule in triggered_rules:
                action = rule["action"]
                weight = rule["weight"]

                if action == "buy":
                    rule_score += weight * 0.3
                    reasons.append(f"规则[{rule['name']}]: {rule['reason']}")
                elif action == "sell":
                    rule_score -= weight * 0.3
                    reasons.append(f"规则[{rule['name']}]: {rule['reason']}")
                elif action == "block":
                    # 禁止交易 -> 直接返回hold
                    return TradeSignal(
                        symbol=symbol,
                        action="hold",
                        strength=0.0,
                        confidence=1.0,
                        target_weight=0.0,
                        urgency=0.0,
                        sources={"block_rule": 1.0},
                        reasons=[f"规则禁止: {rule['reason']}"],
                        timestamp=datetime.now(),
                    )

            rule_score = max(-1.0, min(1.0, rule_score))
            sources["rules"] = rule_score

        # ========== 3. NLP情感信号 ==========
        sentiment_score = 0.0
        if self.nlp:
            sentiment = self.nlp.get_aggregate_sentiment(symbol, hours=24)
            sentiment_score = sentiment.get("avg_score", 0)
            momentum = sentiment.get("momentum", 0)

            # 情感动量加成
            sentiment_score = sentiment_score * 0.7 + momentum * 0.3
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            sources["sentiment"] = sentiment_score

            if abs(sentiment_score) > 0.3:
                label = "正面" if sentiment_score > 0 else "负面"
                reasons.append(
                    f"舆情{label} (分数={sentiment_score:.2f}, "
                    f"新闻数={sentiment.get('count', 0)})"
                )

        # ========== 4. 盘前批处理信号 ==========
        batch_score = 0.0
        if self.redis:
            batch_preds = self.redis.get_state("batch_predictions")
            if batch_preds and symbol in batch_preds:
                batch_pred = batch_preds[symbol]
                batch_score = max(-1.0, min(1.0, batch_pred * 20))  # 归一化
                sources["batch"] = batch_score

        # ========== 5. 信号融合 ==========
        final_score = (
            sources.get("model", 0) * self.weights["model"] +
            sources.get("rules", 0) * self.weights["rules"] +
            sources.get("sentiment", 0) * self.weights["sentiment"] +
            sources.get("batch", 0) * self.weights["batch"]
        )
        final_score = max(-1.0, min(1.0, final_score))

        # 置信度: 各信号一致性
        signal_values = [v for v in sources.values() if v != 0]
        if len(signal_values) >= 2:
            signs = [1 if v > 0 else -1 for v in signal_values]
            agreement = abs(sum(signs)) / len(signs)
            confidence = agreement * 0.7 + min(abs(final_score), 1.0) * 0.3
        else:
            confidence = min(abs(final_score) + 0.2, 1.0)

        # ========== 6. 确定动作 ==========
        if final_score > self.buy_threshold and confidence >= self.min_confidence:
            action = "buy"
            target_weight = min(final_score * 0.15, 0.10)  # 最大10%
        elif final_score < self.sell_threshold and confidence >= self.min_confidence:
            action = "sell"
            target_weight = 0.0
        else:
            action = "hold"
            target_weight = None  # 保持当前

        # 紧急度
        urgency = min(abs(final_score) * confidence, 1.0)

        signal = TradeSignal(
            symbol=symbol,
            action=action,
            strength=round(final_score, 4),
            confidence=round(confidence, 4),
            target_weight=target_weight if target_weight is not None else 0,
            urgency=round(urgency, 4),
            sources={k: round(v, 4) for k, v in sources.items()},
            reasons=reasons,
            timestamp=datetime.now(),
        )

        # 写入Redis
        if self.redis:
            self.redis.set_signal(symbol, {
                "action": signal.action,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "urgency": signal.urgency,
                "sources": signal.sources,
                "timestamp": signal.timestamp.isoformat(),
            })

        return signal

    def generate_batch(
        self,
        symbols: List[str],
        model_predictions: Dict[str, Tuple[float, float]],
        all_factors: Dict[str, Dict[str, float]],
        positions: Dict[str, dict],
    ) -> Dict[str, TradeSignal]:
        """批量生成信号"""
        signals = {}
        for sym in symbols:
            pred_ret, pred_vol = model_predictions.get(sym, (0, 0.02))
            factors = all_factors.get(sym, {})
            pos_info = positions.get(sym)

            signals[sym] = self.generate(
                symbol=sym,
                model_pred_return=pred_ret,
                model_pred_vol=pred_vol,
                realtime_factors=factors,
                position_info=pos_info,
            )
        return signals

    @staticmethod
    def _model_signal(pred_return: float, pred_vol: float) -> float:
        """模型预测转换为信号分数"""
        if pred_vol < 0.001:
            pred_vol = 0.02
        # 风险调整后的分数
        risk_adjusted = pred_return / pred_vol
        # 映射到 [-1, 1]
        score = np.tanh(risk_adjusted * 2)
        return float(score)
