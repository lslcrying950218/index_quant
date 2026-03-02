"""
NLP Pipeline - 新闻情感分析
- 基于预训练模型的金融文本情感分析
- 关键事件检测 (利好/利空/中性)
- 实体识别 (公司/行业/概念)
- GPU加速推理
"""
import re
import time
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        pipeline as hf_pipeline,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("transformers 未安装, NLP功能降级为规则模式")


@dataclass
class SentimentResult:
    """情感分析结果"""
    symbol: str
    score: float              # -1.0 (极度利空) ~ +1.0 (极度利好)
    label: str                # positive / negative / neutral
    confidence: float         # 置信度 0~1
    keywords: List[str]       # 关键词
    event_type: str           # normal / major_positive / major_negative / warning
    source_count: int         # 来源新闻数量
    timestamp: datetime


class NLPPipeline:
    """NLP 处理管线"""

    # 金融关键词词典
    POSITIVE_KEYWORDS = {
        "利好", "大涨", "突破", "新高", "增持", "回购", "超预期",
        "业绩大增", "中标", "战略合作", "获批", "订单", "产能扩张",
        "分红", "送转", "净利润增长", "营收增长", "毛利率提升",
        "机构调研", "北向资金", "主力净流入", "涨停",
    }

    NEGATIVE_KEYWORDS = {
        "利空", "大跌", "暴跌", "减持", "质押", "违规", "处罚",
        "亏损", "下滑", "退市", "ST", "诉讼", "调查", "警示",
        "业绩下降", "商誉减值", "计提", "债务", "逾期",
        "跌停", "闪崩", "爆雷", "造假",
    }

    WARNING_KEYWORDS = {
        "退市", "ST", "立案调查", "重大违法", "暂停上市",
        "财务造假", "实控人被捕", "债务违约",
    }

    def __init__(self, config: dict, redis_cache=None):
        self.config = config
        self.redis = redis_cache
        self.device = config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu") if HF_AVAILABLE else "cpu"

        # 情感分数缓存: {symbol: deque of (timestamp, score)}
        self._sentiment_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=200)
        )
        self._lock = threading.Lock()

        # 加载模型
        self._model_loaded = False
        if HF_AVAILABLE and config.get("use_model", True):
            self._load_model()

    def _load_model(self):
        """加载预训练情感分析模型"""
        model_name = self.config.get(
            "model_name",
            "uer/roberta-base-finetuned-jd-binary-chinese"
            # 备选: "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
        )
        try:
            logger.info(f"加载NLP模型: {model_name} -> {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            ).to(self.device)
            self.model.eval()
            self._model_loaded = True
            logger.info("✅ NLP模型加载完成")
        except Exception as e:
            logger.error(f"NLP模型加载失败: {e}, 降级为规则模式")
            self._model_loaded = False

    # ==================== 核心分析方法 ====================

    def analyze(self, text: str, symbol: str = "") -> SentimentResult:
        """
        分析单条文本的情感

        Returns:
            SentimentResult
        """
        # 文本预处理
        text = self._preprocess(text)

        if not text:
            return self._neutral_result(symbol)

        # 关键词检测
        keywords, keyword_score = self._keyword_analysis(text)
        event_type = self._detect_event_type(text, keywords)

        # 模型推理
        if self._model_loaded:
            model_score, confidence = self._model_inference(text)
        else:
            model_score = keyword_score
            confidence = min(abs(keyword_score) + 0.3, 0.9)

        # 融合: 模型分数 * 0.6 + 关键词分数 * 0.4
        if self._model_loaded:
            final_score = model_score * 0.6 + keyword_score * 0.4
        else:
            final_score = keyword_score

        final_score = max(-1.0, min(1.0, final_score))

        # 标签
        if final_score > 0.15:
            label = "positive"
        elif final_score < -0.15:
            label = "negative"
        else:
            label = "neutral"

        result = SentimentResult(
            symbol=symbol,
            score=round(final_score, 4),
            label=label,
            confidence=round(confidence, 4),
            keywords=keywords,
            event_type=event_type,
            source_count=1,
            timestamp=datetime.now(),
        )

        # 写入历史
        if symbol:
            with self._lock:
                self._sentiment_history[symbol].append(
                    (datetime.now(), final_score)
                )

            # 写入Redis
            if self.redis:
                self.redis.set_news_sentiment(symbol, {
                    "score": result.score,
                    "label": result.label,
                    "event_type": result.event_type,
                    "keywords": result.keywords[:5],
                    "timestamp": result.timestamp.isoformat(),
                })

        return result

    def analyze_batch(self, texts: List[str],
                      symbols: List[str] = None) -> List[SentimentResult]:
        """批量分析"""
        if symbols is None:
            symbols = [""] * len(texts)

        results = []
        for text, sym in zip(texts, symbols):
            results.append(self.analyze(text, sym))
        return results

    def get_aggregate_sentiment(self, symbol: str,
                                 hours: int = 24) -> Dict[str, float]:
        """
        获取聚合情感分数

        Returns:
            {
                "current_score": 最新分数,
                "avg_score": 平均分数,
                "momentum": 情感动量 (趋势),
                "count": 新闻数量,
                "volatility": 情感波动率,
            }
        """
        with self._lock:
            history = list(self._sentiment_history.get(symbol, []))

        if not history:
            return {
                "current_score": 0.0,
                "avg_score": 0.0,
                "momentum": 0.0,
                "count": 0,
                "volatility": 0.0,
            }

        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [(t, s) for t, s in history if t > cutoff]

        if not recent:
            return {
                "current_score": history[-1][1],
                "avg_score": 0.0,
                "momentum": 0.0,
                "count": 0,
                "volatility": 0.0,
            }

        scores = [s for _, s in recent]
        n = len(scores)

        # 时间衰减加权平均 (最近的权重更大)
        weights = np.exp(np.linspace(-1, 0, n))
        weights /= weights.sum()
        weighted_avg = float(np.average(scores, weights=weights))

        # 动量: 后半段均值 - 前半段均值
        mid = n // 2
        if mid > 0:
            momentum = np.mean(scores[mid:]) - np.mean(scores[:mid])
        else:
            momentum = 0.0

        return {
            "current_score": round(scores[-1], 4),
            "avg_score": round(weighted_avg, 4),
            "momentum": round(float(momentum), 4),
            "count": n,
            "volatility": round(float(np.std(scores)), 4),
        }

    # ==================== 内部方法 ====================

    @staticmethod
    def _preprocess(text: str) -> str:
        """文本预处理"""
        if not text:
            return ""
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 去除URL
        text = re.sub(r'https?://\S+', '', text)
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        # 截断
        return text[:512]

    def _keyword_analysis(self, text: str) -> Tuple[List[str], float]:
        """关键词情感分析"""
        found_pos = [kw for kw in self.POSITIVE_KEYWORDS if kw in text]
        found_neg = [kw for kw in self.NEGATIVE_KEYWORDS if kw in text]

        pos_count = len(found_pos)
        neg_count = len(found_neg)
        total = pos_count + neg_count

        if total == 0:
            return [], 0.0

        score = (pos_count - neg_count) / total
        keywords = found_pos + found_neg
        return keywords, score

    def _detect_event_type(self, text: str, keywords: List[str]) -> str:
        """检测事件类型"""
        # 重大警告事件
        for kw in self.WARNING_KEYWORDS:
            if kw in text:
                return "warning"

        pos_kw = [kw for kw in keywords if kw in self.POSITIVE_KEYWORDS]
        neg_kw = [kw for kw in keywords if kw in self.NEGATIVE_KEYWORDS]

        if len(pos_kw) >= 3:
            return "major_positive"
        if len(neg_kw) >= 3:
            return "major_negative"

        return "normal"

    def _model_inference(self, text: str) -> Tuple[float, float]:
        """模型推理"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            # 假设: label 0=negative, 1=positive
            prob_neg = probs[0][0].item()
            prob_pos = probs[0][1].item() if probs.shape[1] > 1 else 1 - prob_neg

            score = prob_pos - prob_neg  # -1 ~ +1
            confidence = max(prob_pos, prob_neg)

            return score, confidence

        except Exception as e:
            logger.error(f"模型推理异常: {e}")
            return 0.0, 0.5

    @staticmethod
    def _neutral_result(symbol: str) -> SentimentResult:
        return SentimentResult(
            symbol=symbol,
            score=0.0,
            label="neutral",
            confidence=0.5,
            keywords=[],
            event_type="normal",
            source_count=0,
            timestamp=datetime.now(),
        )
