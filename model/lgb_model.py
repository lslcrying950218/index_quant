"""LightGBM 模型封装"""
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class LGBModel:
    """LightGBM 预测模型"""

    def __init__(self, config: dict):
        self.config = config
        self.model: Optional[lgb.Booster] = None

    def load(self, path: str):
        if not LGB_AVAILABLE:
            logger.error("lightgbm 未安装")
            return
        try:
            self.model = lgb.Booster(model_file=path)
            logger.info(f"LightGBM模型加载: {path}")
        except Exception as e:
            logger.error(f"LightGBM加载失败: {e}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测
        features: (n_stocks, n_features)
        returns: (n_stocks,) 预测收益
        """
        if self.model is None:
            return np.zeros(features.shape[0])

        try:
            features = np.nan_to_num(features, nan=0.0)
            preds = self.model.predict(features)
            return preds
        except Exception as e:
            logger.error(f"LightGBM预测异常: {e}")
            return np.zeros(features.shape[0])

    def save(self, path: str):
        if self.model:
            self.model.save_model(path)
