"""因子基类"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseFactor(ABC):
    """因子基类"""

    def __init__(self, name: str, category: str, description: str = ""):
        self.name = name
        self.category = category
        self.description = description

    @abstractmethod
    def compute(self, data: Dict[str, Any]) -> float:
        """计算因子值"""
        raise NotImplementedError

    def __repr__(self):
        return f"Factor({self.name}, {self.category})"
