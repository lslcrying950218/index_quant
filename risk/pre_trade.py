"""
盘前风控
- 股票池黑白名单
- 涨跌停检测
- 停牌检测
- 基本面风险检查
"""
import logging
from typing import List, Set, Tuple
from datetime import date

logger = logging.getLogger(__name__)


class PreTradeRiskCheck:
    """盘前风控检查"""

    def __init__(self, config: dict):
        self.config = config
        self.blacklist: Set[str] = set(config.get("banned_stocks", []))
        self.st_stocks: Set[str] = set()
        self.suspended: Set[str] = set()
        self.limit_up: Set[str] = set()
        self.limit_down: Set[str] = set()

    def update_market_status(self, st_list: List[str] = None,
                              suspended_list: List[str] = None,
                              limit_up_list: List[str] = None,
                              limit_down_list: List[str] = None):
        """更新市场状态 (每日盘前调用)"""
        if st_list is not None:
            self.st_stocks = set(st_list)
        if suspended_list is not None:
            self.suspended = set(suspended_list)
        if limit_up_list is not None:
            self.limit_up = set(limit_up_list)
        if limit_down_list is not None:
            self.limit_down = set(limit_down_list)

        logger.info(
            f"盘前状态更新: ST={len(self.st_stocks)} "
            f"停牌={len(self.suspended)} "
            f"涨停={len(self.limit_up)} 跌停={len(self.limit_down)}"
        )

    def check(self, symbol: str, side: str) -> Tuple[bool, str]:
        """
        盘前检查

        Returns:
            (通过, 原因)
        """
        # 黑名单
        if symbol in self.blacklist:
            return False, f"{symbol} 在黑名单中"

        # ST
        if symbol in self.st_stocks:
            return False, f"{symbol} 为ST股"

        # 停牌
        if symbol in self.suspended:
            return False, f"{symbol} 停牌"

        # 涨停不能买
        if side == "buy" and symbol in self.limit_up:
            return False, f"{symbol} 涨停, 禁止买入"

        # 跌停不能卖
        if side == "sell" and symbol in self.limit_down:
            return False, f"{symbol} 跌停, 禁止卖出"

        return True, "OK"

    def filter_pool(self, symbols: List[str]) -> List[str]:
        """过滤股票池"""
        excluded = self.blacklist | self.st_stocks | self.suspended
        filtered = [s for s in symbols if s not in excluded]
        removed = len(symbols) - len(filtered)
        if removed > 0:
            logger.info(f"股票池过滤: 移除 {removed} 只 (黑名单/ST/停牌)")
        return filtered
