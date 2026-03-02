"""
规则引擎
- YAML配置驱动
- 支持条件组合 (AND/OR/NOT)
- 优先级排序
- 规则热加载
- 与模型信号融合
"""
import yaml
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, time as dtime
import logging

logger = logging.getLogger(__name__)


class RuleAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    BLOCK = "block"        # 禁止交易
    REDUCE = "reduce"      # 减仓
    INCREASE = "increase"  # 加仓


@dataclass
class RuleCondition:
    """规则条件"""
    field: str              # 因子/指标名
    operator: str           # gt / lt / eq / gte / lte / between / in
    value: Any              # 阈值
    description: str = ""

    def evaluate(self, context: dict) -> bool:
        """评估条件"""
        actual = context.get(self.field)
        if actual is None:
            return False

        try:
            if self.operator == "gt":
                return float(actual) > float(self.value)
            elif self.operator == "lt":
                return float(actual) < float(self.value)
            elif self.operator == "gte":
                return float(actual) >= float(self.value)
            elif self.operator == "lte":
                return float(actual) <= float(self.value)
            elif self.operator == "eq":
                return actual == self.value
            elif self.operator == "neq":
                return actual != self.value
            elif self.operator == "between":
                low, high = self.value
                return float(low) <= float(actual) <= float(high)
            elif self.operator == "in":
                return actual in self.value
            elif self.operator == "not_in":
                return actual not in self.value
            else:
                logger.warning(f"未知操作符: {self.operator}")
                return False
        except (ValueError, TypeError) as e:
            logger.debug(f"条件评估异常: {self.field} {self.operator} {self.value}: {e}")
            return False


@dataclass
class Rule:
    """交易规则"""
    rule_id: str
    name: str
    description: str
    priority: int                       # 优先级 (越大越优先)
    action: RuleAction
    conditions: List[RuleCondition]     # 条件列表
    logic: str = "AND"                  # AND / OR
    enabled: bool = True
    cooldown_seconds: int = 0           # 冷却时间
    weight: float = 1.0                 # 信号权重
    valid_time: Tuple[str, str] = ("09:30", "15:00")  # 生效时段

    _last_triggered: Dict[str, float] = field(default_factory=dict)

    def evaluate(self, symbol: str, context: dict) -> Tuple[bool, str]:
        """
        评估规则

        Returns:
            (是否触发, 原因说明)
        """
        if not self.enabled:
            return False, "规则已禁用"

        # 时段检查
        now = datetime.now().time()
        start = dtime(*map(int, self.valid_time[0].split(":")))
        end = dtime(*map(int, self.valid_time[1].split(":")))
        if not (start <= now <= end):
            return False, "不在生效时段"

        # 冷却检查
        if self.cooldown_seconds > 0:
            last = self._last_triggered.get(symbol, 0)
            if time.time() - last < self.cooldown_seconds:
                return False, "冷却中"

        # 条件评估
        results = [cond.evaluate(context) for cond in self.conditions]

        if self.logic == "AND":
            triggered = all(results)
        elif self.logic == "OR":
            triggered = any(results)
        else:
            triggered = all(results)

        if triggered:
            self._last_triggered[symbol] = time.time()
            reasons = [
                cond.description or f"{cond.field} {cond.operator} {cond.value}"
                for cond, result in zip(self.conditions, results) if result
            ]
            return True, "; ".join(reasons)

        return False, ""


class RuleEngine:
    """规则引擎"""

    def __init__(self, config: dict):
        self.config = config
        self.rules_dir = Path(config.get("rules_dir", "config/rules"))
        self._rules: List[Rule] = []
        self._lock = threading.Lock()

        # 加载规则
        self._load_rules()

        # 启动热加载监控
        self._watch_thread = threading.Thread(
            target=self._watch_rules, daemon=True, name="rule_watcher"
        )
        self._watch_thread.start()

    def _load_rules(self):
        """从YAML加载规则"""
        rules = []

        for yaml_file in self.rules_dir.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                for rule_data in data.get("rules", []):
                    conditions = [
                        RuleCondition(
                            field=c["field"],
                            operator=c["operator"],
                            value=c["value"],
                            description=c.get("description", ""),
                        )
                        for c in rule_data.get("conditions", [])
                    ]

                    rule = Rule(
                        rule_id=rule_data["id"],
                        name=rule_data["name"],
                        description=rule_data.get("description", ""),
                        priority=rule_data.get("priority", 50),
                        action=RuleAction(rule_data["action"]),
                        conditions=conditions,
                        logic=rule_data.get("logic", "AND"),
                        enabled=rule_data.get("enabled", True),
                        cooldown_seconds=rule_data.get("cooldown", 0),
                        weight=rule_data.get("weight", 1.0),
                        valid_time=tuple(rule_data.get("valid_time", ["09:30", "15:00"])),
                    )
                    rules.append(rule)

                logger.info(f"加载规则文件: {yaml_file.name} ({len(data.get('rules', []))} 条)")

            except Exception as e:
                logger.error(f"规则文件加载失败 {yaml_file}: {e}")

        # 按优先级排序
        rules.sort(key=lambda r: r.priority, reverse=True)

        with self._lock:
            self._rules = rules

        logger.info(f"✅ 规则引擎加载完成: {len(rules)} 条规则")

    def _watch_rules(self):
        """监控规则文件变化, 自动热加载"""
        last_mtime = {}
        while True:
            try:
                changed = False
                for yaml_file in self.rules_dir.glob("*.yaml"):
                    mtime = yaml_file.stat().st_mtime
                    if yaml_file.name not in last_mtime or last_mtime[yaml_file.name] != mtime:
                        last_mtime[yaml_file.name] = mtime
                        changed = True

                if changed:
                    logger.info("🔄 检测到规则文件变更, 重新加载...")
                    self._load_rules()

            except Exception as e:
                logger.error(f"规则监控异常: {e}")

            time.sleep(10)

    # ==================== 规则评估 ====================

    def evaluate(self, symbol: str, context: dict) -> List[dict]:
        """
        评估所有规则

        Args:
            symbol: 股票代码
            context: 上下文 (包含因子值、持仓信息等)

        Returns:
            触发的规则列表 [{
                "rule_id": ...,
                "action": ...,
                "weight": ...,
                "reason": ...,
            }]
        """
        triggered = []

        with self._lock:
            rules = list(self._rules)

        for rule in rules:
            try:
                hit, reason = rule.evaluate(symbol, context)
                if hit:
                    triggered.append({
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "action": rule.action.value,
                        "weight": rule.weight,
                        "priority": rule.priority,
                        "reason": reason,
                    })
            except Exception as e:
                logger.error(f"规则评估异常 {rule.rule_id}: {e}")

        return triggered

    def evaluate_batch(self, symbols: List[str],
                        contexts: Dict[str, dict]) -> Dict[str, List[dict]]:
        """批量评估"""
        results = {}
        for sym in symbols:
            ctx = contexts.get(sym, {})
            results[sym] = self.evaluate(sym, ctx)
        return results

    # ==================== 规则管理 ====================

    def enable_rule(self, rule_id: str):
        with self._lock:
            for rule in self._rules:
                if rule.rule_id == rule_id:
                    rule.enabled = True
                    logger.info(f"规则已启用: {rule_id}")
                    return

    def disable_rule(self, rule_id: str):
        with self._lock:
            for rule in self._rules:
                if rule.rule_id == rule_id:
                    rule.enabled = False
                    logger.info(f"规则已禁用: {rule_id}")
                    return

    def get_rules_summary(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "id": r.rule_id,
                    "name": r.name,
                    "action": r.action.value,
                    "priority": r.priority,
                    "enabled": r.enabled,
                }
                for r in self._rules
            ]
