"""
Grafana 仪表盘自动配置
- 生成 Grafana Dashboard JSON
- 自动导入
"""
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GrafanaDashboardGenerator:
    """Grafana 仪表盘生成器"""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path("config/grafana")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_main_dashboard(self) -> dict:
        """生成主仪表盘JSON"""
        dashboard = {
            "dashboard": {
                "id": None,
                "uid": "quant-main",
                "title": "量化交易系统 - 实时监控",
                "tags": ["quant", "trading"],
                "timezone": "browser",
                "refresh": "5s",
                "time": {"from": "now-6h", "to": "now"},
                "panels": [
                    # ---- Row 1: 核心指标 ----
                    self._stat_panel(
                        title="总资产", target="quant_total_asset",
                        grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
                        unit="currencyCNY",
                    ),
                    self._stat_panel(
                        title="日盈亏", target="quant_daily_pnl",
                        grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
                        unit="currencyCNY",
                        thresholds=[
                            {"color": "red", "value": None},
                            {"color": "green", "value": 0},
                        ],
                    ),
                    self._stat_panel(
                        title="日收益率", target="quant_daily_pnl_pct",
                        grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
                        unit="percent",
                        thresholds=[
                            {"color": "red", "value": None},
                            {"color": "green", "value": 0},
                        ],
                    ),
                    self._stat_panel(
                        title="最大回撤", target="quant_max_drawdown_pct",
                        grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
                        unit="percent",
                        thresholds=[
                            {"color": "green", "value": None},
                            {"color": "orange", "value": -3},
                            {"color": "red", "value": -5},
                        ],
                    ),
                    self._stat_panel(
                        title="持仓数", target="quant_position_count",
                        grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
                    ),
                    self._stat_panel(
                        title="风控等级", target="quant_risk_level",
                        grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
                        thresholds=[
                            {"color": "green", "value": None},
                            {"color": "orange", "value": 1},
                            {"color": "red", "value": 2},
                            {"color": "dark-red", "value": 3},
                        ],
                        mappings=[
                            {"type": "value", "options": {
                                "0": {"text": "正常", "color": "green"},
                                "1": {"text": "警告", "color": "orange"},
                                "2": {"text": "严重", "color": "red"},
                                "3": {"text": "熔断", "color": "dark-red"},
                            }},
                        ],
                    ),

                    # ---- Row 2: 净值曲线 & 仓位 ----
                    self._timeseries_panel(
                        title="净值曲线",
                        targets=["quant_nav"],
                        grid_pos={"x": 0, "y": 4, "w": 16, "h": 8},
                    ),
                    self._gauge_panel(
                        title="仓位比例",
                        target="quant_position_pct",
                        grid_pos={"x": 16, "y": 4, "w": 8, "h": 8},
                        max_val=100,
                        thresholds=[
                            {"color": "green", "value": 0},
                            {"color": "orange", "value": 60},
                            {"color": "red", "value": 80},
                        ],
                    ),

                    # ---- Row 3: 延迟 & 吞吐 ----
                    self._timeseries_panel(
                        title="策略周期延迟 (ms)",
                        targets=["quant_strategy_cycle_latency_ms"],
                        grid_pos={"x": 0, "y": 12, "w": 12, "h": 6},
                    ),
                    self._timeseries_panel(
                        title="模型推理延迟 (ms)",
                        targets=["quant_model_inference_latency_ms"],
                        grid_pos={"x": 12, "y": 12, "w": 12, "h": 6},
                    ),

                    # ---- Row 4: 交易统计 ----
                    self._timeseries_panel(
                        title="订单提交/成交",
                        targets=[
                            "quant_orders_submitted_total",
                            "quant_orders_filled_total",
                        ],
                        grid_pos={"x": 0, "y": 18, "w": 12, "h": 6},
                    ),
                    self._timeseries_panel(
                        title="滑点分布 (bps)",
                        targets=["quant_slippage_bps"],
                        grid_pos={"x": 12, "y": 18, "w": 12, "h": 6},
                    ),
                ],
            },
            "overwrite": True,
        }

        # 保存
        output_path = self.output_dir / "main_dashboard.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False)

        logger.info(f"Grafana仪表盘已生成: {output_path}")
        return dashboard

    # ==================== 面板模板 ====================

    @staticmethod
    def _stat_panel(title: str, target: str, grid_pos: dict,
                    unit: str = "short", thresholds: list = None,
                    mappings: list = None) -> dict:
        panel = {
            "type": "stat",
            "title": title,
            "gridPos": grid_pos,
            "targets": [{"expr": target, "legendFormat": title}],
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": thresholds or [
                            {"color": "green", "value": None}
                        ],
                    },
                    "mappings": mappings or [],
                },
            },
        }
        return panel

    @staticmethod
    def _timeseries_panel(title: str, targets: list, grid_pos: dict) -> dict:
        return {
            "type": "timeseries",
            "title": title,
            "gridPos": grid_pos,
            "targets": [
                {"expr": t, "legendFormat": t.replace("quant_", "")}
                for t in targets
            ],
            "fieldConfig": {"defaults": {"custom": {"lineWidth": 2}}},
        }

    @staticmethod
    def _gauge_panel(title: str, target: str, grid_pos: dict,
                     max_val: float = 100, thresholds: list = None) -> dict:
        return {
            "type": "gauge",
            "title": title,
            "gridPos": grid_pos,
            "targets": [{"expr": target}],
            "fieldConfig": {
                "defaults": {
                    "max": max_val,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": thresholds or [
                            {"color": "green", "value": 0}
                        ],
                    },
                },
            },
        }
