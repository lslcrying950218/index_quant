"""
每日绩效报告自动生成
- HTML格式报告
- 自动发送邮件/推送
- 包含净值曲线、持仓明细、交易记录、风控日志
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """每日绩效报告生成器"""

    def __init__(self, config: dict):
        self.config = config
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)

    def generate_daily_report(
        self,
        performance: dict,
        positions: Dict[str, dict],
        trades: List[dict],
        risk_report: dict,
        factor_ic: Dict[str, dict] = None,
        slippage_report: dict = None,
    ) -> str:
        """
        生成HTML日报

        Returns:
            HTML文件路径
        """
        today = date.today().isoformat()

        # 持仓表格
        pos_rows = ""
        for sym, pos in sorted(positions.items()):
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = pos.get("unrealized_pnl_pct", 0)
            color = "#22c55e" if pnl >= 0 else "#ef4444"
            pos_rows += f"""
            <tr>
                <td>{sym}</td>
                <td>{pos.get('volume', 0):,}</td>
                <td>{pos.get('avg_cost', 0):.2f}</td>
                <td>{pos.get('current_price', 0):.2f}</td>
                <td>{pos.get('market_value', 0):,.0f}</td>
                <td style="color:{color}">{pnl:+,.0f}</td>
                <td style="color:{color}">{pnl_pct:+.2%}</td>
            </tr>"""

        # 交易表格
        trade_rows = ""
        for t in trades[-50:]:  # 最近50条
            trade_rows += f"""
            <tr>
                <td>{t.get('timestamp', '')[:19]}</td>
                <td>{t.get('symbol', '')}</td>
                <td style="color:{'#22c55e' if t.get('side')=='buy' else '#ef4444'}">{t.get('side', '').upper()}</td>
                <td>{t.get('volume', 0):,}</td>
                <td>{t.get('price', 0):.2f}</td>
                <td>{t.get('amount', 0):,.0f}</td>
                <td>{t.get('commission', 0):.2f}</td>
            </tr>"""

        # 因子IC表格
        ic_rows = ""
        if factor_ic:
            for fname, metrics in list(factor_ic.items())[:20]:
                ic_val = metrics.get("ic", 0)
                color = "#22c55e" if abs(ic_val) > 0.03 else "#888"
                ic_rows += f"""
                <tr>
                    <td>{fname}</td>
                    <td style="color:{color}">{ic_val:.4f}</td>
                    <td>{metrics.get('rank_ic', 0):.4f}</td>
                    <td>{metrics.get('t_stat', 0):.2f}</td>
                    <td>{metrics.get('n_samples', 0)}</td>
                </tr>"""

        # 风控信息
        risk_level = risk_report.get("level", "normal")
        risk_color_map = {
            "normal": "#22c55e", "warning": "#f59e0b",
            "critical": "#ef4444", "halt": "#7f1d1d"
        }
        risk_color = risk_color_map.get(risk_level, "#888")

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>量化交易日报 {today}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; color: #f8fafc; }}
    h2 {{ font-size: 18px; margin: 24px 0 12px; color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
    .subtitle {{ color: #64748b; margin-bottom: 24px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .kpi {{ background: #1e293b; border-radius: 12px; padding: 16px; text-align: center; border: 1px solid #334155; }}
    .kpi .value {{ font-size: 24px; font-weight: 700; margin: 4px 0; }}
    .kpi .label {{ font-size: 12px; color: #64748b; }}
    .positive {{ color: #22c55e; }}
    .negative {{ color: #ef4444; }}
    table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; margin-bottom: 16px; }}
    th {{ background: #334155; padding: 10px 12px; text-align: left; font-size: 13px; color: #94a3b8; font-weight: 600; }}
    td {{ padding: 8px 12px; font-size: 13px; border-bottom: 1px solid #1e293b; }}
    tr:hover {{ background: #334155; }}
    .risk-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; }}
</style>
</head>
<body>
<div class="container">
    <h1>📊 量化交易日报</h1>
    <p class="subtitle">{today} | 资金规模: {self.config.get('risk', {}).get('total_capital', 0):,.0f}</p>

    <div class="kpi-grid">
        <div class="kpi">
            <div class="label">总资产</div>
            <div class="value">{performance.get('current_nav', 1) * self.config.get('risk', {}).get('total_capital', 1000000):,.0f}</div>
        </div>
        <div class="kpi">
            <div class="label">日盈亏</div>
            <div class="value {'positive' if performance.get('avg_daily_return', 0) >= 0 else 'negative'}">{performance.get('avg_daily_return', 0):+.2f}%</div>
        </div>
        <div class="kpi">
            <div class="label">累计收益</div>
            <div class="value {'positive' if performance.get('total_return', 0) >= 0 else 'negative'}">{performance.get('total_return', 0):+.2f}%</div>
        </div>
        <div class="kpi">
            <div class="label">年化收益</div>
            <div class="value">{performance.get('annual_return', 0):.2f}%</div>
        </div>
        <div class="kpi">
            <div class="label">夏普比率</div>
            <div class="value">{performance.get('sharpe_ratio', 0):.3f}</div>
        </div>
        <div class="kpi">
            <div class="label">最大回撤</div>
            <div class="value negative">{performance.get('max_drawdown', 0):.2f}%</div>
        </div>
        <div class="kpi">
            <div class="label">胜率</div>
            <div class="value">{performance.get('win_rate', 0):.1f}%</div>
        </div>
        <div class="kpi">
            <div class="label">风控状态</div>
            <div class="value"><span class="risk-badge" style="background:{risk_color}33;color:{risk_color}">{risk_level.upper()}</span></div>
        </div>
    </div>

    <h2>📋 持仓明细 ({len(positions)} 只)</h2>
    <table>
        <tr><th>代码</th><th>数量</th><th>成本</th><th>现价</th><th>市值</th><th>盈亏</th><th>盈亏%</th></tr>
        {pos_rows}
    </table>

    <h2>🔄 今日交易 ({len(trades)} 笔)</h2>
    <table>
        <tr><th>时间</th><th>代码</th><th>方向</th><th>数量</th><th>价格</th><th>金额</th><th>费用</th></tr>
        {trade_rows}
    </table>

    {"<h2>📈 因子IC分析 (Top 20)</h2>" if ic_rows else ""}
    {"<table><tr><th>因子</th><th>IC</th><th>Rank IC</th><th>t统计量</th><th>样本数</th></tr>" + ic_rows + "</table>" if ic_rows else ""}

    <h2>🛡️ 风控报告</h2>
    <table>
        <tr><th>指标</th><th>值</th></tr>
        <tr><td>风控等级</td><td><span class="risk-badge" style="background:{risk_color}33;color:{risk_color}">{risk_level}</span></td></tr>
        <tr><td>日最大回撤</td><td>{risk_report.get('max_drawdown_pct', '0%')}</td></tr>
        <tr><td>仓位比例</td><td>{risk_report.get('total_position_pct', '0%')}</td></tr>
        <tr><td>今日违规次数</td><td>{risk_report.get('violations_today', 0)}</td></tr>
        <tr><td>交易频率</td><td>{risk_report.get('trades_per_minute', 0)} 笔/分钟</td></tr>
    </table>

    {"<h2>📉 滑点分析</h2><table><tr><th>指标</th><th>值</th></tr>" +
     f"<tr><td>平均滑点</td><td>{slippage_report.get('avg_slippage_bps', 0):.2f} bps</td></tr>" +
     f"<tr><td>最大滑点</td><td>{slippage_report.get('max_slippage_bps', 0):.2f} bps</td></tr>" +
     f"<tr><td>滑点总成本</td><td>{slippage_report.get('total_slippage_cost', 0):,.0f}</td></tr>" +
     "</table>" if slippage_report else ""}

    <p style="text-align:center;color:#475569;margin-top:32px;font-size:12px;">
        Generated at {datetime.now():%Y-%m-%d %H:%M:%S} | Quant Trading System v1.0
    </p>
</div>
</body>
</html>"""

        # 保存
        filepath = self.report_dir / f"report_{today}.html"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"📊 日报已生成: {filepath}")
        return str(filepath)
