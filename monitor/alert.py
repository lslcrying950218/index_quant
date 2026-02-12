"""
监控告警模块
- 微信/钉钉推送
- 日志记录
- Grafana指标上报
"""
import json
import time
import requests
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertManager:
    """告警管理器"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.wechat_webhook = self.config.get('wechat_webhook', '')
        self.dingtalk_webhook = self.config.get('dingtalk_webhook', '')
        self._last_alert_time = {}
        self._cooldown = 60  # 同类告警冷却时间(秒)

    def send_info(self, message: str):
        """普通信息"""
        logger.info(f"[INFO] {message}")
        self._push(f"ℹ️ {message}", level="info")

    def send_warning(self, message: str):
        """警告"""
        logger.warning(f"[WARNING] {message}")
        self._push(f"⚠️ {message}", level="warning")

    def send_critical(self, message: str):
        """严重告警 (无冷却)"""
        logger.critical(f"[CRITICAL] {message}")
        self._push(f"🚨 {message}", level="critical", force=True)

    def send_trade_report(self, report: dict):
        """交易报告"""
        msg = (
            f"📊 **交易日报** {datetime.now():%Y-%m-%d}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💰 日盈亏: {report.get('daily_pnl', 0):+,.0f} "
            f"({report.get('daily_pnl_pct', '0%')})\n"
            f"📈 总资产: {report.get('total_asset', 0):,.0f}\n"
            f"📉 最大回撤: {report.get('max_drawdown_pct', '0%')}\n"
            f"🔄 交易笔数: {report.get('trade_count', 0)}\n"
            f"📋 持仓数: {report.get('position_count', 0)}\n"
            f"💵 仓位: {report.get('total_position_pct', '0%')}\n"
            f"━━━━━━━━━━━━━━━"
        )
        self._push(msg, level="info", force=True)

    def _push(self, message: str, level: str = "info", force: bool = False):
        """推送消息"""
        # 冷却检查
        if not force:
            key = hash(message[:50])
            now = time.time()
            if key in self._last_alert_time:
                if now - self._last_alert_time[key] < self._cooldown:
                    return
            self._last_alert_time[key] = now

        # 钉钉推送
        if self.dingtalk_webhook:
            self._send_dingtalk(message)

        # 微信推送
        if self.wechat_webhook:
            self._send_wechat(message)

    def _send_dingtalk(self, message: str):
        """钉钉机器人推送"""
        try:
            data = {
                "msgtype": "markdown",
                "markdown": {
                    "title": "量化系统通知",
                    "text": message
                }
            }
            resp = requests.post(
                self.dingtalk_webhook,
                json=data,
                timeout=5
            )
            if resp.status_code != 200:
                logger.error(f"钉钉推送失败: {resp.text}")
        except Exception as e:
            logger.error(f"钉钉推送异常: {e}")

    def _send_wechat(self, message: str):
        """企业微信机器人推送"""
        try:
            data = {
                "msgtype": "markdown",
                "markdown": {
                    "content": message
                }
            }
            resp = requests.post(
                self.wechat_webhook,
                json=data,
                timeout=5
            )
            if resp.status_code != 200:
                logger.error(f"微信推送失败: {resp.text}")
        except Exception as e:
            logger.error(f"微信推送异常: {e}")
