"""
新闻/公告爬虫
- 财经新闻: 东方财富、新浪财经、同花顺
- 公司公告: 巨潮资讯
- 研报摘要: Wind/Choice
- 增量爬取 + 去重
"""
import re
import time
import hashlib
import threading
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """新闻条目"""
    news_id: str              # 唯一ID (内容hash)
    title: str
    content: str
    source: str               # eastmoney / sina / cninfo
    url: str
    publish_time: datetime
    symbols: List[str]        # 关联股票代码
    category: str             # news / announcement / report
    crawl_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.news_id:
            raw = f"{self.title}{self.content[:100]}{self.publish_time}"
            self.news_id = hashlib.md5(raw.encode()).hexdigest()


class NewsCrawler:
    """新闻爬虫管理器"""

    def __init__(self, config: dict, redis_cache=None):
        self.config = config
        self.redis = redis_cache
        self._running = False
        self._seen_ids: set = set()       # 去重集合
        self._news_buffer: deque = deque(maxlen=5000)
        self._callbacks = []

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
        })

        # 爬取间隔
        self.crawl_interval = config.get("crawl_interval", 60)

    def register_callback(self, callback):
        """注册新闻回调 (新闻到达时触发NLP处理)"""
        self._callbacks.append(callback)

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._crawl_loop, daemon=True, name="news_crawler"
        )
        self._thread.start()
        logger.info("📰 新闻爬虫已启动")

    def stop(self):
        self._running = False
        logger.info("新闻爬虫已停止")

    def _crawl_loop(self):
        while self._running:
            try:
                # 并行爬取多个源
                news_items = []
                news_items.extend(self._crawl_eastmoney())
                news_items.extend(self._crawl_cninfo())
                news_items.extend(self._crawl_sina_finance())

                # 去重 & 分发
                new_count = 0
                for item in news_items:
                    if item.news_id not in self._seen_ids:
                        self._seen_ids.add(item.news_id)
                        self._news_buffer.append(item)
                        new_count += 1

                        # 触发回调
                        for cb in self._callbacks:
                            try:
                                cb(item)
                            except Exception as e:
                                logger.error(f"新闻回调异常: {e}")

                if new_count > 0:
                    logger.info(f"📰 新增 {new_count} 条新闻/公告")

                # 定期清理去重集合
                if len(self._seen_ids) > 50000:
                    self._seen_ids = set(list(self._seen_ids)[-30000:])

            except Exception as e:
                logger.error(f"爬虫异常: {e}")

            time.sleep(self.crawl_interval)

    # ==================== 东方财富 ====================

    def _crawl_eastmoney(self) -> List[NewsItem]:
        """爬取东方财富7x24快讯"""
        items = []
        try:
            url = "https://np-listapi.eastmoney.com/comm/web/getNewsByColumns"
            params = {
                "columns": "74",       # 7x24
                "pageSize": "30",
                "lastTime": "",
                "source": "web",
            }
            resp = self.session.get(url, params=params, timeout=10)
            data = resp.json()

            for item in data.get("data", {}).get("list", []):
                title = item.get("title", "")
                content = item.get("digest", "") or item.get("content", "")
                pub_time = item.get("showTime", "")

                # 提取关联股票
                symbols = self._extract_symbols(title + content)

                news = NewsItem(
                    news_id="",
                    title=title,
                    content=content,
                    source="eastmoney",
                    url=item.get("url_unique", ""),
                    publish_time=self._parse_time(pub_time),
                    symbols=symbols,
                    category="news",
                )
                items.append(news)

        except Exception as e:
            logger.debug(f"东方财富爬取异常: {e}")

        return items

    # ==================== 巨潮资讯 (公告) ====================

    def _crawl_cninfo(self) -> List[NewsItem]:
        """爬取巨潮资讯公司公告"""
        items = []
        try:
            url = "http://www.cninfo.com.cn/new/disclosure"
            params = {
                "action": "getLatestBulletinList",
                "pageSize": "30",
                "column": "szse",
                "tabName": "latest",
            }
            resp = self.session.post(url, data=params, timeout=10)
            data = resp.json()

            for item in data.get("classifiedAnnouncements", []):
                for ann in item if isinstance(item, list) else [item]:
                    if not isinstance(ann, dict):
                        continue
                    title = ann.get("announcementTitle", "")
                    code = ann.get("secCode", "")
                    symbol = self._normalize_symbol(code)

                    news = NewsItem(
                        news_id="",
                        title=title,
                        content=title,  # 公告标题即摘要
                        source="cninfo",
                        url=f"http://www.cninfo.com.cn/new/disclosure/detail?annoId={ann.get('announcementId', '')}",
                        publish_time=datetime.now(),
                        symbols=[symbol] if symbol else [],
                        category="announcement",
                    )
                    items.append(news)

        except Exception as e:
            logger.debug(f"巨潮资讯爬取异常: {e}")

        return items

    # ==================== 新浪财经 ====================

    def _crawl_sina_finance(self) -> List[NewsItem]:
        """爬取新浪财经要闻"""
        items = []
        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get"
            params = {
                "pageid": "153",
                "lid": "2516",
                "k": "",
                "num": "30",
                "page": "1",
            }
            resp = self.session.get(url, params=params, timeout=10)
            data = resp.json()

            for item in data.get("result", {}).get("data", []):
                title = item.get("title", "")
                content = item.get("intro", "") or title
                pub_time = item.get("ctime", "")

                symbols = self._extract_symbols(title + content)

                news = NewsItem(
                    news_id="",
                    title=title,
                    content=content,
                    source="sina",
                    url=item.get("url", ""),
                    publish_time=self._parse_time(pub_time),
                    symbols=symbols,
                    category="news",
                )
                items.append(news)

        except Exception as e:
            logger.debug(f"新浪财经爬取异常: {e}")

        return items

    # ==================== 工具方法 ====================

    @staticmethod
    def _extract_symbols(text: str) -> List[str]:
        """从文本中提取股票代码"""
        # 匹配6位数字代码
        codes = re.findall(r'[（(](\d{6})[)）]', text)
        # 匹配 SH/SZ 前缀
        codes += re.findall(r'(?:SH|SZ|sh|sz)(\d{6})', text)
        return list(set(codes))

    @staticmethod
    def _normalize_symbol(code: str) -> str:
        """标准化股票代码"""
        if not code or len(code) != 6:
            return ""
        if code.startswith(("6", "5")):
            return f"{code}.SH"
        elif code.startswith(("0", "3", "1")):
            return f"{code}.SZ"
        return code

    @staticmethod
    def _parse_time(time_str: str) -> datetime:
        """解析时间字符串"""
        if not time_str:
            return datetime.now()
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S"]:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromtimestamp(int(time_str))
        except (ValueError, OSError):
            return datetime.now()

    def get_recent_news(self, symbol: str = None, hours: int = 24) -> List[NewsItem]:
        """获取最近N小时的新闻"""
        cutoff = datetime.now() - timedelta(hours=hours)
        items = [n for n in self._news_buffer if n.crawl_time > cutoff]
        if symbol:
            items = [n for n in items if symbol in n.symbols or not n.symbols]
        return items
