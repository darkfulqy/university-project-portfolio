"""Pagination adapters reusing the verified notice/detail parsers."""

from __future__ import annotations

import hashlib
import json
import re
import time
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from announcement_fields import dates
from collect_announcements import Collector, CollectionError


SUPPORTED = {"beijing_ccgp", "beijing_ggzy", "guangdong_ccgp", "guangdong_gdegp", "jilin_ggzy",
             "yunnan_ggzy", "yunnan_easyjcx", "yunnan_kust", "xinjiang_ggzy"}


class PagedCollector(Collector):
    def __init__(self, key, output, page=1, page_size=20, interval=1.0, before="", window_start=1):
        super().__init__(key, output, page_size, 0)
        self.page = page
        self.interval = interval
        self.list_ids = []
        self.source_total = None
        self.list_count = None
        self.list_error = ""
        self.before = before
        self.window_start = window_start
        self.oldest_timestamp = ""

    def fetch(self, *args, **kwargs):
        for attempt in range(3):
            if (self.output.parents[2] / "STOP").exists():
                raise CollectionError("收到停止请求")
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
            try:
                return super().fetch(*args, **kwargs)
            except CollectionError as exc:
                if attempt == 2 or not re.search(r"HTTP (?:0|5\d\d)", str(exc)):
                    raise
                time.sleep(2 ** attempt)

    def remember(self, items, total=None):
        self.list_count = len(items)
        self.source_total = total
        self.list_ids = [str(i.get("id") or i.get("guid") or i.get("docpuburl") or i.get("orderMainId") or i.get("url") or i) for i in items]

    def api(self, url, **kwargs):
        parsed = urlsplit(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        mode = None
        if "was5/web/search" in url:
            query.update(page=self.page, prepage=self.limit)
            mode = "jilin"
        elif "selectInfoForIndex" in url:
            query.update(currPage=self.page, pageSize=self.limit)
            mode = "guangdong"
        elif "getCgggList" in url:
            kwargs["body"] = {**kwargs["body"], "pageNum": self.page, "pageSize": self.limit}
            mode = "yunnan"
        elif url.endswith("/portalCli/bidNeed"):
            kwargs["body"] = {**kwargs["body"], "current": self.page, "size": self.limit}
            mode = "easyjcx"
        elif kwargs.get("form", {}).get("page") == "cms.psms.publish.query":
            kwargs["form"] = {**kwargs["form"], "start": (self.page - 1) * self.limit + 1, "limit": self.limit}
            mode = "kust"
        if query:
            url = urlunsplit(parsed._replace(query=urlencode(query)))
        obj, path = super().api(url, **kwargs)
        if mode == "jilin":
            self.remember(obj.get("datas", []), obj.get("recordnum"))
        elif mode == "guangdong":
            self.remember(obj.get("data", {}).get("rows", []), obj.get("data", {}).get("total"))
        elif mode == "yunnan":
            self.remember(obj.get("value", {}).get("list", []), obj.get("value", {}).get("total"))
        elif mode == "easyjcx":
            self.remember((obj.get("data") or {}).get("records", []), (obj.get("data") or {}).get("total"))
        elif mode == "kust":
            self.remember(obj.get("resultset", []), obj.get("count"))
        return obj, path

    def static(self):
        if self.key == "xinjiang_ggzy":
            return self.xinjiang()
        if self.key == "beijing_ccgp":
            urls = [f"http://www.ccgp-beijing.gov.cn/xxgg/sjxxgg/zbgg/A002004001001index_{self.page}.htm",
                    f"http://www.ccgp-beijing.gov.cn/xxgg/qjxxgg/qjzbgg/A002004002001index_{self.page}.htm"]
            pattern = r"/(?:zbgg|qjzbgg)/\d"
        elif self.key == "beijing_ggzy":
            urls = ["https://ggzyfw.beijing.gov.cn/jyxxcggg/" + ("index.html" if self.page == 1 else f"index_{self.page}.html")]
            pattern = r"/jyxxcggg/\d"
        else:
            urls = ["https://gdegp.gds.edu.cn/dzfpList.aspx"]
            pattern = r"dzfpContent\.aspx\?BillGuid="
        found = {}
        for url in urls:
            try:
                markup, _, _ = self.fetch(url)
                soup = BeautifulSoup(markup, "html.parser")
                if self.key == "guangdong_gdegp" and self.page > 1:
                    controls = " ".join(x.get("href") or x.get("onclick") or "" for x in soup.select("a[href],input[onclick]"))
                    match = re.search(r"__doPostBack\('([^']*UcPagination)'", controls)
                    if not match:
                        raise CollectionError("未发现反拍列表分页控件")
                    inputs = {x["name"]: x.get("value", "") for x in soup.select("input[type=hidden][name]")}
                    inputs.update(__EVENTTARGET=match.group(1), __EVENTARGUMENT=str(self.page))
                    markup, _, _ = self.fetch(url, form=inputs)
                    soup = BeautifulSoup(markup, "html.parser")
                for a in soup.select("a[href]"):
                    href = urljoin(url, a["href"])
                    if not re.search(pattern, href, re.I):
                        continue
                    title = a.get("title") or a.get_text(" ", strip=True)
                    if re.search("更正|终止|废标|成交结果|中标结果", title):
                        continue
                    row = a.find_parent("li") or a.parent
                    ds = dates(row.get_text(" ", strip=True))
                    found[href] = (title, {"list_publish_date": ds[-1] if ds else None})
            except CollectionError as exc:
                self.errors.append(str(exc))
                self.list_error = str(exc)
        self.remember([{"url": url} for url in found])
        for url, (title, item) in found.items():
            try:
                markup, path, _ = self.fetch(url)
                self.add(url, markup, title, data=item, raw_file=path)
            except CollectionError as exc:
                self.errors.append(str(exc))

    def xinjiang(self):
        endpoint = "https://ggzy.xinjiang.gov.cn/inteligentsearchnew/rest/esinteligentsearch/getFullTextDataNew"
        body = {"token": "", "pn": (self.page - self.window_start) * self.limit, "rn": self.limit,
                "sdt": "", "edt": "", "wd": "", "inc_wd": "", "exc_wd": "", "fields": "title;projectnum;projectname",
                "cnum": "002", "sort": "{\"webdate\":\"0\"}", "ssort": "title", "cl": 200,
                "terminal": "", "condition": [{"fieldName": "categorynum", "isLike": True, "likeType": 2, "equal": "001004003"}],
                "time": None, "highlights": "", "statistics": None, "unionCondition": [],
                "accuracy": "100", "noParticiple": "0", "searchRange": None, "isBusiness": 1}
        if self.before:
            body["time"] = [{"fieldName": "webdate", "startTime": "1900-01-01 00:00:00", "endTime": self.before}]
        obj, _ = self.api(endpoint, body=body)
        data = obj.get("result", obj)
        if isinstance(data, str):
            data = json.loads(data)
        rows = data.get("records", [])
        self.remember(rows, data.get("totalcount"))
        timestamps = [str(row.get("webdate") or "").replace("T", " ")[:19] for row in rows]
        self.oldest_timestamp = min((value for value in timestamps if re.match(r"\d{4}-\d{2}-\d{2}", value)), default="")
        for item in rows:
            url = item.get("linkurl") or ""
            if not url:
                continue
            if not url.startswith("/xinjiangggzy_new/"):
                url = "/xinjiangggzy_new/" + url.lstrip("/")
            url = urljoin("https://ggzy.xinjiang.gov.cn", url)
            try:
                markup, path, _ = self.fetch(url)
                self.add(url, markup, item.get("title", ""), data={"list_publish_date": item.get("webdate")},
                         raw_file=path, extra={"list_item": item})
            except CollectionError as exc:
                self.errors.append(str(exc))

    @property
    def signature(self):
        return hashlib.sha256(json.dumps(self.list_ids, sort_keys=True).encode()).hexdigest() if self.list_ids else ""
