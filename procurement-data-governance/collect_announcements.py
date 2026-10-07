#!/usr/bin/env python3
"""Collect a bounded sample of tender notices and export all fields to Excel."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit

from bs4 import BeautifulSoup

from announcement_fields import COLUMNS, DATE_COLUMNS, NUMBER_COLUMNS, parse_notice, html_text, dates, pdf_paragraphs
from download_site import (SITES, HttpClient, decode_text, parse_jsonp, flatten_attachments,
                           walk_file_items, normalize_easyjcx_file_url, sanitize_filename)

ROOT = Path(__file__).resolve().parent
STATIC = {
    "beijing_ccgp": (["http://www.ccgp-beijing.gov.cn/xxgg/sjxxgg/zbgg/A002004001001index_1.htm"], r"/zbgg/\d"),
    "beijing_ggzy": (["https://ggzyfw.beijing.gov.cn/"], r"/jyxxcggg/\d"),
    "guangdong_gdegp": (["https://gdegp.gds.edu.cn/dzfpList.aspx"], r"dzfpContent\.aspx\?BillGuid="),
    "xinjiang_ggzy": (["https://ggzy.xinjiang.gov.cn/"], r"/jyxx/.*?/\d{8}/"),
}


class CollectionError(Exception):
    pass


class Collector:
    def __init__(self, key: str, output: Path, limit: int, attachment_limit: int):
        self.key, self.config = key, SITES[key]
        self.output = output / key
        self.output.mkdir(parents=True, exist_ok=True)
        self.limit, self.attachment_limit = limit, attachment_limit
        self.client = HttpClient(use_curl=True)
        self.notices, self.requests, self.errors = [], [], []
        self.last_request = 0.0
        self.downloaded = 0
        self.started = datetime.now().isoformat(sep=" ", timespec="seconds")

    def fetch(self, url, *, body=None, form=None, referer=None):
        headers = {"Referer": referer or self.config["start_urls"][0], "Accept": "application/json,text/html;q=0.9,*/*;q=0.8"}
        payload = None
        if body is not None:
            payload = json.dumps(body, ensure_ascii=False).encode()
            headers["Content-Type"] = "application/json;charset=UTF-8"
        if form is not None:
            payload = urlencode(form, doseq=True).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
            headers["X-Requested-With"] = "XMLHttpRequest"
        time.sleep(max(0, 0.35 - (time.monotonic() - self.last_request)))
        result = self.client.fetch(url, method="POST" if payload is not None else "GET", body=payload,
                                   headers=headers, timeout=12)
        self.last_request = time.monotonic()
        fingerprint = hashlib.sha256(url.encode() + (payload or b"")).hexdigest()[:20]
        suffix = ".pdf" if result.content.startswith(b"%PDF") else ".json" if "json" in result.headers.get("content-type", "") else ".html"
        target = self.output / (fingerprint + suffix)
        target.write_bytes(result.content)
        self.requests.append({"url": url, "final_url": result.url, "method": "POST" if payload is not None else "GET",
                              "body": body if body is not None else form, "status": result.status,
                              "error": result.error, "file": str(target), "bytes": len(result.content),
                              "collected_at": datetime.now().isoformat(sep=" ", timespec="seconds")})
        if result.status != 200 or not result.content:
            raise CollectionError(f"HTTP {result.status}: {result.error or '空响应'} ({url})")
        text = decode_text(result)
        if len(text) < 20000 and re.search(r"<title>[^<]*(?:Forbidden|AccessDeny|风控|验证码|405|403)|访问被拒绝|访问验证|JFE Forbidden", text, re.I):
            raise CollectionError(f"页面返回访问限制 ({url})")
        return text, str(target), result

    def api(self, url, **kwargs):
        text, path, _ = self.fetch(url, **kwargs)
        try:
            obj = parse_jsonp(text) if "callback=" in url else json.loads(text)
        except (ValueError, TypeError) as exc:
            raise CollectionError(f"接口未返回有效JSON ({url})") from exc
        if isinstance(obj, dict) and obj.get("code") not in (None, 0, 1, 200, "0", "1", "200"):
            raise CollectionError(f"接口业务错误: {obj.get('code')} {obj.get('message')}")
        return obj, path

    def add(self, url, markup, title="", data=None, raw_file="", api_url="", extra=None):
        notice = parse_notice(self.key, self.config, url, markup, title=title, data=data,
                              raw_file=raw_file, api_url=api_url,
                              collected_at=datetime.now().isoformat(sep=" ", timespec="seconds"))
        if extra:
            notice["raw"].update(extra)
        if not any(x["record"]["公告ID"] == notice["record"]["公告ID"] for x in self.notices):
            self.notices.append(notice)
        return notice

    def static(self):
        urls, pattern = STATIC[self.key]
        candidates = {}
        for url in urls:
            markup, _, _ = self.fetch(url)
            soup = BeautifulSoup(markup, "html.parser")
            for a in soup.select("a[href]"):
                href = urljoin(url, a["href"])
                title = a.get("title") or a.get_text(" ", strip=True)
                if not re.search(pattern, href, re.I):
                    continue
                if self.key == "xinjiang_ggzy" and (not re.search("招标公告|磋商公告|谈判公告|询价公告|采购公告", title) or re.search("更正|异常|废标|终止|结果", title)):
                    continue
                row_dates = dates(a.parent.get_text(" ", strip=True))
                candidates[href] = (title, {"list_publish_date": row_dates[-1] if row_dates else None})
        # Prefer government procurement over construction when both are exposed.
        ordered = sorted(candidates.items(), key=lambda pair: 0 if re.search("政府采购|磋商|谈判|采购公告", pair[1][0]) else 1)
        for url, (title, metadata) in ordered[:self.limit * 3]:
            if len(self.notices) >= self.limit:
                break
            try:
                markup, path, _ = self.fetch(url)
                self.add(url, markup, title, data=metadata, raw_file=path)
            except CollectionError as exc:
                self.errors.append(str(exc))

    def jilin(self):
        query = "modal<>3 and gtitle<>'' and gtitle<>'null' and tType='政府采购' and iType='采购公告' "
        endpoint = "http://was.jl.gov.cn/was5/web/search?" + urlencode({"channelid": "237687", "page": 1, "prepage": self.limit, "searchword": query, "callback": "result"})
        obj, _ = self.api(endpoint)
        for item in obj.get("datas", [])[:self.limit]:
            url = item["docpuburl"].replace("http://ceshi5.jl.gov.cn/", "http://www.jl.gov.cn/")
            try:
                markup, path, _ = self.fetch(url)
                self.add(url, markup, item["title"], item, path, extra={"list_item": item})
            except CollectionError as exc:
                self.errors.append(str(exc))

    def guangdong(self):
        base = "https://gdgpo.czt.gd.gov.cn/gpcms/rest/web/v2"
        site, _ = self.api(base + "/index/getDeploymentSiteId?domain=gdgpo.czt.gd.gov.cn")
        site_id = site["data"]["id"]
        dictionary, _ = self.api(base + "/index/getDictInfo?" + urlencode({"dictType": "xmcg-noticeType", "site": site_id}))
        category = next((x for x in dictionary.get("data", []) if x.get("dictName") == "采购公告"), None)
        if not category:
            raise CollectionError("新版广东平台未返回采购公告分类")
        params = {"siteId": site_id, "channel": category.get("remark") or "fca71be5-fc0c-45db-96af-f513e9abda9d",
                  "currPage": 1, "pageSize": self.limit, "noticeType": category["dictCode"],
                  "regionCode": "", "cityOrArea": "", "subChannel": "false"}
        obj, _ = self.api(base + "/info/selectInfoForIndex?" + urlencode(params))
        for item in obj.get("data", {}).get("rows", [])[:self.limit]:
            endpoint = base + "/info/getInfoById?" + urlencode({"id": item["id"]})
            try:
                detail, path = self.api(endpoint)
                fields = detail.get("data") or {}
                public_url = "https://gdgpo.czt.gd.gov.cn/gpcms-center-web/#/noticeGd?" + urlencode({"type": "notice", "id": item["id"], "noticeType": item.get("noticeType") or "", "openTenderCode": item.get("openTenderCode") or "", "channelName": "项目采购公告"})
                self.add(public_url, fields.get("content") or "", title=fields.get("title") or item["title"],
                         data=fields, raw_file=path, api_url=endpoint, extra={"list_item": item})
            except CollectionError as exc:
                self.errors.append(str(exc))

    def yunnan(self):
        base = "https://ggzy.yn.gov.cn/ynggfwpt-home-api"
        obj, _ = self.api(base + "/jyInfo/zfcg/getCgggList", body={"pageNum": 1, "pageSize": self.limit, "cityId": "", "industryCode": "", "childType": "", "title": ""})
        for item in obj.get("value", {}).get("list", [])[:self.limit]:
            endpoint = base + "/jyInfo/zfcg/findCgggByGuid?" + urlencode({"guid": item["guid"]})
            try:
                detail, path = self.api(endpoint)
                data = detail.get("value") or {}
                notice = self.add(endpoint, data.get("bulletincontent") or "", data=data,
                                  raw_file=path, api_url=endpoint, extra={"list_item": item})
                try:
                    attach, _ = self.api(base + "/u/file/queryAttachmentList", body={"ownerGuid": item["guid"], "publishTime": None, "isYingCangFuJian": None})
                    notice["raw"]["attachment_response"] = attach
                    for file in flatten_attachments(attach.get("value")):
                        fid = file.get("fileId") or file.get("id") or file.get("guid")
                        if fid:
                            notice["attachments"].append({"name": file.get("fileName") or file.get("name") or str(fid), "url": base + "/u/file/downloadFile?" + urlencode({"fileId": fid}), "status": "仅采集链接"})
                except CollectionError as exc:
                    notice["raw"]["attachment_error"] = str(exc)
            except CollectionError as exc:
                self.errors.append(str(exc))

    def easyjcx(self):
        endpoint = "https://api.easyjcx.com/api/portalCli/bidNeed"
        obj, _ = self.api(endpoint, body={"keyword": "", "current": 1, "size": self.limit,
            "condition": {"procurementType": "", "sourceType": "", "collegeNameList": [], "areaScope": ""}})
        for item in (obj.get("data") or {}).get("records", [])[:self.limit]:
            endpoint = "https://api.easyjcx.com/api/portalCli/bidNeed/detail"
            try:
                detail, path = self.api(endpoint, body={"orderMainId": item["orderMainId"]})
                data = detail.get("data") or {}
                # All scalar and nested fields are retained in raw, without a fixed-field HTML renderer.
                markup = "<h1>" + html.escape(data.get("orderTitle") or "") + "</h1>"
                markup += "".join("<p>" + html.escape(k + ": " + str(v)) + "</p>" for k, v in data.items() if v is not None and not isinstance(v, (dict, list)))
                markup += "".join("<p>" + html.escape(x.get("deviceName") or "") + "</p><pre>" + html.escape(x.get("deviceSpec") or "") + "</pre>" for x in data.get("detailList") or [])
                url = "https://www.easyjcx.com/#/purchase/detail/" + item["orderMainId"]
                notice = self.add(url, markup, data=data, raw_file=path, api_url=endpoint, extra={"list_item": item})
                for file in walk_file_items(detail):
                    url = normalize_easyjcx_file_url(file.get("ossPath") or "")
                    if url:
                        notice["attachments"].append({"name": file.get("attachmentName") or url.rsplit("/", 1)[-1], "url": url, "status": "仅采集链接"})
            except CollectionError as exc:
                self.errors.append(str(exc))

    def kust(self):
        endpoint = "https://cgzx.kust.edu.cn/sfw_cms/e"
        obj, _ = self.api(endpoint, form={"window_": "json", "request_method_": "ajax", "page": "cms.psms.publish.query",
             "start": 1, "limit": self.limit * 3, "sort": "begin_time desc", "type": ["ZCXQ", "YQXQ", "BYXQ", "CSXQ"],
             "categoryId": ["104926"], "notShopType": ["DYLY", "WTDY", "DYWT"], "keywords": ""})
        eligible = [item for item in obj.get("resultset", []) if not re.search("更正|变更|终止|废标|结果", item.get("subject", ""))]
        for item in eligible[:self.limit]:
            url = item.get("fullUrl") or self.config["start_urls"][0]
            try:
                api_url = "https://provider.yuncaitong.cn/api/publish/" + item["syncId"]
                detail, path = self.api(api_url)
                created = datetime.fromtimestamp(detail["createTime"] / 1000, ZoneInfo("Asia/Shanghai"))
                base = "https://provider.yuncaitong.cn/publish/" + created.strftime("%Y/%m/%d/") + detail["id"]
                published, _ = self.api(base + ".sson")
                content_type = detail.get("contentType")
                markup = detail.get("content") or ""
                content_url = ""
                pdf_attachment = None
                if content_type == "PDF":
                    from pypdf import PdfReader
                    content_url = base + "/content.pdf"
                    _, path, res = self.fetch(content_url)
                    if not res.content.startswith(b"%PDF"):
                        raise CollectionError("公告PDF返回了非PDF响应")
                    reader = PdfReader(io.BytesIO(res.content))
                    plain = "\n".join(p.extract_text() or "" for p in reader.pages)
                    markup = "<pre>" + html.escape(pdf_paragraphs(plain)) + "</pre>"
                    pdf_attachment = {"name": "公告正文.pdf", "url": content_url, "status": "已下载", "file": path,
                                      "bytes": len(res.content), "sha256": hashlib.sha256(res.content).hexdigest()}
                elif content_type == "HTML":
                    content_url = base + "/content.html"
                    markup, path, _ = self.fetch(content_url)
                notice = self.add(url, markup, data=item, raw_file=path, api_url=api_url,
                                  extra={"list_item": item, "provider_metadata": detail, "provider_detail": published, "content_url": content_url})
                if pdf_attachment:
                    notice["attachments"].append(pdf_attachment)
                for attachment in (published.get("publish") or {}).get("attach") or []:
                    notice["attachments"].append({"name": attachment.get("name") or "公告附件", "url": "https://file.yuncaitong.cn/" + attachment["id"], "status": "仅采集链接"})
            except Exception as exc:
                self.errors.append(f"详情失败 {url}: {exc}")
                self.add(url, "", data=item, raw_file=path if 'path' in locals() else "", extra={"detail_error": str(exc)})

    def probe(self):
        # Retain every attempted entrance and its HTTP evidence. An entrance is not a notice.
        errors = []
        for url in self.config["start_urls"]:
            try:
                text, _, _ = self.fetch(url)
                soup = BeautifulSoup(text, "html.parser")
                errors.append(f"入口可访问，尚无公告详情适配：{soup.title.get_text(strip=True) if soup.title else url}")
            except CollectionError as exc:
                errors.append(str(exc))
        self.errors.extend(errors)

    def attachments(self):
        for notice in self.notices:
            unique = {item["url"]: item for item in notice["attachments"]}
            notice["attachments"] = list(unique.values())
            notice["record"]["附件数量"] = len(unique)
            for item in notice["attachments"]:
                if item["status"] == "已下载":
                    continue
                if self.downloaded >= self.attachment_limit:
                    break
                self.downloaded += 1
                result = self.client.fetch(item["url"], timeout=20)
                signature = result.content[:20].lstrip().lower()
                if result.status != 200 or not result.content or signature.startswith((b"<!doctype", b"<html", b"{\"")) or "text/html" in result.headers.get("content-type", ""):
                    item["status"] = f"下载失败或非附件响应 HTTP {result.status}"
                    continue
                ext = ".pdf" if signature.startswith(b"%pdf") else ".zip" if signature.startswith(b"pk") else ".doc" if signature.startswith(b"\xd0\xcf") else ".bin"
                name = sanitize_filename(item["name"])
                if not re.search(r"\.(pdf|docx?|xlsx?|zip|rar|7z)$", name, re.I):
                    name += ext
                folder = self.output / "attachments"
                folder.mkdir(exist_ok=True)
                path = folder / (hashlib.sha256(item["url"].encode()).hexdigest()[:8] + "_" + name)
                path.write_bytes(result.content)
                item.update({"status": "已下载", "file": str(path), "bytes": len(result.content), "sha256": hashlib.sha256(result.content).hexdigest()})

    def run(self):
        try:
            if self.key == "guangdong_ccgp":
                self.guangdong()
            elif self.key in STATIC:
                self.static()
            else:
                {"jilin_ggzy": self.jilin, "yunnan_ggzy": self.yunnan, "easyjcx": self.easyjcx,
                 "kust": self.kust}.get(self.config["adapter"], self.probe)()
            self.attachments()
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
        status = {"站点代码": self.key, "省份": self.config["province"], "网站": self.config["name"],
                  "公告数": len(self.notices), "已取得正文数": sum(n["record"]["采集状态"] == "已采集" for n in self.notices),
                  "状态": "已采集" if self.notices else "未取得公告", "说明": "；".join(self.errors) or ("本次样本采集完成" if self.notices else "公开列表没有匹配公告"),
                  "入口URL": self.config["start_urls"][0], "采集时间": self.started}
        (self.output / "requests.json").write_text(json.dumps(self.requests, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output / "notices.json").write_text(json.dumps(self.notices, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{self.key}: {len(self.notices)} 条公告，{status['已取得正文数']} 条正文", flush=True)
        return self.notices, status


def export_excel(dataset: Path, output: Path):
    runtime = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    node = os.environ.get("NODE_BIN") or (str(runtime / "bin/node") if (runtime / "bin/node").exists() else shutil.which("node"))
    if not node:
        raise RuntimeError("未找到 Node.js；JSON 已保存。请安装 Node.js 并配置 @oai/artifact-tool。")
    subprocess.run([node, str(ROOT / "export_excel.mjs"), str(dataset), str(output)], check=True)


def reparse(notice):
    old = notice["record"]
    key = next(key for key, config in SITES.items() if config["name"] == old["来源网站"])
    raw = notice["raw"]
    path = Path(old["原始文件"])
    source = raw.get("api") or {}
    markup = ""
    if path.suffix == ".html" and path.exists():
        from download_site import FetchResult
        markup = decode_text(FetchResult(url=old["来源URL"], status=200, headers={}, content=path.read_bytes()))
    elif source.get("bulletincontent") or source.get("content"):
        markup = source.get("bulletincontent") or source["content"]
    else:
        markup = "<pre>" + html.escape(pdf_paragraphs(notice["text"]) if path.suffix == ".pdf" else notice["text"]) + "</pre>"
    # Original list HTML contains publication dates absent from some detail pages.
    if not source.get("list_publish_date") and key in STATIC:
        log = path.parent / "requests.json"
        if log.exists():
            for request in json.loads(log.read_text(encoding="utf-8")):
                if request["url"] not in STATIC[key][0]:
                    continue
                page = BeautifulSoup(Path(request["file"]).read_bytes(), "html.parser")
                for link in page.select("a[href]"):
                    if urljoin(request["url"], link["href"]) == old["来源URL"]:
                        matches = dates(link.parent.get_text(" ", strip=True))
                        if matches:
                            source["list_publish_date"] = matches[-1]
    fresh = parse_notice(key, SITES[key], old["来源URL"], markup, title=old["公告标题"], data=source,
                         collected_at=old["采集时间"], raw_file=old["原始文件"], api_url=old.get("详情接口URL") or "")
    fresh["attachments"] = list({item["url"]: item for item in fresh["attachments"] + notice["attachments"]}.values())
    fresh["record"]["附件数量"] = len(fresh["attachments"])
    fresh["record"]["数据来源方式"] = old.get("数据来源方式") or "本次在线采集"
    for key, value in raw.items():
        if key not in ("api", "sections", "html_meta", "html_tables"):
            fresh["raw"][key] = value
    # Pre-rendered API/PDF text has no HTML metadata; retain previously captured data.
    if not fresh["raw"]["html_tables"]:
        fresh["raw"]["html_tables"] = raw.get("html_tables") or []
    return fresh


def cached_yunnan(cache: Path, limit: int):
    root = cache / "yunnan_ggzy"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    notices = []
    for item in manifest["records"]:
        if item.get("category") != "api" or "findCgggByGuid" not in item.get("source_url", ""):
            continue
        path = Path(item["file"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        obj = payload.get("value") or {}
        notice = parse_notice("yunnan_ggzy", SITES["yunnan_ggzy"], item["source_url"], obj.get("bulletincontent") or "",
                              data=obj, collected_at=manifest["generated_at"], raw_file=str(path), api_url=item["source_url"])
        notice["record"]["数据来源方式"] = "历史本地样本"
        notice["record"]["字段备注"] = "；".join(filter(None, [notice["record"]["字段备注"], "本次接口访问受限，使用已有下载样本重新提取字段"]))
        notices.append(notice)
        if len(notices) >= limit:
            break
    return notices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="all", help="all 或逗号分隔的网站代码")
    parser.add_argument("--limit", type=int, default=2, help="每站公告样本数，默认2")
    parser.add_argument("--attachment-limit", type=int, default=2, help="每站最多尝试下载的附件数；0仅收集链接")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "collection_runs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--xlsx", type=Path, default=ROOT / "outputs" / "liqingyang_announcements" / "招标公告采集汇总.xlsx")
    parser.add_argument("--from-json", type=Path, help="从已经采集的dataset.json重新生成Excel，不联网")
    parser.add_argument("--merge-json", type=Path, nargs="+", help="合并多次采集结果，相同站点采用最后一次结果")
    parser.add_argument("--reparse", action="store_true", help="根据已保存的原始文件重新提取字段")
    parser.add_argument("--cache-fallback", type=Path, help="云南接口不可用时允许使用指定downloads目录中的旧样本，并明确标识")
    parser.add_argument("--no-excel", action="store_true")
    args = parser.parse_args()
    if args.from_json and not args.reparse and not args.cache_fallback:
        if not args.no_excel:
            export_excel(args.from_json.resolve(), args.xlsx.resolve())
        return 0
    keys = list(SITES) if args.site == "all" else args.site.split(",")
    if any(k not in SITES for k in keys) or args.limit < 1 or args.attachment_limit < 0:
        parser.error("站点代码或样本数量无效")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.merge_json or args.from_json:
        for file in args.merge_json or [args.from_json]:
            saved = json.loads(file.read_text(encoding="utf-8"))
            for status in saved["sites"]:
                results[status["站点代码"]] = ([n for n in saved["notices"] if n["record"]["来源网站"] == status["网站"]], status)
        keys = [k for k in SITES if k in results]
    else:
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
            tasks = {pool.submit(Collector(k, args.out, args.limit, args.attachment_limit).run): k for k in keys}
            for task in as_completed(tasks):
                results[tasks[task]] = task.result()
    notices = [n for key in keys for n in results[key][0]]
    if args.reparse:
        notices = [reparse(n) for n in notices]
    if args.cache_fallback and "yunnan_ggzy" in results and not results["yunnan_ggzy"][0]:
        cached = cached_yunnan(args.cache_fallback.resolve(), args.limit)
        notices.extend(cached)
        status = results["yunnan_ggzy"][1]
        status.update({"状态": "历史样本", "公告数": len(cached), "已取得正文数": len(cached),
                       "说明": status["说明"] + "；已从本地历史下载样本提取字段，非本次联网取得"})
    dataset = {"generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"), "columns": COLUMNS,
               "date_columns": sorted(DATE_COLUMNS), "number_columns": sorted(NUMBER_COLUMNS),
               "notices": notices, "sites": [results[key][1] for key in keys]}
    path = args.out / "dataset.json"
    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON: {path}\n公告总数: {len(notices)}", flush=True)
    if not args.no_excel:
        export_excel(path, args.xlsx.resolve())
    return 0 if notices else 1


if __name__ == "__main__":
    raise SystemExit(main())
