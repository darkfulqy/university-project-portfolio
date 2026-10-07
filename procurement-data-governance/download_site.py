#!/usr/bin/env python3
"""
Small-sample procurement downloaders for Li Qingyang's assigned platforms.

The scripts intentionally download only a few public samples for validation.
They save HTML/JSON announcement details when tender files are behind login or
when a site does not expose public attachments.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from network_route import open_url, curl_route_args


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)

FILE_EXTS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".rar",
    ".7z",
    ".txt",
    ".csv",
}

ASSET_EXTS = {
    ".css",
    ".js",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
}


SITES: dict[str, dict[str, Any]] = {
    "beijing_ccgp": {
        "name": "北京市政府采购网",
        "province": "北京市",
        "adapter": "static",
        "start_urls": [
            "http://www.ccgp-beijing.gov.cn/",
            "http://www.ccgp-beijing.gov.cn/xxgg/A002004index_1.htm",
        ],
        "notes": "静态公告页可直接下载 content_file 附件。",
    },
    "beijing_ggzy": {
        "name": "北京市公共资源交易服务平台",
        "province": "北京市",
        "adapter": "static",
        "start_urls": ["https://ggzyfw.beijing.gov.cn/"],
        "use_curl": True,
        "notes": "本机 Python TLS 会 BAD_ECPOINT，脚本使用 curl -k 后备。",
    },
    "beijing_mkt": {
        "name": "京华云采电子卖场",
        "province": "北京市",
        "adapter": "probe",
        "start_urls": ["https://mkt-bjzc.zhongcy.com/mall-view/"],
        "use_curl": True,
        "notes": "电子卖场入口有 JFE 风控，公开附件通常不在首页暴露。",
    },
    "guangdong_ccgp": {
        "name": "广东省政府采购网",
        "province": "广东省",
        "adapter": "probe",
        "start_urls": [
            "http://www.ccgp-guangdong.gov.cn/",
            "https://gdgpo.czt.gd.gov.cn/",
        ],
        "use_curl": True,
        "notes": "文档旧域名当前会断连；新版域名在部分网络下 SSL 断连。",
    },
    "guangdong_zcy": {
        "name": "广东政府采购智慧云平台",
        "province": "广东省",
        "adapter": "probe",
        "start_urls": ["http://gdgpo.cz.tgd.gov.cn/gp-auth-center/login?origin=oauth"],
        "use_curl": True,
        "notes": "登录/统一认证入口，公开下载需跳转到公告页或登录后操作。",
    },
    "guangdong_ggzy": {
        "name": "广东省公共资源交易平台",
        "province": "广东省",
        "adapter": "probe",
        "start_urls": ["http://bs.gdggzy.org.cn/osh-web/"],
        "use_curl": True,
        "notes": "入口在当前网络返回 empty reply，保留脚本用于验证环境。",
    },
    "guangdong_gdegp": {
        "name": "广东省教育部门政府采购管理平台",
        "province": "广东省",
        "adapter": "static",
        "start_urls": [
            "https://gdegp.gds.edu.cn/",
            "https://gdegp.gds.edu.cn/tzzcList.aspx?NoticeTypeEnum=1",
            "https://gdegp.gds.edu.cn/dzfpList.aspx",
        ],
        "notes": "传统 aspx 页面，通知详情里可直接下载 xls/rar 等附件。",
    },
    "jilin_ccgp": {
        "name": "吉林省政府采购网",
        "province": "吉林省",
        "adapter": "probe",
        "start_urls": ["http://www.ccgp-jilin.gov.cn/"],
        "use_curl": True,
        "notes": "入口有 Tengine/政采云风控；吉林公共资源接口另有专门脚本。",
    },
    "jilin_ggzy": {
        "name": "吉林省公共资源交易服务平台",
        "province": "吉林省",
        "adapter": "jilin_ggzy",
        "start_urls": ["https://www.jl.gov.cn/ggzy/"],
        "notes": "栏目页调用 was.jl.gov.cn JSONP，可直接取公告列表并保存详情。",
    },
    "jilin_zcy": {
        "name": "吉林省政府采购云平台",
        "province": "吉林省",
        "adapter": "probe",
        "start_urls": ["https://www.zcygov.cn/"],
        "use_curl": True,
        "notes": "政采云入口风控/登录，招标文件通常需供应商账号获取。",
    },
    "yunnan_yngp": {
        "name": "云南政府采购网",
        "province": "云南省",
        "adapter": "probe",
        "start_urls": ["http://www.yngp.com/"],
        "use_curl": True,
        "notes": "当前网络返回云防护/ACL；公共采购公告可用云南公共资源 API 验证。",
    },
    "yunnan_ggzy": {
        "name": "云南省公共资源交易信息网",
        "province": "云南省",
        "adapter": "yunnan_ggzy",
        "start_urls": ["https://ggzy.yn.gov.cn/"],
        "notes": "Vue 前端，直接调用 ynggfwpt-home-api 的政府采购接口。",
    },
    "yunnan_zcy": {
        "name": "云南政府采购云平台/政采云",
        "province": "云南省",
        "adapter": "probe",
        "start_urls": ["https://www.zcygov.cn/"],
        "use_curl": True,
        "notes": "政采云入口风控/登录，采购文件获取通常需要账号。",
    },
    "yunnan_easyjcx": {
        "name": "高校竞价采购网/竞采星",
        "province": "云南省",
        "adapter": "easyjcx",
        "start_urls": ["https://www.easyjcx.com"],
        "notes": "公开 API 可直接获取采购需求列表和详情。",
    },
    "yunnan_kust": {
        "name": "昆明理工大学招标采购网",
        "province": "云南省",
        "adapter": "kust",
        "start_urls": [
            "https://cgzx.kust.edu.cn/sfw_cms/e?page=cms.psms.gglist&typeDetail=XQ&categoryId=104926"
        ],
        "notes": "思必得 CMS datalist 控件，POST action=cms.psms.publish.query。",
    },
    "xinjiang_ccgp": {
        "name": "新疆政府采购网",
        "province": "新疆维吾尔自治区",
        "adapter": "probe",
        "start_urls": ["http://www.ccgp-xinjiang.gov.cn/"],
        "use_curl": True,
        "notes": "入口有 Tengine/政采云风控；公共资源平台另有专门脚本。",
    },
    "xinjiang_ggzy": {
        "name": "新疆公共资源交易网",
        "province": "新疆维吾尔自治区",
        "adapter": "static",
        "start_urls": ["https://ggzy.xinjiang.gov.cn/"],
        "notes": "首页静态暴露交易公告链接，可保存公告详情 HTML。",
    },
}


@dataclass
class FetchResult:
    url: str
    status: int
    headers: dict[str, str]
    content: bytes
    error: str = ""


class LinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.links: list[tuple[str, str]] = []
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k.lower(): v for k, v in attrs if v is not None}
        if tag.lower() == "title":
            self._in_title = True
        for key in ("href", "src"):
            value = attr.get(key)
            if value:
                text = attr.get("title") or attr.get("alt") or attr.get("download") or ""
                self.links.append((urllib.parse.urljoin(self.base_url, value), text))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and data.strip():
            self.title_parts.append(data.strip())

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()


class HttpClient:
    def __init__(self, use_curl: bool = False) -> None:
        self.use_curl = use_curl
        self.ctx = ssl._create_unverified_context()

    def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: int = 20,
    ) -> FetchResult:
        headers2 = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/json,text/plain,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        }
        if headers:
            headers2.update(headers)
        try:
            req = urllib.request.Request(url, data=body, headers=headers2, method=method)
            with open_url(req, timeout=timeout, context=self.ctx) as res:
                raw = res.read()
                raw = decompress(raw, res.headers.get("content-encoding", ""))
                return FetchResult(
                    url=res.geturl(),
                    status=res.status,
                    headers={k.lower(): v for k, v in res.headers.items()},
                    content=raw,
                )
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            raw = decompress(raw, exc.headers.get("content-encoding", ""))
            return FetchResult(
                url=exc.geturl(),
                status=exc.code,
                headers={k.lower(): v for k, v in exc.headers.items()},
                content=raw,
                error=str(exc),
            )
        except Exception as exc:
            if self.use_curl:
                return self.fetch_with_curl(url, timeout=timeout, error_prefix=str(exc),
                                            method=method, body=body, headers=headers2)
            return FetchResult(url=url, status=0, headers={}, content=b"", error=str(exc))

    def fetch_with_curl(self, url: str, *, timeout: int = 20, error_prefix: str = "",
                        method: str = "GET", body: bytes | None = None,
                        headers: dict[str, str] | None = None) -> FetchResult:
        cmd = [
            "curl",
            "-k",
            "-L",
            "--compressed",
            "--max-time",
            str(timeout),
            "-A",
            USER_AGENT,
            "--silent",
            "--show-error",
            "--write-out", "\nCODEX_HTTP_META:%{http_code}\t%{content_type}\t%{url_effective}",
            "--request", method,
        ]
        for key, value in (headers or {}).items():
            cmd.extend(["--header", f"{key}: {value}"])
        if body is not None:
            cmd.extend(["--data-binary", "@-"])
        cmd.extend(curl_route_args(url))
        cmd.append(url)
        try:
            proc = subprocess.run(cmd, input=body, check=False, capture_output=True, timeout=timeout + 5)
        except Exception as exc:
            return FetchResult(url=url, status=0, headers={}, content=b"", error=f"{error_prefix}; curl: {exc}")
        err = proc.stderr.decode("utf-8", "ignore").strip()
        content, _, metadata = proc.stdout.rpartition(b"\nCODEX_HTTP_META:")
        fields = metadata.decode("utf-8", "replace").split("\t", 2)
        status = int(fields[0]) if fields and fields[0].isdigit() else 0
        error = "; ".join(x for x in [error_prefix, err] if x) if proc.returncode or status >= 400 else ""
        return FetchResult(url=fields[2] if len(fields) > 2 else url, status=status,
                           headers={"content-type": fields[1]} if len(fields) > 1 else {},
                           content=content, error=error)


class ArtifactWriter:
    def __init__(self, output_dir: Path, site_key: str, site_name: str) -> None:
        self.root = output_dir / site_key
        self.site_key = site_key
        self.site_name = site_name
        self.records: list[dict[str, Any]] = []
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bytes(
        self,
        category: str,
        filename: str,
        content: bytes,
        *,
        source_url: str,
        status: int | None = None,
        note: str = "",
        content_type: str = "",
    ) -> Path:
        target_dir = self.root / category
        target_dir.mkdir(parents=True, exist_ok=True)
        safe = unique_path(target_dir / sanitize_filename(filename))
        safe.write_bytes(content)
        self.records.append(
            {
                "site": self.site_name,
                "category": category,
                "file": str(safe),
                "source_url": source_url,
                "status": status,
                "content_type": content_type,
                "bytes": len(content),
                "note": note,
            }
        )
        print(f"saved {category}: {safe.name}")
        return safe

    def save_text(
        self,
        category: str,
        filename: str,
        text: str,
        *,
        source_url: str,
        status: int | None = None,
        note: str = "",
        content_type: str = "text/plain; charset=utf-8",
    ) -> Path:
        return self.save_bytes(
            category,
            filename,
            text.encode("utf-8"),
            source_url=source_url,
            status=status,
            note=note,
            content_type=content_type,
        )

    def add_note(self, note: str, *, source_url: str = "", status: int | None = None) -> None:
        self.records.append(
            {
                "site": self.site_name,
                "category": "note",
                "file": "",
                "source_url": source_url,
                "status": status,
                "bytes": 0,
                "note": note,
            }
        )
        print(f"note: {note}")

    def finish(self) -> Path:
        manifest = {
            "site_key": self.site_key,
            "site_name": self.site_name,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "records": self.records,
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"manifest: {path}")
        return path


def decompress(raw: bytes, encoding: str) -> bytes:
    encoding = (encoding or "").lower()
    try:
        if "gzip" in encoding:
            return gzip.decompress(raw)
        if "deflate" in encoding:
            return zlib.decompress(raw)
    except Exception:
        return raw
    return raw


def decode_text(result: FetchResult) -> str:
    content_type = result.headers.get("content-type", "")
    match = re.search(r"charset=([\w.-]+)", content_type, re.I)
    encodings = []
    if match:
        encodings.append(match.group(1))
    head = result.content[:2000].decode("ascii", "ignore")
    meta = re.search(r"charset=['\"]?([\w.-]+)", head, re.I)
    if meta:
        encodings.append(meta.group(1))
    encodings.extend(["utf-8", "gb18030", "big5"])
    for enc in encodings:
        try:
            return result.content.decode(enc)
        except Exception:
            continue
    return result.content.decode("utf-8", "ignore")


def sanitize_filename(name: str, default: str = "sample") -> str:
    name = urllib.parse.unquote(name or "")
    name = html.unescape(name)
    name = name.strip().replace("\x00", "")
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        name = default
    if len(name) > 140:
        stem, ext = os.path.splitext(name)
        name = stem[:120].rstrip(" .") + ext
    return name


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(2, 1000):
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"too many duplicate files for {path}")


def filename_from_url(url: str, fallback: str = "download") -> str:
    parsed = urllib.parse.urlparse(url)
    name = os.path.basename(parsed.path.rstrip("/"))
    if not name:
        name = fallback
    if not os.path.splitext(name)[1] and "downloadFile" in url:
        name += ".bin"
    return sanitize_filename(name, fallback)


def parse_links(text: str, base_url: str) -> LinkParser:
    parser = LinkParser(base_url)
    try:
        parser.feed(text)
    except Exception:
        pass
    return parser


def is_probable_file_link(url: str) -> bool:
    lower = url.lower()
    parsed = urllib.parse.urlparse(lower)
    ext = os.path.splitext(parsed.path)[1]
    if ext in FILE_EXTS:
        return True
    markers = [
        "content_file",
        "/annex/",
        "virtual_attach_file",
        "downloadfile",
        "download_file",
        "attachment",
        "attach",
        "userfiles",
    ]
    if any(m in lower for m in markers):
        return not any(lower.endswith(ext) for ext in ASSET_EXTS)
    return False


def is_probable_article_link(url: str, base_hosts: set[str]) -> bool:
    lower = url.lower()
    if lower.startswith("javascript:") or lower.startswith("mailto:"):
        return False
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc and parsed.netloc.lower() not in base_hosts:
        return False
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext in ASSET_EXTS or ext in FILE_EXTS:
        return False
    article_markers = [
        ".html",
        ".htm",
        "tzzccontent.aspx",
        "dzfpcontent.aspx",
        "fpjgcontent.aspx",
        "cms.detail",
        "/jyxx/",
        "/zfcg/",
        "/xxgg/",
        "showbulletininfo",
    ]
    return any(marker in lower for marker in article_markers)


def download_file(client: HttpClient, writer: ArtifactWriter, url: str, *, name_hint: str = "") -> bool:
    result = client.fetch(url, timeout=30)
    content_type = result.headers.get("content-type", "")
    if not result.content:
        writer.add_note(f"附件下载失败或为空: {result.error}", source_url=url, status=result.status)
        return False
    if result.status >= 400 or looks_like_block_page(result):
        writer.save_bytes(
            "blocked_responses",
            filename_from_url(url, "blocked_response.html"),
            result.content,
            source_url=url,
            status=result.status,
            note=result.error or "blocked/non-file response",
            content_type=content_type,
        )
        return False
    name = name_hint or filename_from_url(result.url or url, "download")
    if not os.path.splitext(name)[1]:
        name = filename_from_url(url, name) + guess_ext(content_type)
    writer.save_bytes(
        "files",
        name,
        result.content,
        source_url=result.url or url,
        status=result.status,
        content_type=content_type,
    )
    return True


def looks_like_block_page(result: FetchResult) -> bool:
    content_type = result.headers.get("content-type", "").lower()
    sample = result.content[:1000].decode("utf-8", "ignore").lower()
    if "text/html" in content_type and any(word in sample for word in ["forbidden", "拒绝", "风控", "405", "云防护"]):
        return True
    return False


def guess_ext(content_type: str) -> str:
    content_type = (content_type or "").lower()
    if "pdf" in content_type:
        return ".pdf"
    if "word" in content_type:
        return ".doc"
    if "excel" in content_type or "spreadsheet" in content_type:
        return ".xls"
    if "zip" in content_type:
        return ".zip"
    if "html" in content_type:
        return ".html"
    return ".bin"


def save_json(writer: ArtifactWriter, category: str, filename: str, data: Any, source_url: str) -> None:
    writer.save_text(
        category,
        filename,
        json.dumps(data, ensure_ascii=False, indent=2),
        source_url=source_url,
        content_type="application/json; charset=utf-8",
    )


def response_title_or_name(text: str, base_url: str, fallback: str) -> str:
    parser = parse_links(text, base_url)
    if parser.title:
        return sanitize_filename(parser.title) + ".html"
    return filename_from_url(base_url, fallback + ".html")


def run_static(site_key: str, config: dict[str, Any], writer: ArtifactWriter, limit: int) -> None:
    client = HttpClient(use_curl=bool(config.get("use_curl")))
    base_hosts = {urllib.parse.urlparse(u).netloc.lower() for u in config["start_urls"] if urllib.parse.urlparse(u).netloc}
    article_seen: set[str] = set()
    file_seen: set[str] = set()
    file_count = 0
    article_count = 0

    for index, url in enumerate(config["start_urls"], start=1):
        result = client.fetch(url)
        text = decode_text(result) if result.content else ""
        page_name = response_title_or_name(text, result.url or url, f"start_{index}")
        writer.save_bytes(
            "pages",
            f"{index:02d}_{page_name}",
            result.content,
            source_url=result.url or url,
            status=result.status,
            note=result.error,
            content_type=result.headers.get("content-type", ""),
        )
        if not text:
            continue
        parser = parse_links(text, result.url or url)

        for link, title in parser.links:
            if file_count >= limit:
                break
            if link not in file_seen and is_probable_file_link(link):
                file_seen.add(link)
                if download_file(client, writer, link, name_hint=title):
                    file_count += 1

        article_links = []
        for link, _title in parser.links:
            if link not in article_seen and is_probable_article_link(link, base_hosts):
                article_links.append(link)
                article_seen.add(link)
        for link in article_links[: max(limit * 5, limit)]:
            if article_count >= limit and file_count >= limit:
                break
            article_result = client.fetch(link)
            article_text = decode_text(article_result) if article_result.content else ""
            article_name = response_title_or_name(article_text, article_result.url or link, f"article_{article_count + 1}")
            writer.save_bytes(
                "pages",
                f"article_{article_count + 1:02d}_{article_name}",
                article_result.content,
                source_url=article_result.url or link,
                status=article_result.status,
                note=article_result.error,
                content_type=article_result.headers.get("content-type", ""),
            )
            article_count += 1
            if article_text:
                article_parser = parse_links(article_text, article_result.url or link)
                for file_link, file_title in article_parser.links:
                    if file_count >= limit:
                        break
                    if file_link not in file_seen and is_probable_file_link(file_link):
                        file_seen.add(file_link)
                        if download_file(client, writer, file_link, name_hint=file_title):
                            file_count += 1

    writer.add_note(f"static adapter completed: {article_count} detail pages, {file_count} files")


def run_probe(site_key: str, config: dict[str, Any], writer: ArtifactWriter, limit: int) -> None:
    client = HttpClient(use_curl=bool(config.get("use_curl")))
    for index, url in enumerate(config["start_urls"][:limit], start=1):
        result = client.fetch(url)
        name = filename_from_url(result.url or url, f"probe_{index}.html")
        if not os.path.splitext(name)[1]:
            name += ".html"
        writer.save_bytes(
            "probe",
            f"{index:02d}_{name}",
            result.content or result.error.encode("utf-8"),
            source_url=result.url or url,
            status=result.status,
            note=result.error or config.get("notes", ""),
            content_type=result.headers.get("content-type", ""),
        )
    writer.add_note(config.get("notes", "probe only"))


def run_jilin_ggzy(site_key: str, config: dict[str, Any], writer: ArtifactWriter, limit: int) -> None:
    client = HttpClient()
    searchword = "modal<>3 and gtitle<>'' and gtitle<>'null' and tType='政府采购' and iType='采购公告' "
    url = (
        "http://was.jl.gov.cn/was5/web/search?"
        f"channelid=237687&page=1&prepage={limit}&searchword={urllib.parse.quote(searchword)}&callback=result"
    )
    result = client.fetch(url, headers={"Referer": "https://www.jl.gov.cn/ggzy/zfcg/cggg/"})
    text = decode_text(result)
    payload = parse_jsonp(text)
    save_json(writer, "api", "jilin_ggzy_list.json", payload, result.url or url)
    docs = payload.get("datas", [])[:limit]
    for index, item in enumerate(docs, start=1):
        detail_url = (item.get("docpuburl") or "").replace("http://ceshi5.jl.gov.cn/", "http://www.jl.gov.cn/")
        if not detail_url:
            continue
        detail = client.fetch(detail_url, headers={"Referer": "https://www.jl.gov.cn/ggzy/zfcg/cggg/"})
        detail_text = decode_text(detail)
        title = sanitize_filename(item.get("title") or f"jilin_detail_{index}")
        writer.save_bytes(
            "pages",
            f"{index:02d}_{title}.html",
            detail.content,
            source_url=detail.url or detail_url,
            status=detail.status,
            note=detail.error,
            content_type=detail.headers.get("content-type", ""),
        )
        parser = parse_links(detail_text, detail.url or detail_url)
        downloaded = 0
        for file_url, file_title in parser.links:
            if downloaded >= limit:
                break
            if is_probable_file_link(file_url) and download_file(client, writer, file_url, name_hint=file_title):
                downloaded += 1
    writer.add_note(f"jilin API completed: {len(docs)} details saved")


def run_yunnan_ggzy(site_key: str, config: dict[str, Any], writer: ArtifactWriter, limit: int) -> None:
    client = HttpClient()
    base = "https://ggzy.yn.gov.cn/ynggfwpt-home-api"
    headers = {
        "Referer": "https://ggzy.yn.gov.cn/tradeHall/tradeList",
        "Content-Type": "application/json;charset=UTF-8",
        "Accept": "application/json, text/plain, */*",
    }
    body = {"pageNum": 1, "pageSize": limit, "cityId": "", "industryCode": "", "childType": "", "title": ""}
    list_url = base + "/jyInfo/zfcg/getCgggList"
    result = client.fetch(list_url, method="POST", body=json.dumps(body).encode("utf-8"), headers=headers)
    payload = json.loads(decode_text(result))
    save_json(writer, "api", "yunnan_ggzy_zfcg_list.json", payload, result.url or list_url)
    items = payload.get("value", {}).get("list", [])[:limit]
    for index, item in enumerate(items, start=1):
        guid = item.get("guid")
        if not guid:
            continue
        detail_url = base + "/jyInfo/zfcg/findCgggByGuid?" + urllib.parse.urlencode({"guid": guid})
        detail = client.fetch(detail_url, headers={"Referer": "https://ggzy.yn.gov.cn/tradeHall/tradeDetail"})
        detail_payload = json.loads(decode_text(detail))
        title = sanitize_filename(item.get("bulletintitle") or f"yunnan_detail_{index}")
        save_json(writer, "api", f"{index:02d}_{title}.json", detail_payload, detail.url or detail_url)
        content_html = detail_payload.get("value", {}).get("bulletincontent")
        if content_html:
            writer.save_text(
                "pages",
                f"{index:02d}_{title}.html",
                wrap_html(title, content_html),
                source_url=detail.url or detail_url,
                status=detail.status,
                content_type="text/html; charset=utf-8",
            )
        attach_url = base + "/u/file/queryAttachmentList"
        attach_body = {"ownerGuid": guid, "publishTime": None, "isYingCangFuJian": None}
        attach = client.fetch(
            attach_url,
            method="POST",
            body=json.dumps(attach_body).encode("utf-8"),
            headers=headers,
        )
        attach_payload = json.loads(decode_text(attach))
        save_json(writer, "api", f"{index:02d}_{title}_attachments.json", attach_payload, attach.url or attach_url)
        for file_item in flatten_attachments(attach_payload.get("value")):
            file_id = file_item.get("fileId") or file_item.get("id") or file_item.get("guid")
            if not file_id:
                continue
            name = file_item.get("fileName") or file_item.get("name") or file_item.get("attachmentName") or f"{file_id}.bin"
            download_url = base + "/u/file/downloadFile?" + urllib.parse.urlencode({"fileId": file_id})
            download_file(client, writer, download_url, name_hint=name)
    writer.add_note(f"yunnan API completed: {len(items)} details saved")


def run_easyjcx(site_key: str, config: dict[str, Any], writer: ArtifactWriter, limit: int) -> None:
    client = HttpClient()
    headers = {
        "Origin": "https://www.easyjcx.com",
        "Referer": "https://www.easyjcx.com/",
        "Content-Type": "application/json;charset=UTF-8",
        "Accept": "application/json, text/plain, */*",
    }
    list_url = "https://api.easyjcx.com/api/portalCli/home/purchase"
    body = {
        "keyword": "",
        "current": 1,
        "size": limit,
        "condition": {"procurementType": "", "sourceType": "", "collegeNameList": [], "selectType": ""},
    }
    result = client.fetch(list_url, method="POST", body=json.dumps(body).encode("utf-8"), headers=headers)
    payload = json.loads(decode_text(result))
    save_json(writer, "api", "easyjcx_purchase_list.json", payload, result.url or list_url)
    for index, item in enumerate(payload.get("data", [])[:limit], start=1):
        order_id = item.get("orderMainId")
        if not order_id:
            continue
        detail_url = "https://api.easyjcx.com/api/portalCli/bidNeed/detail"
        detail_body = {"orderMainId": order_id}
        detail = client.fetch(detail_url, method="POST", body=json.dumps(detail_body).encode("utf-8"), headers=headers)
        detail_payload = json.loads(decode_text(detail))
        title = sanitize_filename(item.get("orderTitle") or f"easyjcx_detail_{index}")
        save_json(writer, "api", f"{index:02d}_{title}.json", detail_payload, detail.url or detail_url)
        writer.save_text(
            "pages",
            f"{index:02d}_{title}.html",
            render_easyjcx_detail(detail_payload.get("data", {})),
            source_url=f"https://www.easyjcx.com/#/purchase/detail/{order_id}",
            content_type="text/html; charset=utf-8",
        )
        for file_item in walk_file_items(detail_payload):
            url = normalize_easyjcx_file_url(file_item.get("ossPath") or "")
            if url:
                name = file_item.get("attachmentName") or filename_from_url(url, "easyjcx_attachment")
                download_file(client, writer, url, name_hint=name)
    writer.add_note(f"easyjcx API completed: {len(payload.get('data', [])[:limit])} details saved")


def run_kust(site_key: str, config: dict[str, Any], writer: ArtifactWriter, limit: int) -> None:
    client = HttpClient()
    url = "https://cgzx.kust.edu.cn/sfw_cms/e"
    headers = {
        "Referer": config["start_urls"][0],
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }
    data = {
        "window_": "json",
        "t_": str(time.time()),
        "request_method_": "ajax",
        "browser_version_": "126",
        "browser_": "notmsie",
        "page": "cms.psms.publish.query",
        "start": "1",
        "limit": str(limit),
        "filter": "",
        "sort": "begin_time desc",
        "filterfields": "",
        "type": ["ZCXQ", "YQXQ", "BYXQ", "CSXQ"],
        "isEnd": "",
        "catalog": "",
        "categoryId": ["104926"],
        "notShopType": ["DYLY", "WTDY", "DYWT"],
        "keywords": "",
    }
    body = urllib.parse.urlencode(data, doseq=True).encode("utf-8")
    result = client.fetch(url, method="POST", body=body, headers=headers)
    payload = json.loads(decode_text(result))
    save_json(writer, "api", "kust_purchase_list.json", payload, result.url or url)
    for index, item in enumerate(payload.get("resultset", [])[:limit], start=1):
        title = sanitize_filename(item.get("subject") or f"kust_item_{index}")
        if item.get("fullUrl"):
            page = client.fetch(item["fullUrl"], headers={"Referer": config["start_urls"][0]})
            writer.save_bytes(
                "pages",
                f"{index:02d}_{title}.html",
                page.content,
                source_url=page.url or item["fullUrl"],
                status=page.status,
                note=page.error,
                content_type=page.headers.get("content-type", ""),
            )
        attach = item.get("attach")
        if attach:
            attach_url = f"https://cgzx.kust.edu.cn/sfw_cms/rm/s/base/{attach}"
            download_file(client, writer, attach_url, name_hint=f"{title}.bin")
    writer.add_note(f"kust API completed: {len(payload.get('resultset', [])[:limit])} records saved")


def parse_jsonp(text: str) -> dict[str, Any]:
    match = re.search(r"^[^(]*\((.*)\)\s*;?\s*$", text, re.S)
    if not match:
        raise ValueError("response is not JSONP")
    return json.loads(match.group(1))


def wrap_html(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title></head><body>{body}</body></html>"
    )


def render_easyjcx_detail(data: dict[str, Any]) -> str:
    title = data.get("orderTitle") or "easyjcx detail"
    rows = []
    for key in ["collegeName", "orderCode", "budget", "startBidtime", "endBidtime", "deliverPlace", "supplierReqs"]:
        rows.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(data.get(key, '')))}</td></tr>")
    details = data.get("detailList") or []
    detail_html = ""
    for item in details:
        detail_html += "<h3>" + html.escape(str(item.get("deviceName", ""))) + "</h3>"
        detail_html += "<pre>" + html.escape(str(item.get("deviceSpec", ""))) + "</pre>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title></head><body>"
        f"<h1>{html.escape(title)}</h1><table>{''.join(rows)}</table>{detail_html}</body></html>"
    )


def flatten_attachments(value: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for v in value.values():
            items.extend(flatten_attachments(v))
        if any(k in value for k in ("fileId", "fileName", "attachmentName")):
            items.append(value)
    elif isinstance(value, list):
        for v in value:
            items.extend(flatten_attachments(v))
    return items


def walk_file_items(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "ossPath" in value or "attachmentName" in value:
            found.append(value)
        for v in value.values():
            found.extend(walk_file_items(v))
    elif isinstance(value, list):
        for v in value:
            found.extend(walk_file_items(v))
    return found


def normalize_easyjcx_file_url(path: str) -> str:
    if not path:
        return ""
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if path.startswith("//"):
        return "https:" + path
    if path.startswith("/"):
        return "https://api.easyjcx.com/alioss" + path
    return "https://api.easyjcx.com/alioss/" + path


def run_one(site_key: str, *, output_dir: Path, limit: int) -> Path:
    if site_key not in SITES:
        raise KeyError(f"unknown site: {site_key}")
    config = SITES[site_key]
    writer = ArtifactWriter(output_dir, site_key, config["name"])
    writer.add_note(f"{config['province']} - {config['name']}: {config.get('notes', '')}")
    adapter = config["adapter"]
    if adapter == "static":
        run_static(site_key, config, writer, limit)
    elif adapter == "probe":
        run_probe(site_key, config, writer, limit)
    elif adapter == "jilin_ggzy":
        run_jilin_ggzy(site_key, config, writer, limit)
    elif adapter == "yunnan_ggzy":
        run_yunnan_ggzy(site_key, config, writer, limit)
    elif adapter == "easyjcx":
        run_easyjcx(site_key, config, writer, limit)
    elif adapter == "kust":
        run_kust(site_key, config, writer, limit)
    else:
        raise ValueError(f"unsupported adapter: {adapter}")
    return writer.finish()


def build_parser(default_site: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a few validation samples from assigned procurement sites.")
    parser.add_argument("--site", default=default_site, help="site key, or 'all'. Use --list-sites to inspect keys.")
    parser.add_argument("--limit", type=int, default=3, help="number of sample records/files per site")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "downloads",
        help="output directory",
    )
    parser.add_argument("--list-sites", action="store_true", help="print site keys and exit")
    return parser


def main(argv: list[str] | None = None, *, default_site: str | None = None) -> int:
    parser = build_parser(default_site)
    args = parser.parse_args(argv)
    if args.list_sites:
        for key, cfg in SITES.items():
            print(f"{key:18s} {cfg['province']} {cfg['name']} [{cfg['adapter']}]")
        return 0
    if not args.site:
        parser.error("--site is required unless wrapper script provides one")
    site_keys = list(SITES) if args.site == "all" else [args.site]
    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for key in site_keys:
        print(f"\n=== {key} ===")
        try:
            run_one(key, output_dir=args.out, limit=max(1, args.limit))
        except Exception as exc:
            failures += 1
            print(f"FAILED {key}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
