"""Normalize public tender notices while retaining every source field."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup


# Excel columns are deliberately stable across sites and repeated runs.
COLUMNS = [
    "公告ID", "公告标题", "任务省份", "项目地区", "来源网站", "发布时间", "采购方式",
    "项目名称", "项目编号", "交易项目编号", "公告类型", "预算金额_元", "最高限价_元",
    "预算金额原文", "最高限价原文", "币种", "采购人名称", "采购人地址", "采购人联系人",
    "采购人联系方式", "代理机构名称", "代理机构地址", "代理机构联系人", "代理机构联系方式",
    "项目联系人", "项目联系电话", "获取文件开始时间", "获取文件结束时间", "获取文件时间原文",
    "获取文件地点", "获取文件方式", "文件售价_元", "文件售价原文", "报名截止时间",
    "投标截止时间", "开标时间", "竞价开始时间", "竞价结束时间", "投标地点", "开标地点", "采购需求", "合同履约期限",
    "是否接受联合体", "资格要求", "政府采购政策", "特定资格要求", "公告期限",
    "保证金信息", "其他补充事宜", "交货时间", "交货地点", "付款方式", "发票类型",
    "安装要求", "评审方法", "采购品目", "公开邮箱", "质量要求", "质保期", "更正事项", "更正内容", "首次公告日期",
    "采购明细行数", "附件数量", "正文字符数",
    "采集状态", "字段备注", "来源URL", "详情接口URL", "采集时间", "原始文件", "地区代码", "数据来源方式",
]
DATE_COLUMNS = {"发布时间", "获取文件开始时间", "获取文件结束时间", "报名截止时间", "投标截止时间", "开标时间", "采集时间", "首次公告日期", "竞价开始时间", "竞价结束时间"}
NUMBER_COLUMNS = {"预算金额_元", "最高限价_元", "文件售价_元", "采购明细行数", "附件数量", "正文字符数"}
DATE_RE = re.compile(r"(20\d{2})\s*[年./-]\s*(\d{1,2})\s*[月./-]\s*(\d{1,2})\s*日?(?:\s*(上午|下午|中午|晚上)?\s*(\d{1,2})[:时](\d{2})(?:[:分](\d{2}))?)?")
SECTION_RE = re.compile(r"(?m)^\s*[一二三四五六七八九十]+[、.．]\s*([^\n]{2,70})")
BLOCK_TAGS = ["p", "div", "section", "h1", "h2", "h3", "h4", "li", "tr", "br", "pre"]


def clean(value: Any) -> str:
    return re.sub(r"[ \t\r\f\v\u3000\xa0]+", " ", str(value or "")).strip()


def html_text(markup: str) -> str:
    soup = BeautifulSoup(markup or "", "html.parser")
    for node in soup.select("script,style,noscript,iframe"):
        node.decompose()
    for node in soup.find_all(BLOCK_TAGS):
        node.insert_before("\n")
        node.insert_after("\n")
    for node in soup.find_all(["td", "th"]):
        node.insert_after("\t")
    return "\n".join(clean(line) for line in soup.get_text().splitlines() if clean(line))


def pdf_paragraphs(text: str) -> str:
    """Join PDF layout wraps, keeping numbered items and explicit labels separate."""
    lines = []
    boundary = re.compile(r"^(?:[一二三四五六七八九十]+[、．.]|\d+(?:\.\d+)*[.、．]|[\u4e00-\u9fff ]{1,18}[:：])")
    for line in text.splitlines():
        line = clean(line)
        if not line:
            continue
        if lines and not boundary.match(line) and not re.search(r"[。；;：:]$", lines[-1]):
            lines[-1] += line
        else:
            lines.append(line)
    return "\n".join(lines)


def dates(value: str) -> list[str]:
    result = []
    value = clean(value)
    if re.fullmatch(r"20\d{12}", value):
        try:
            return [datetime.strptime(value, "%Y%m%d%H%M%S").isoformat(sep=" ")]
        except ValueError:
            return []
    for match in DATE_RE.finditer(value):
        y, m, d, period, hh, mm, ss = match.groups()
        try:
            hour = int(hh or 0)
            if period in ("下午", "晚上", "中午") and hour < 12:
                hour += 12
            parsed = datetime(int(y), int(m), int(d), hour, int(mm or 0), int(ss or 0))
            result.append(parsed.isoformat(sep=" ") if hh else parsed.date().isoformat())
        except ValueError:
            continue
    return result


def money(value: str) -> float | None:
    """Only normalize unambiguous, explicitly denominated amounts."""
    value = clean(value).replace(",", "").replace("，", "")
    if value in ("免费", "免费获取", "无偿获取", "不收费"):
        return 0.0
    if not value or re.search(r"保密|不公开|未公开|另行|详见|不超过.*%", value):
        return None
    # Multiple package amounts must not be mistaken for a project total.
    tokens = re.findall(r"(?<![A-Za-z\d])\d+(?:\.\d+)?", value)
    if len(tokens) != 1:
        return None
    if "元" not in value and not any(c in value for c in "￥¥"):
        return None
    multiplier = Decimal("100000000") if "亿元" in value else Decimal("10000") if "万元" in value else Decimal(1)
    try:
        return float((Decimal(tokens[0]) * multiplier).quantize(Decimal("0.01")))
    except InvalidOperation:
        return None


def label_value(text: str, *labels: str) -> str:
    for label in labels:
        pattern = r"\s*".join(re.escape(c) for c in label)
        match = re.search(pattern + r"\s*([（(][^）)\n]{0,12}[）)])?\s*[:：]\s*([^\n]+)", text)
        if match:
            value = clean(match.group(2))
            # Adjacent labels sometimes share a table row.
            value = re.split(r"\s+(?:项目编号|项目名称|地址|联系方式|联系人|电话)\s*[:：]", value)[0]
            unit = match.group(1) or ""
            return value + (" " + unit if unit and "元" in unit else "")
    return ""


def section_map(text: str) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(text))
    return {m.group(1).strip(): text[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
            for i, m in enumerate(matches)}


def section(sections: dict[str, str], *words: str) -> str:
    return next((value for key, value in sections.items() if any(word in key for word in words)), "")


def between(text: str, start: str, end: str) -> str:
    m = re.search(start, text)
    if not m:
        return ""
    tail = text[m.end():]
    stop = re.search(end, tail)
    return tail[:stop.start() if stop else len(tail)].strip(" :：\n")


def select_content(soup: BeautifulSoup) -> Any:
    for selector in ["#BodyLabel", "#template-center-page", ".ewb-article-info", ".newsCon",
                     "#ctl00_ContentPlaceHolder1_divContent", "#zoom", ".ewb-article-content",
                     ".article-content", "#mainText", "article", ".detail-content"]:
        node = soup.select_one(selector)
        if node and len(node.get_text(strip=True)) > 100:
            return node
    return soup.body or soup


def flatten(value: Any, path: str = ""):
    """Retain nulls, empty containers and indexed nested fields without collisions."""
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from flatten(child, f"{path}.{key}" if path else key)
    elif isinstance(value, list) and value:
        for index, child in enumerate(value):
            yield from flatten(child, f"{path}[{index}]")
    else:
        yield path, value


def parse_notice(site_key: str, config: dict, url: str, markup: str, *, title: str = "",
                 data: dict | None = None, collected_at: str = "", raw_file: str = "",
                 api_url: str = "") -> dict:
    data = data or {}
    soup = BeautifulSoup(markup, "html.parser")
    meta = {x.get("name") or x.get("property"): x.get("content", "") for x in soup.select("meta[content]")}
    content = select_content(soup)
    text = html_text(str(content))
    headings = [clean(x.get_text()) for x in soup.select("h1,h2,.article-title,.ewb-article-title,.div-title>h3")]
    title = (data.get("bulletintitle") or data.get("orderTitle") or data.get("subject") or data.get("title") or
             meta.get("ArticleTitle") or next((h for h in headings if len(h) > 12), "") or title)
    if not title and soup.title:
        title = soup.title.get_text(strip=True)
    record = {key: None for key in COLUMNS}
    record.update({"公告ID": hashlib.sha256((site_key + "|" + url).encode()).hexdigest()[:20],
                   "公告标题": clean(title), "任务省份": config["province"], "来源网站": config["name"],
                   "来源URL": url, "详情接口URL": api_url or None, "采集时间": collected_at,
                   "原始文件": raw_file, "采集状态": "已采集", "正文字符数": len(text), "数据来源方式": "本次在线采集"})
    notes = []
    sections = section_map(text)
    basic = section(sections, "项目基本", "项目概况") or text
    record["项目名称"] = label_value(basic, "项目名称", "电子反拍名称") or data.get("purchaseprojectname") or data.get("tenderName") or data.get("orderTitle")
    record["项目编号"] = label_value(basic, "项目编号", "采购项目编号", "采购编号", "电子反拍编号") or data.get("purchaseprojectcode") or data.get("orderCode") or data.get("tenderNo") or data.get("openTenderCode")
    full_page_text = html_text(str(soup))
    record["交易项目编号"] = label_value(full_page_text, "交易项目编号") or data.get("unifieddealcode")
    record["采购方式"] = label_value(basic, "采购方式") or data.get("shopTypeName") or next((x for x in ["竞争性磋商", "竞争性谈判", "公开招标", "询价", "单一来源", "反拍", "竞价"] if x in title), None)
    if not record["采购方式"] and site_key == "yunnan_easyjcx":
        record["采购方式"] = "竞价"
    if not record["采购方式"] and "[公开]" in title:
        record["采购方式"] = "公开招标"
    record["公告类型"] = "更正公告" if re.search("更正|变更", title) else "采购公告"
    body_start = text.splitlines()[0] if text else ""
    page_header = full_page_text.split(body_start)[0] if body_start else ""
    record["发布时间"] = next(iter(dates(str(data.get("bulletinstarttime") or data.get("timestamp") or data.get("beginTime") or data.get("noticeTime") or data.get("list_publish_date") or label_value(full_page_text, "发布时间", "发布日期") or meta.get("PubDate") or meta.get("pubdate") or page_header))), None)
    for target, labels in [("预算金额原文", ("预算金额", "项目预算")), ("最高限价原文", ("最高限价", "采购最高限价"))]:
        record[target] = label_value(basic, *labels) or None
        record[target.replace("原文", "_元")] = money(record[target])
    record["币种"] = data.get("usingCorrency") or ("人民币" if "人民币" in text or record["预算金额_元"] is not None else None)
    if site_key == "yunnan_easyjcx":
        record["预算金额_元"] = data.get("budget") if data.get("usingCorrency") == "人民币" else None
        record["预算金额原文"] = str(data["budget"]) + " 元" if record["预算金额_元"] is not None else None
        if data.get("budgetIsOpen") == "N":
            notes.append("预算未公开")
    record["采购需求"] = between(basic, r"采购需求\s*[:：]", r"合同履[行约]期限|本项目.{0,20}联合体") or None
    record["合同履约期限"] = label_value(basic, "合同履约期限", "合同履行期限") or data.get("contractLimit")
    union = re.search(r"(?:本项目|本标项|本采购包)[^\n]{0,50}接受联合体(?:投标|竞标)?", text)
    if union:
        record["是否接受联合体"] = "否" if re.search(r"不[）) ]*接受|否", union.group()) else "是" if re.search(r"接受", union.group()) else None
    record["资格要求"] = section(sections, "资格要求", "资格条件") or data.get("supplierReqs")
    record["政府采购政策"] = between(record["资格要求"] or "", r"落实政府采购政策需满足的资格要求\s*[:：]", r"3\s*[.、．]|本项目的特定资格") or None
    record["特定资格要求"] = between(record["资格要求"] or "", r"(?:本项目的)?特定资格要求\s*[:：]", r"$^") or None
    obtain = section(sections, "获取招标文件", "获取采购文件", "获取竞争性", "招标文件的获取")
    record["获取文件时间原文"] = between(obtain, r"时间\s*[:：]", r"(?:\d+\s*[.、．]\s*)?(?:获取)?(?:地点|方式)\s*[:：]") or None
    ds = dates(record["获取文件时间原文"] or obtain)
    for key, val in zip(["获取文件开始时间", "获取文件结束时间"], ds[:2]):
        record[key] = val
    record["获取文件地点"] = label_value(obtain, "地点") or data.get("biddocreferaddress")
    record["获取文件方式"] = between(obtain, r"方式\s*[:：]", r"售价\s*[:：（(]") or None
    record["文件售价原文"] = label_value(obtain, "售价", "文件售价") or None
    if not record["文件售价原文"]:
        sale = re.search(r"售价(?:人民币)?\s*[¥￥]?\s*\d+(?:\.\d+)?\s*(?:元)?", obtain)
        record["文件售价原文"] = sale.group() if sale else None
    record["文件售价_元"] = money(record["文件售价原文"])
    submit = section(sections, "提交投标", "响应文件提交", "投标文件的递交")
    opening = section(sections, "响应文件开启", "开标时间", "开启")
    deadline_line = next((line for line in submit.splitlines() if "截止时间" in line), "")
    deadline_value = label_value(submit, "提交响应文件截止时间及谈判开始时间", "提交投标文件截止时间", "截止时间")
    record["投标截止时间"] = next(iter(dates(deadline_line) or dates(deadline_value) or dates(submit) or dates(label_value(text, "询价材料递交截止时间"))), None)
    record["开标时间"] = next(iter(dates(opening)), None) or next(iter(dates(str(data.get("openTime") or ""))), None)
    record["投标地点"] = label_value(submit, "投标地点", "地点") or None
    record["开标地点"] = label_value(opening, "开标地点", "地点") or data.get("bidopeningaddress")
    record["报名截止时间"] = next(iter(dates(label_value(text, "报名截止时间"))), None)
    for field, api_key in [("投标截止时间", "bidclosingtime"), ("获取文件结束时间", "endTime"), ("投标截止时间", "endBidtime"), ("竞价开始时间", "startBidtime"), ("竞价结束时间", "endBidtime")]:
        if api_key == "endTime" and not obtain:
            continue
        candidate = next(iter(dates(str(data.get(api_key) or ""))), None)
        if candidate:
            if record[field] and len(record[field]) == 10 and candidate.startswith(record[field]):
                record[field] = candidate
            elif record[field] and record[field] != candidate and record[field][:16] != candidate[:16]:
                notes.append(f"{field}正文与接口{api_key}不一致，采用正文；接口值{candidate}")
            elif not record[field]:
                record[field] = candidate
    record["公告期限"] = section(sections, "公告期限") or None
    record["保证金信息"] = section(sections, "保证金") or None
    record["其他补充事宜"] = section(sections, "其他补充") or data.get("remark")
    contact = section(sections, "联系", "询问") or text
    buyer = between(contact, r"1\s*[.、．]\s*采购人(?:信息)?", r"2\s*[.、．]\s*采购代理")
    agent = between(contact, r"2\s*[.、．]\s*采购代理机构(?:信息)?", r"(?:3\s*[.、．]\s*)?项目联系方式")
    project_contact = between(contact, r"(?:3\s*[.、．]\s*)?项目联系方式", r"附件信息|相关链接")
    for prefix, part in [("采购人", buyer), ("代理机构", agent)]:
        record[prefix + "名称"] = label_value(part, "名称") or None
        record[prefix + "地址"] = label_value(part, "联系地址", "地址") or None
        record[prefix + "联系人"] = label_value(part, "联系人") or None
        record[prefix + "联系方式"] = label_value(part, "联系方式", "联系电话", "电话") or None
    record["采购人名称"] = record["采购人名称"] or data.get("collegeName") or label_value(text, "采购人", "发布单位") or ("昆明理工大学" if site_key == "yunnan_kust" else None)
    record["代理机构名称"] = record["代理机构名称"] or data.get("agentName")
    record["项目联系人"] = label_value(project_contact, "项目联系人", "联系人") or label_value(text, "项目联系人") or None
    record["项目联系电话"] = label_value(project_contact, "电话", "联系方式") or None
    if site_key == "yunnan_kust" and not buyer:
        record["采购人联系人"] = re.split(r"[，,]\s*(?:座机号|电话|邮箱)", label_value(text, "联系人"))[0] or None
        record["采购人联系方式"] = re.split(r"[，,]\s*邮箱", label_value(text, "座机号", "联系电话"))[0] or None
        record["投标地点"] = record["投标地点"] or label_value(text, "询价材料递交方式") or None
    if site_key == "guangdong_gdegp":
        record["采购人联系人"] = label_value(text, "联系人") or None
        record["采购人联系方式"] = label_value(text, "联系电话") or None
        record["项目名称"] = record["项目名称"] or label_value(text, "电子反拍名称")
        record["项目编号"] = label_value(text, "电子反拍编号", "采购编号") or record["项目编号"]
        record["投标截止时间"] = next(iter(dates(label_value(text, "一轮竞价结束时间"))), None)
        record["评审方法"] = label_value(text, "确定成交供应商原则") or None
    for target, key in [("交货时间", "deliverTime"), ("交货地点", "deliverPlace"), ("付款方式", "payType"), ("发票类型", "invoiceType"), ("安装要求", "installations"), ("采购品目", "deviceType")]:
        record[target] = data.get(key) or label_value(text, target) or None
    for target in ["质量要求", "质保期", "更正事项"]:
        record[target] = label_value(text, target) or None
    record["公开邮箱"] = "；".join(dict.fromkeys(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))) or None
    record["采购品目"] = record["采购品目"] or data.get("catalogueNameList")
    record["更正内容"] = between(section(sections, "更正信息"), r"更正内容\s*[:：]", r"更正日期") or None
    record["首次公告日期"] = next(iter(dates(label_value(text, "首次公告日期"))), None)
    for target in ["项目编号", "采购方式"]:
        if record[target]:
            record[target] = record[target].rstrip("。； ")
    record["项目地区"] = data.get("secondAreaName") or data.get("regionName") or None
    record["地区代码"] = data.get("xiaqucode") or data.get("areaCode") or data.get("regionCode") or None
    if data.get("deliverPlace"):
        record["项目地区"] = "/".join(data["deliverPlace"].split("/")[:3])
    if site_key == "yunnan_easyjcx" and record["项目地区"] and not record["项目地区"].startswith("云南"):
        notes.append("全国性平台：任务省份不代表项目所在地")
    tables = []
    for ti, table in enumerate(content.find_all("table"), 1):
        rows = [[html_text(str(cell)) for cell in tr.find_all(["td", "th"], recursive=False)] for tr in table.find_all("tr")]
        tables.append({"table": ti, "rows": [row for row in rows if row]})
    attachments = []
    for a in soup.select("a[href]"):
        href = urljoin(url, a["href"])
        name = a.get_text(" ", strip=True) or a.get("title") or ""
        if href.startswith(("http://", "https://")) and re.search(r"\.(?:pdf|docx?|xlsx?|zip|rar|7z|txt)(?:[?#]|$)|attachment|download|/file/|/oss/", href, re.I):
            if not any(x["url"] == href for x in attachments):
                attachments.append({"name": name or href.rsplit("/", 1)[-1], "url": href, "status": "仅采集链接"})
    items = data.get("detailList") or []
    if items:
        record["采购需求"] = "\n\n".join(str(x.get("deviceName", "")) + "\n" + str(x.get("deviceSpec", "")) for x in items)
    record["采购明细行数"] = len(items)
    record["附件数量"] = len(attachments)
    if not text or len(text) < 100:
        record["采集状态"] = "仅列表字段"
        notes.append("详情正文未取得")
    if not record["发布时间"]:
        notes.append("页面未识别到发布时间")
    record["字段备注"] = "；".join(notes) or None
    raw = {"api": data, "html_meta": meta, "html_tables": tables, "sections": sections}
    return {"record": record, "text": text, "raw": raw, "items": items, "attachments": attachments}
