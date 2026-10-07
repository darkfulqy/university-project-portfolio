"""Download, inspect and safely unpack public procurement attachments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

from pypdf import PdfReader

from download_site import USER_AGENT, sanitize_filename
from network_route import open_url, curl_route_args

GIB = 1024 ** 3


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_format(path):
    with open(path, "rb") as stream:
        head = stream.read(8192)
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"PK") and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
        if "word/document.xml" in names:
            return "docx"
        if "xl/workbook.xml" in names:
            return "xlsx"
        return "zip"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "ole"
    if head.startswith(b"Rar!\x1a\x07"):
        return "rar"
    if head.startswith(b"7z\xbc\xaf\x27\x1c"):
        return "7z"
    if head.lstrip().lower().startswith(b"{\\rtf"):
        return "rtf"
    if b"wordDocument" in head or b"mso-" in head:
        return "word_html"
    if re.search(br"<!doctype\s+html|<html|<head|<script", head, re.I) or head.lstrip().startswith((b'{"', b"[{")):
        raise ValueError("返回了网页或JSON，不能作为标书文件")
    if not head:
        raise ValueError("附件为空")
    return "bin"


def classify(name, text, pages=None):
    compact = re.sub(r"\s+", "", text)
    tender = re.compile(r"招标文件|采购文件|竞争性(?:磋商|谈判)文件|磋商文件|询价文件")
    if re.search("承诺函|评价指标|代理协议|编制说明|工程量清单", name):
        return "其他附件", "文件名称表明为配套表单、协议或清单"
    if "采购需求" in name or "询价表" in name:
        return "采购需求/询价表", "需求或报价表附件，不因引用标书条款而视为完整标书"
    if re.search(r"公告|采购询价函", name) and not tender.search(name):
        return "公告", "文件名称为公告；不计为完整标书"
    if pages is not None and pages <= 6 and re.search(r"招标公告|磋商公告|谈判公告|采购询价函", compact[:1500]):
        return "公告", "短篇正文为公告或询价函"
    structure = [term for term in ("投标人须知", "供应商须知", "磋商须知", "响应文件格式", "投标文件格式", "评标办法", "评审方法", "评审标准", "合同条款", "合同格式", "合同协议书") if term in compact]
    if tender.search(compact[:20000]) and len(structure) >= 2:
        return "完整标书（结构识别）", "正文含标书标题及" + "、".join(structure)
    if tender.search(name) and pages and pages >= 15 and all(term in compact for term in ("第一章", "第二章")):
        return "完整标书（结构识别）", "标书文件名，PDF不少于15页，检查文本含第一章及第二章"
    if tender.search(name) or tender.search(compact[:800]):
        return "标书候选", "名称或首页含标书关键词，完整性待核对"
    return "其他附件", "未识别到完整标书结构"


class TenderFiles:
    def __init__(self, root: Path, *, min_free_gb=10, max_file_mb=1024, interval=1.0):
        self.root = root
        self.min_free = min_free_gb * GIB
        self.max_bytes = max_file_mb * 1024 * 1024
        self.interval = interval
        self.last_request = 0.0
        (root / "objects").mkdir(parents=True, exist_ok=True)
        (root / "tmp").mkdir(exist_ok=True)

    def check_space(self):
        if shutil.disk_usage(self.root).free < self.min_free:
            raise OSError("剩余磁盘空间低于保留值，采集已停止；释放空间后可续采")

    def download(self, url, referer=""):
        self.check_space()
        fd, temporary = tempfile.mkstemp(dir=self.root / "tmp", suffix=".part")
        os.close(fd)
        target = Path(temporary)
        headers = {"User-Agent": USER_AGENT, "Referer": referer, "Accept": "*/*"}
        time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
        try:
            try:
                request = urllib.request.Request(url, headers=headers)
                with open_url(request, timeout=45, context=ssl._create_unverified_context()) as response:
                    if int(response.headers.get("Content-Length") or 0) > self.max_bytes:
                        raise ValueError("附件超过单文件大小上限")
                    with target.open("wb") as out:
                        size = 0
                        while chunk := response.read(1024 * 1024):
                            size += len(chunk)
                            if size > self.max_bytes:
                                raise ValueError("附件超过单文件大小上限")
                            self.check_space()
                            out.write(chunk)
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, ssl.SSLError, TimeoutError):
                # Some provincial servers negotiate TLS only with system curl.
                proc = subprocess.run(["curl", "-k", "-L", "--fail", "--silent", "--show-error", "--max-time", "180",
                    "--max-filesize", str(self.max_bytes), "-A", USER_AGENT, "-e", referer, "-o", str(target), *curl_route_args(url), url],
                    capture_output=True, timeout=190)
                if proc.returncode:
                    raise ValueError("附件请求失败: " + proc.stderr.decode("utf-8", "replace")[-350:])
            file_format(target)
            return target
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            self.last_request = time.monotonic()

    def ingest(self, source: Path, *, notice, name, url, parent_id="", member="", depth=0):
        self.check_space()
        rid = notice["record"]["公告ID"]
        kind = file_format(source)
        suffix = {"ole": Path(name).suffix.lower().lstrip(".") or "doc", "word_html": "doc"}.get(kind, kind)
        digest = sha256(source)
        folder = self.root / "objects" / digest[:2]
        folder.mkdir(exist_ok=True)
        obj = folder / (digest + "." + suffix)
        if not obj.exists():
            fd, object_temp = tempfile.mkstemp(dir=folder, suffix=".tmp")
            os.close(fd)
            try:
                shutil.copyfile(source, object_temp)
                os.replace(object_temp, obj)
            finally:
                Path(object_temp).unlink(missing_ok=True)
        doc_id = hashlib.sha256((rid + "|" + digest + "|" + parent_id + "|" + member).encode()).hexdigest()[:24]
        project = self.root / "projects" / rid
        project.mkdir(parents=True, exist_ok=True)
        filename = sanitize_filename(name)
        if Path(filename).suffix.lower() != "." + suffix:
            filename += "." + suffix
        # Filesystem limits are byte limits, especially important for Chinese titles.
        filename = Path(filename).stem.encode("utf-8")[:220].decode("utf-8", "ignore") + Path(filename).suffix
        file = project / (doc_id[:8] + "_" + filename)
        if not file.exists():
            try:
                os.link(obj, file)
            except OSError:
                shutil.copyfile(obj, file)
        text, pages, extraction_error = "", None, ""
        try:
            if kind == "pdf":
                reader = PdfReader(obj)
                pages = len(reader.pages)
                # Inspection, not OCR: retain the untouched original document.
                text = "\n".join(reader.pages[i].extract_text() or "" for i in range(min(pages, 8)))
            elif kind == "docx":
                with zipfile.ZipFile(obj) as archive:
                    tree = ET.fromstring(archive.read("word/document.xml"))
                text = "\n".join(p.text or "" for p in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
            elif suffix in ("doc", "rtf") and shutil.which("textutil"):
                proc = subprocess.run(["textutil", "-convert", "txt", "-stdout", str(file)], capture_output=True, timeout=30)
                if proc.returncode == 0:
                    text = proc.stdout.decode("utf-8", "replace")
                else:
                    extraction_error = "Word文本提取失败，原文件已保留"
        except Exception as exc:
            extraction_error = str(exc)[:300]
        category, evidence = classify(name, text, pages)
        if kind in ("zip", "rar", "7z"):
            category, evidence = "压缩包", "保留原始包；内部文件另行建立关联"
        snippet_path = ""
        if text:
            snippet_path = str(project / (doc_id + "_检查文本.txt"))
            Path(snippet_path).write_text(text, encoding="utf-8")
        code = re.sub(r"\W", "", notice["record"].get("项目编号") or "").lower()
        code_match = "匹配" if code and code in re.sub(r"\W", "", text).lower() else "未在检查文本中匹配" if text and code else "未核验"
        document = {"document_id": doc_id, "notice_id": rid, "name": name, "url": url, "file": str(file),
                    "sha256": digest, "format": suffix, "category": category, "pages": pages,
                    "parent_id": parent_id, "archive_member": member, "bytes": obj.stat().st_size,
                    "status": "已保存", "evidence": evidence, "project_code_check": code_match,
                    "inspection_text": snippet_path, "inspection_error": extraction_error}
        documents = [document]
        if kind == "zip" and depth < 2:
            try:
                with zipfile.ZipFile(obj) as archive:
                    entries = archive.infolist()
                    if len(entries) > 5000 or sum(i.file_size for i in entries) > 4 * self.max_bytes:
                        raise ValueError("压缩包超过自动展开上限，原包保留")
                    for entry in entries:
                        member_path = PurePosixPath(entry.filename.replace("\\", "/"))
                        if entry.is_dir():
                            continue
                        if member_path.is_absolute() or ".." in member_path.parts or ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                            raise ValueError("压缩包包含不安全路径或符号链接")
                        if member_path.suffix.lower() not in (".pdf", ".doc", ".docx", ".rtf", ".xls", ".xlsx", ".zip", ".rar", ".7z", ".txt", ".csv"):
                            continue
                        if entry.file_size > self.max_bytes or (entry.compress_size and entry.file_size / entry.compress_size > 500):
                            raise ValueError("压缩成员超过展开大小或压缩率上限")
                        fd, temporary = tempfile.mkstemp(dir=self.root / "tmp")
                        os.close(fd)
                        try:
                            with archive.open(entry) as input_stream, open(temporary, "wb") as out:
                                shutil.copyfileobj(input_stream, out)
                            documents.extend(self.ingest(Path(temporary), notice=notice, name=member_path.name, url=url,
                                parent_id=doc_id, member=entry.filename, depth=depth + 1))
                        finally:
                            Path(temporary).unlink(missing_ok=True)
            except Exception as exc:
                document["inspection_error"] = "解包未完成: " + str(exc)
        elif kind in ("rar", "7z") and depth < 2:
            executable = shutil.which("bsdtar") or shutil.which("tar")
            try:
                if not executable:
                    raise ValueError("未安装支持RAR/7Z的libarchive/bsdtar")
                listing = subprocess.run([executable, "-tf", str(obj)], capture_output=True, timeout=30)
                if listing.returncode:
                    raise ValueError(listing.stderr.decode("utf-8", "replace")[:300])
                members = listing.stdout.decode("utf-8").splitlines()
                if len(members) > 5000:
                    raise ValueError("压缩包成员超过自动展开上限")
                expanded = 0
                for member_name in members:
                    member_path = PurePosixPath(member_name.replace("\\", "/"))
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise ValueError("压缩包包含不安全路径")
                    if member_name.endswith("/") or member_path.suffix.lower() not in (".pdf", ".doc", ".docx", ".rtf", ".xls", ".xlsx", ".txt", ".csv"):
                        continue
                    fd, temporary = tempfile.mkstemp(dir=self.root / "tmp")
                    os.close(fd)
                    process = subprocess.Popen([executable, "-xOf", str(obj), "--", member_name], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    timer = threading.Timer(60, process.kill)
                    timer.start()
                    try:
                        size = 0
                        with open(temporary, "wb") as out:
                            while chunk := process.stdout.read(1024 * 1024):
                                size += len(chunk)
                                expanded += len(chunk)
                                if size > self.max_bytes or expanded > self.max_bytes * 4:
                                    raise ValueError("压缩成员超过展开大小上限")
                                self.check_space()
                                out.write(chunk)
                        if process.wait(timeout=5):
                            raise ValueError("解压失败、加密或超时")
                        documents.extend(self.ingest(Path(temporary), notice=notice, name=member_path.name, url=url,
                                                     parent_id=doc_id, member=member_name, depth=depth + 1))
                    finally:
                        timer.cancel()
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                        process.stdout.close()
                        Path(temporary).unlink(missing_ok=True)
            except Exception as exc:
                document["inspection_error"] = "原始压缩包已保存，解包未完成: " + str(exc)
        atomic_json(project / (doc_id + ".json"), {"document": document, "announcement": notice["record"]})
        return documents
