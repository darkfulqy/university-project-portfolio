#!/usr/bin/env python3
"""Durable bulk collection of tender originals and associated notice fields."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from announcement_fields import COLUMNS, DATE_COLUMNS, NUMBER_COLUMNS
from collect_announcements import ROOT, export_excel
from download_site import SITES
from paged_collectors import PagedCollector, SUPPORTED
from tender_files import TenderFiles, atomic_json, sha256

DEFAULT_ROOT = ROOT.parent / "批量标书库"


def now():
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def transient_network_error(error):
    return bool(re.search(r"HTTP (?:0|5\d\d)\b|Could not resolve host|nodename nor servname|timed? out|Connection reset|UNEXPECTED_EOF", error, re.I))


def source_weight(site, notices=0, files=0):
    if notices >= 100 and files == 0:
        return 0.25
    return {"guangdong_ccgp": 3.0, "jilin_ggzy": 3.0, "xinjiang_ggzy": 3.0, "beijing_ccgp": 2.0}.get(site, 1.0)


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "index.sqlite3", timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS notices(id TEXT PRIMARY KEY, site TEXT, title TEXT, url TEXT, data TEXT, updated TEXT);
          CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, notice_id TEXT, sha256 TEXT, format TEXT, category TEXT, data TEXT);
          CREATE INDEX IF NOT EXISTS document_notice ON documents(notice_id);
          CREATE INDEX IF NOT EXISTS document_category ON documents(category,format);
          CREATE INDEX IF NOT EXISTS document_hash ON documents(sha256);
          CREATE TABLE IF NOT EXISTS downloads(url TEXT PRIMARY KEY, status TEXT, file TEXT, sha256 TEXT, error TEXT, attempts INTEGER DEFAULT 0);
          CREATE TABLE IF NOT EXISTS sites(site TEXT PRIMARY KEY, next_page INTEGER DEFAULT 1, status TEXT DEFAULT '待采集',
              signature TEXT DEFAULT '', source_total INTEGER, error TEXT DEFAULT '', updated TEXT);
          CREATE TABLE IF NOT EXISTS pages(site TEXT, page INTEGER, signature TEXT, detail_count INTEGER, list_count INTEGER,
              status TEXT, error TEXT, updated TEXT, PRIMARY KEY(site,page));
          CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
        """)
        existing = {row[1] for row in self.db.execute("PRAGMA table_info(sites)")}
        for column, declaration in [("before", "TEXT DEFAULT ''"), ("window_start", "INTEGER DEFAULT 1")]:
            if column not in existing:
                with self.db:
                    self.db.execute(f"ALTER TABLE sites ADD COLUMN {column} {declaration}")

    def register(self, keys):
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO sites(site,updated) VALUES(?,?)", [(k, now()) for k in keys])

    def retry_network(self, keys, cooldown=300):
        """Requeue transient failures; keep access restrictions and parser errors stopped."""
        waiting = False
        for row in list(self.db.execute("SELECT * FROM sites WHERE status IN ('访问/接口失败','详情失败待重试')")):
            if row["site"] not in keys or not transient_network_error(row["error"]):
                continue
            if (datetime.now() - datetime.fromisoformat(row["updated"])).total_seconds() < cooldown:
                waiting = True
                continue
            page = row["next_page"]
            partial = self.db.execute("SELECT page,error FROM pages WHERE site=? AND status='部分失败' AND page>=? AND page<? ORDER BY page",
                                      (row["site"], row["window_start"], page))
            page = next((p["page"] for p in partial if transient_network_error(p["error"])), page)
            prior = self.db.execute("SELECT signature FROM pages WHERE site=? AND page<? ORDER BY page DESC LIMIT 1", (row["site"], page)).fetchone()
            with self.db:
                self.db.execute("UPDATE sites SET next_page=?,status='待采集',signature=?,updated=? WHERE site=?",
                                (page, prior[0] if prior else "", now(), row["site"]))
        return waiting

    def put_notice(self, site, notice):
        r = notice["record"]
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO notices VALUES(?,?,?,?,?,?)",
                            (r["公告ID"], site, r["公告标题"], r["来源URL"], json.dumps(notice, ensure_ascii=False), now()))
        atomic_json(self.root / "projects" / r["公告ID"] / "announcement.json", notice)

    def put_documents(self, documents):
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO documents VALUES(?,?,?,?,?,?)", [
                (d["document_id"], d["notice_id"], d["sha256"], d["format"], d["category"], json.dumps(d, ensure_ascii=False)) for d in documents])

    def attach(self, notice, manager):
        for attachment in notice["attachments"]:
            if (self.root / "STOP").exists():
                break
            url = attachment["url"]
            old = self.db.execute("SELECT * FROM downloads WHERE url=?", (url,)).fetchone()
            candidate = Path(attachment.get("file") or "__missing__")
            if old and old["status"] == "已保存" and Path(old["file"]).exists():
                candidate = Path(old["file"])
            # Successful files are content-verified before reuse. Failed URLs have bounded retries.
            if old and old["status"] == "失败" and old["attempts"] >= 3 and not candidate.exists():
                continue
            temp = None
            try:
                if not candidate.is_file():
                    temp = manager.download(url, notice["record"]["来源URL"])
                    candidate = temp
                if old and old["status"] == "已保存" and sha256(candidate) != old["sha256"]:
                    raise ValueError("本地文件哈希不一致，需检查后重新下载")
                documents = manager.ingest(candidate, notice=notice, name=attachment["name"], url=url)
                self.put_documents(documents)
                doc = documents[0]
                with self.db:
                    self.db.execute("INSERT INTO downloads(url,status,file,sha256,error,attempts) VALUES(?,?,?,?,?,1) "
                                    "ON CONFLICT(url) DO UPDATE SET status=excluded.status,file=excluded.file,sha256=excluded.sha256,error='',attempts=attempts+1",
                                    (url, "已保存", doc["file"], doc["sha256"], ""))
                attachment.update(status="已下载", file=doc["file"], sha256=doc["sha256"], bytes=doc["bytes"], document_id=doc["document_id"], category=doc["category"])
            except OSError as exc:
                if "磁盘" in str(exc):
                    raise
                self.failed(url, str(exc))
                attachment.update(status="下载/检查失败", error=str(exc))
            except Exception as exc:
                self.failed(url, str(exc))
                attachment.update(status="下载/检查失败", error=str(exc))
            finally:
                if temp:
                    temp.unlink(missing_ok=True)

    def failed(self, url, error):
        with self.db:
            self.db.execute("INSERT INTO downloads(url,status,error,attempts) VALUES(?,?,?,1) "
                            "ON CONFLICT(url) DO UPDATE SET status='失败',error=excluded.error,attempts=attempts+1", (url, "失败", error[:1000]))

    def report(self):
        count = lambda sql: self.db.execute(sql).fetchone()[0]
        return {"updated_at": now(), "announcements": count("SELECT COUNT(*) FROM notices"),
                "documents": count("SELECT COUNT(*) FROM documents"),
                "unique_files": count("SELECT COUNT(DISTINCT sha256) FROM documents"),
                "pdf_files": count("SELECT COUNT(DISTINCT sha256) FROM documents WHERE format='pdf'"),
                "word_files": count("SELECT COUNT(DISTINCT sha256) FROM documents WHERE format IN ('doc','docx')"),
                "pdf_notices": count("SELECT COUNT(DISTINCT sha256) FROM documents WHERE format='pdf' AND category='公告'"),
                "full_tender_documents": count("SELECT COUNT(*) FROM documents WHERE category='完整标书（结构识别）'"),
                "full_tender_pdf": count("SELECT COUNT(*) FROM documents WHERE category='完整标书（结构识别）' AND format='pdf'"),
                "unique_tender_pdf": count("SELECT COUNT(DISTINCT sha256) FROM documents WHERE category='完整标书（结构识别）' AND format='pdf'"),
                "full_tender_projects": count("SELECT COUNT(DISTINCT notice_id) FROM documents WHERE category='完整标书（结构识别）'"),
                "failed_downloads": count("SELECT COUNT(*) FROM downloads WHERE status='失败'"),
                "sites": [dict(row) for row in self.db.execute("SELECT * FROM sites ORDER BY site")]}

    def snapshot(self, csv_files=False):
        report = self.report()
        atomic_json(self.root / "progress.json", report)
        if not csv_files:
            return report
        # Streaming CSV is immediately usable in Excel without a large in-memory workbook.
        for name, headers, query, transform in [
            ("公告字段.csv", COLUMNS, "SELECT data FROM notices ORDER BY site,id", lambda d: [d["record"].get(k) for k in COLUMNS]),
            ("标书文件索引.csv", ["文件ID", "公告ID", "文件名", "文件类型", "格式", "页数", "本地文件", "来源URL", "SHA256", "父压缩包文件ID", "包内路径", "项目编号核验", "识别依据", "检查备注"],
             "SELECT data FROM documents ORDER BY notice_id,id", lambda d: [d.get(k) for k in ["document_id", "notice_id", "name", "category", "format", "pages", "file", "url", "sha256", "parent_id", "archive_member", "project_code_check", "evidence", "inspection_error"]])]:
            path = self.root / name
            temp = path.with_suffix(".csv.tmp")
            with temp.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(headers)
                for row in self.db.execute(query):
                    values = transform(json.loads(row[0]))
                    writer.writerow([("'" + v) if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v for v in values])
            os.replace(temp, path)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES('csv_count',?)", (str(report["announcements"]),))
        return report


def crawl_page(root, key, page, page_size, interval, min_free):
    store = Store(root)
    manager = TenderFiles(store.root, min_free_gb=min_free, interval=interval)
    manager.check_space()
    if (store.root / "STOP").exists():
        store.db.close()
        return {"site": key, "page": page, "status": "中断未提交", "notices": 0}
    cursor = store.db.execute("SELECT * FROM sites WHERE site=?", (key,)).fetchone()
    collector = PagedCollector(key, store.root / "raw" / f"page_{page:06d}", page, page_size, interval,
                               before=cursor["before"], window_start=cursor["window_start"])
    notices, status = collector.run()
    if (store.root / "STOP").exists():
        store.db.close()
        return {"site": key, "page": page, "status": "中断未提交", "notices": len(notices)}
    for notice in notices:
        if (store.root / "STOP").exists():
            store.db.close()
            return {"site": key, "page": page, "status": "中断未提交", "notices": len(notices)}
        store.put_notice(key, notice)
        store.attach(notice, manager)
        store.put_notice(key, notice)
        if (store.root / "STOP").exists():
            store.db.close()
            return {"site": key, "page": page, "status": "中断未提交", "notices": len(notices)}
    previous = store.db.execute("SELECT signature FROM sites WHERE site=?", (key,)).fetchone()
    repeated = collector.signature and previous and collector.signature == previous[0] and page != cursor["window_start"]
    if key not in SUPPORTED:
        state = "入口受限或未适配"
    elif collector.list_count is None or collector.list_error:
        state = "访问/接口失败"
    elif repeated:
        state = "分页重复待检查"
    elif collector.list_count == 0:
        state = "公开列表已遍历"
    elif collector.errors and not notices:
        state = "详情失败待重试"
    else:
        state = "采集中"
    error = "；".join(collector.errors) or ("分页响应与上一页重复，未认定历史数据已采完" if repeated else "")
    with store.db:
        store.db.execute("INSERT OR REPLACE INTO pages VALUES(?,?,?,?,?,?,?,?)",
                         (key, page, collector.signature, len(notices), collector.list_count, "部分失败" if error and state == "采集中" else state, error, now()))
        store.db.execute("UPDATE sites SET next_page=?,status=?,signature=?,source_total=?,error=?,updated=? WHERE site=?",
                         (page + 1 if state == "采集中" else page, state,
                          collector.signature if state == "采集中" else (previous[0] if previous else ""), collector.source_total, error, now(), key))
        if key == "xinjiang_ggzy" and state == "采集中" and (page - cursor["window_start"] + 1) * page_size >= 9000:
            if collector.oldest_timestamp and collector.oldest_timestamp != cursor["before"]:
                # Include the boundary timestamp again; stable notice IDs remove overlap.
                store.db.execute("UPDATE sites SET before=?,window_start=? WHERE site=?", (collector.oldest_timestamp, page + 1, key))
            else:
                store.db.execute("UPDATE sites SET status='时间分段待检查',error='无法向更早的公开日期推进' WHERE site=?", (key,))
    store.db.close()
    return {"site": key, "page": page, "status": state, "notices": len(notices)}


def crawl(args):
    store = Store(args.root)
    keys = list(SITES) if args.site == "all" else args.site.split(",")
    if any(k not in SITES for k in keys):
        raise ValueError("未知站点代码")
    store.register(keys)
    lock = (store.root / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    saved_size = store.db.execute("SELECT value FROM settings WHERE key='page_size'").fetchone()
    if saved_size and int(saved_size[0]) != args.page_size:
        raise ValueError(f"此库断点使用page-size={saved_size[0]}，续采不能改变分页大小；新设置请使用新--root")
    with store.db:
        store.db.execute("INSERT OR REPLACE INTO settings VALUES('page_size',?)", (str(args.page_size),))
        store.db.execute("UPDATE sites SET status='待采集',error='' WHERE error LIKE '%收到停止请求%'")
    lock.write(str(os.getpid()))
    lock.flush()
    atomic_json(store.root / "worker.json", {"pid": os.getpid(), "started_at": now(), "scope": "不限年份，公开列表尽可能全量", "root": str(store.root)})
    (store.root / "STOP").unlink(missing_ok=True)
    def stop(*_):
        (store.root / "STOP").touch()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    served = defaultdict(int)
    completed = 0
    totals = {r["site"]: r["total"] for r in store.db.execute("SELECT site,COUNT(*) AS total FROM notices GROUP BY site")}
    file_totals = {r["site"]: r["total"] for r in store.db.execute("SELECT n.site,COUNT(DISTINCT d.notice_id) AS total FROM documents d JOIN notices n ON n.id=d.notice_id GROUP BY n.site")}
    weights = {k: source_weight(k, totals.get(k, 0), file_totals.get(k, 0)) for k in keys}
    priority = {"guangdong_ccgp": 0, "jilin_ggzy": 1, "xinjiang_ggzy": 2, "beijing_ccgp": 3, "yunnan_kust": 4}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            while not (store.root / "STOP").exists():
                eligible = [k for k in keys if not args.rounds or served[k] < args.rounds]
                waiting_for_network = store.retry_network(eligible)
                running = set(futures.values())
                rows = [row for row in store.db.execute("SELECT * FROM sites WHERE status IN ('待采集','采集中')")
                        if row["site"] in eligible and row["site"] not in running]
                rows.sort(key=lambda row: (served[row["site"]] / weights[row["site"]], priority.get(row["site"], 10)))
                for row in rows[:max(0, args.workers - len(futures))]:
                    key = row["site"]
                    futures[pool.submit(crawl_page, store.root, key, row["next_page"], args.page_size, args.interval, args.min_free_gb)] = key
                    served[key] += 1
                if not futures:
                    if waiting_for_network:
                        time.sleep(5)
                        continue
                    break
                # Refill a free worker immediately, without waiting for slower sites.
                done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    key = futures.pop(future)
                    try:
                        print(json.dumps(future.result(), ensure_ascii=False), flush=True)
                    except Exception as exc:
                        with store.db:
                            store.db.execute("UPDATE sites SET status='运行失败',error=?,updated=? WHERE site=?", (str(exc), now(), key))
                        print(f"{key}: {exc}", flush=True)
                        if "磁盘" in str(exc):
                            stop()
                    completed += 1
                    report = store.snapshot()
                    csv_count = store.db.execute("SELECT value FROM settings WHERE key='csv_count'").fetchone()
                    if report["announcements"] - int(csv_count[0] if csv_count else 0) >= 500:
                        store.snapshot(csv_files=True)
                    if completed % 10 == 0:
                        print(json.dumps({k: v for k, v in report.items() if k != "sites"}, ensure_ascii=False), flush=True)
    finally:
        store.snapshot(csv_files=True)
        atomic_json(store.root / "worker.json", {"pid": os.getpid(), "stopped_at": now(), "pages_completed": completed, "site_pages": dict(served)})
        store.db.close()
        lock.close()


def export(store, batch_size):
    store.snapshot(csv_files=True)
    # Capture every volume from one read snapshot while the crawler keeps writing.
    store.db.execute("BEGIN")
    jobs = []
    count = store.db.execute("SELECT COUNT(*) FROM notices").fetchone()[0]
    folder = store.root / "excel"
    folder.mkdir(exist_ok=True)
    for offset in range(0, count, batch_size):
        notices = [json.loads(r[0]) for r in store.db.execute("SELECT data FROM notices ORDER BY site,id LIMIT ? OFFSET ?", (batch_size, offset))]
        docs = []
        for n in notices:
            docs.extend(json.loads(row[0]) for row in store.db.execute("SELECT data FROM documents WHERE notice_id=?", (n["record"]["公告ID"],)))
        sites = []
        for row in store.db.execute("SELECT * FROM sites"):
            config = SITES[row["site"]]
            n = store.db.execute("SELECT COUNT(*) FROM notices WHERE site=?", (row["site"],)).fetchone()[0]
            sites.append({"站点代码": row["site"], "省份": config["province"], "网站": config["name"], "公告数": n,
                          "状态": row["status"], "说明": f"下一页：{row['next_page']}；{row['error']}", "入口URL": config["start_urls"][0], "采集时间": row["updated"]})
        data = {"columns": COLUMNS, "date_columns": sorted(DATE_COLUMNS), "number_columns": sorted(NUMBER_COLUMNS),
                "notices": notices, "sites": sites, "documents": docs, "bulk": True, "generated_at": now()}
        number = offset // batch_size + 1
        source = folder / f"snapshot_{number:04d}.json"
        atomic_json(source, data)
        jobs.append((source, folder / f"标书与字段_{number:04d}.xlsx"))
    store.db.rollback()
    for source, target in jobs:
        export_excel(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["crawl", "start", "status", "stop", "retry", "import", "export"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--site", default="all")
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--min-free-gb", type=float, default=10)
    parser.add_argument("--rounds", type=int, default=0, help="0持续处理所有公开列表；正数仅跑指定轮数用于验证")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--batch-size", type=int, default=300, help="Excel按此公告数分卷，原始库不受限制")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if not 1 <= args.page_size <= 100 or not 1 <= args.workers <= 4 or args.interval < 0.2 or args.batch_size < 1:
        parser.error("分页、并发、间隔或导出分卷参数无效")
    store = Store(args.root)
    if args.command == "start":
        # A persistent local job is deliberately detached from the interactive tool session.
        lock = (store.root / "worker.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("已有采集进程运行，请使用 status 查看")
        lock.close()
        command = [sys.executable, str(Path(__file__).resolve()), "crawl", "--root", str(args.root), "--site", args.site,
                   "--page-size", str(args.page_size), "--workers", str(args.workers), "--interval", str(args.interval),
                   "--min-free-gb", str(args.min_free_gb), "--rounds", str(args.rounds)]
        with (args.root / "worker.log").open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({"pid": process.pid, "log": str(args.root / "worker.log")}, ensure_ascii=False))
    elif args.command == "crawl":
        store.db.close()
        crawl(args)
    elif args.command == "status":
        report = store.report()
        report["worker"] = json.loads((store.root / "worker.json").read_text()) if (store.root / "worker.json").exists() else None
        if report["worker"] and "stopped_at" not in report["worker"]:
            try:
                os.kill(report["worker"]["pid"], 0)
                report["worker"]["alive"] = True
            except ProcessLookupError:
                report["worker"]["alive"] = False
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "stop":
        (store.root / "STOP").touch()
        print("已请求停止，将在当前请求结束后保存断点。")
    elif args.command == "retry":
        keys = list(SITES) if args.site == "all" else args.site.split(",")
        with store.db:
            store.db.executemany("UPDATE sites SET status='待采集',error='' WHERE site=? AND status NOT IN ('采集中','公开列表已遍历')", [(k,) for k in keys])
            store.db.execute("UPDATE downloads SET attempts=0 WHERE status='失败'")
        print("失败站点已排入下次运行；已保存文件不会重复下载。")
    elif args.command == "import":
        if not args.dataset:
            parser.error("import需要--dataset")
        data = json.loads(args.dataset.read_text(encoding="utf-8"))
        manager = TenderFiles(store.root, min_free_gb=args.min_free_gb, interval=args.interval)
        for n in data["notices"]:
            key = next(k for k, c in SITES.items() if c["name"] == n["record"]["来源网站"])
            store.register([key])
            store.put_notice(key, n)
            store.attach(n, manager)
            store.put_notice(key, n)
        print(json.dumps({k: v for k, v in store.snapshot(csv_files=True).items() if k != "sites"}, ensure_ascii=False))
    elif args.command == "export":
        export(store, args.batch_size)


if __name__ == "__main__":
    main()
