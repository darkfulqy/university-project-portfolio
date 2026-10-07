import json
import tempfile
import unittest
import zipfile
import threading
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from bulk_tenders import Store, crawl, now, transient_network_error, source_weight
from paged_collectors import PagedCollector
from collect_announcements import Collector
from tender_files import TenderFiles, classify, file_format


class BulkTests(unittest.TestCase):
    def test_easyjcx_list_pagination_and_null_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = PagedCollector("yunnan_easyjcx", Path(directory), 2, 10)
            payload = {"data": {"records": [{"orderMainId": "abc"}], "total": 486, "current": 2}}
            with patch("collect_announcements.Collector.api", return_value=(payload, "list.json")) as request:
                collector.api("https://api.easyjcx.com/api/portalCli/bidNeed", body={"current": 1, "size": 10})
            self.assertEqual(request.call_args.kwargs["body"]["current"], 2)
            self.assertEqual(collector.list_ids, ["abc"])
            self.assertEqual(collector.source_total, 486)
            collector = Collector("yunnan_easyjcx", Path(directory), 2, 0)
            detail = {"data": {"orderTitle": None, "detailList": [{"deviceName": None, "deviceSpec": None}]}}
            with patch.object(collector, "api", side_effect=[(payload, "list.json"), (detail, "detail.json")]) as api:
                collector.easyjcx()
            self.assertEqual(api.call_args_list[0].args[0], "https://api.easyjcx.com/api/portalCli/bidNeed")
            self.assertEqual(len(collector.notices), 1)

    def test_guangdong_uses_paginated_list_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = Collector("guangdong_ccgp", Path(directory), 2, 0)
            responses = [({"data": {"id": "site-id"}}, "site.json"),
                         ({"data": [{"dictName": "采购公告", "dictCode": "00101"}]}, "dict.json"),
                         ({"data": {"rows": []}}, "list.json")]
            with patch.object(collector, "api", side_effect=responses) as api:
                collector.guangdong()
            self.assertIn("/info/selectInfoForIndex?", api.call_args_list[-1].args[0])
            collector = PagedCollector("guangdong_ccgp", Path(directory), 3, 10)
            with patch("collect_announcements.Collector.api", return_value=({"data": {"rows": [{"id": "third-page"}], "total": 182335}}, "page.json")) as api:
                collector.api("https://gdgpo.czt.gd.gov.cn/gpcms/rest/web/v2/info/selectInfoForIndex?currPage=1&pageSize=2")
            self.assertIn("currPage=3", api.call_args.args[0])
            self.assertIn("pageSize=10", api.call_args.args[0])
            self.assertEqual(collector.list_ids, ["third-page"])
            self.assertEqual(collector.source_total, 182335)

    def test_document_priority_retains_low_yield_sites(self):
        self.assertGreater(source_weight("jilin_ggzy", 1000, 700), source_weight("guangdong_gdegp", 1000, 0))
        self.assertGreater(source_weight("guangdong_gdegp", 1000, 0), 0)

    def test_fast_source_does_not_wait_for_slow_source(self):
        with tempfile.TemporaryDirectory() as directory:
            slow_started, fast_second, slow_saw_second = threading.Event(), threading.Event(), threading.Event()
            def fake_page(root, key, page, *unused):
                if key == "beijing_ccgp" and page == 1:
                    slow_started.set()
                    if fast_second.wait(3):
                        slow_saw_second.set()
                elif key == "jilin_ggzy" and page == 1:
                    slow_started.wait(2)
                elif key == "jilin_ggzy" and page == 2:
                    fast_second.set()
                s = Store(root)
                with s.db:
                    s.db.execute("UPDATE sites SET next_page=?,status='采集中' WHERE site=?", (page + 1, key))
                s.db.close()
                return {"site": key, "page": page}
            args = SimpleNamespace(root=Path(directory), site="jilin_ggzy,beijing_ccgp", page_size=10,
                                   workers=2, interval=1, min_free_gb=0, rounds=2)
            with patch("bulk_tenders.crawl_page", side_effect=fake_page), patch("bulk_tenders.signal.signal"), patch("builtins.print"):
                crawl(args)
            self.assertTrue(slow_saw_second.is_set())
            self.assertTrue((Path(directory) / "progress.json").is_file())

    def test_network_retry_does_not_retry_access_blocks(self):
        self.assertTrue(transient_network_error("HTTP 0: Could not resolve host"))
        self.assertTrue(transient_network_error("HTTP 503: unavailable"))
        self.assertFalse(transient_network_error("HTTP 403: Forbidden"))
        self.assertFalse(transient_network_error("分页响应重复"))

    def test_network_recovery_rewinds_partial_page(self):
        with tempfile.TemporaryDirectory() as directory:
            s = Store(directory)
            s.register(["jilin_ggzy", "yunnan_ggzy", "beijing_ccgp"])
            with s.db:
                s.db.execute("UPDATE sites SET next_page=4,status='访问/接口失败',error='HTTP 0: DNS',updated='2000-01-01 00:00:00' WHERE site='jilin_ggzy'")
                s.db.execute("UPDATE sites SET status='访问/接口失败',error='HTTP 403' WHERE site='yunnan_ggzy'")
                s.db.execute("UPDATE sites SET status='访问/接口失败',error='HTTP 0: DNS',updated=? WHERE site='beijing_ccgp'", (now(),))
                s.db.execute("INSERT INTO pages VALUES('jilin_ggzy',2,'good',10,10,'采集中','',?)", (now(),))
                s.db.execute("INSERT INTO pages VALUES('jilin_ggzy',3,'partial',8,10,'部分失败','HTTP 0: DNS',?)", (now(),))
            self.assertTrue(s.retry_network(["jilin_ggzy", "yunnan_ggzy", "beijing_ccgp"]))
            row = s.db.execute("SELECT * FROM sites WHERE site='jilin_ggzy'").fetchone()
            self.assertEqual((row["next_page"], row["signature"], row["status"]), (3, "good", "待采集"))
            self.assertEqual(s.db.execute("SELECT status FROM sites WHERE site='yunnan_ggzy'").fetchone()[0], "访问/接口失败")
            self.assertEqual(s.db.execute("SELECT status FROM sites WHERE site='beijing_ccgp'").fetchone()[0], "访问/接口失败")
            s.db.close()

    def test_fake_pdf_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "招标文件.pdf"
            file.write_bytes(b"<!doctype html><html>login</html>")
            with self.assertRaises(ValueError):
                file_format(file)

    def test_notice_and_tender_are_distinct(self):
        self.assertEqual(classify("招标公告.pdf", "采购项目招标公告", 4)[0], "公告")
        self.assertEqual(classify("磋商文件.pdf", "竞争性磋商文件 目录 磋商须知 合同协议书 响应文件格式", 65)[0], "完整标书（结构识别）")
        self.assertEqual(classify("委托代理协议.pdf", "招标文件 投标人须知 合同条款", 5)[0], "其他附件")
        self.assertEqual(classify("采购需求.docx", "招标文件 投标人须知 合同条款")[0], "采购需求/询价表")
        self.assertEqual(classify("公告.pdf", "招标文件 投标人须知 合同条款", 7)[0], "公告")

    def test_zip_relationship_and_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "files.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("资料/标书.txt", "招标文件 投标人须知 合同条款")
            manager = TenderFiles(root / "store", min_free_gb=0)
            notice = {"record": {"公告ID": "n1", "项目编号": "P1"}}
            first = manager.ingest(archive, notice=notice, name="包.zip", url="https://example.gov.cn/a.zip")
            second = manager.ingest(archive, notice=notice, name="包.zip", url="https://example.gov.cn/a.zip")
            self.assertEqual(len(first), 2)
            self.assertEqual(first[1]["parent_id"], first[0]["document_id"])
            self.assertEqual(first[1]["archive_member"], "资料/标书.txt")
            self.assertEqual([d["document_id"] for d in first], [d["document_id"] for d in second])
            self.assertEqual(len(list((root / "store/objects").glob("*/*"))), 2)

    def test_zip_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.zip"
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("../../escape.pdf", b"%PDF-1.7")
            docs = TenderFiles(root / "store", min_free_gb=0).ingest(source, notice={"record": {"公告ID": "n1"}}, name="bad.zip", url="https://example.gov.cn/bad.zip")
            self.assertEqual(len(docs), 1)
            self.assertIn("不安全", docs[0]["inspection_error"])
            self.assertFalse((root / "escape.pdf").exists())

    def test_pagination_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            c = PagedCollector("jilin_ggzy", Path(directory), 3, 20)
            with patch("collect_announcements.Collector.api", return_value=({"datas": [{"docpuburl": "u1"}], "recordnum": 100}, "test")) as request:
                c.api("http://was.jl.gov.cn/was5/web/search?channelid=237687&page=1&prepage=20")
            self.assertIn("page=3", request.call_args.args[0])
            self.assertEqual(c.list_ids, ["u1"])
            c = PagedCollector("yunnan_kust", Path(directory), 3, 20)
            with patch("collect_announcements.Collector.api", return_value=({"resultset": [], "count": 0}, "test")) as request:
                c.api("https://cgzx.kust.edu.cn/sfw_cms/e", form={"page": "cms.psms.publish.query", "start": 1, "limit": 60})
            self.assertEqual(request.call_args.kwargs["form"]["start"], 41)
            self.assertEqual(request.call_args.kwargs["form"]["limit"], 20)

    def test_sqlite_idempotent_and_relationship(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            n = {"record": {"公告ID": "n1", "公告标题": "test", "来源URL": "https://example.gov.cn"}}
            store.put_notice("beijing_ccgp", n)
            store.put_notice("beijing_ccgp", n)
            doc = {"document_id": "d1", "notice_id": "n1", "sha256": "hash", "format": "pdf", "category": "公告"}
            store.put_documents([doc, doc])
            self.assertEqual(store.report()["announcements"], 1)
            self.assertEqual(store.report()["documents"], 1)
            self.assertEqual(store.report()["full_tender_pdf"], 0)
            store.db.close()


if __name__ == "__main__":
    unittest.main()
