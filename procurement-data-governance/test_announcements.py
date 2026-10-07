import html
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from announcement_fields import dates, flatten, money, parse_notice, pdf_paragraphs
from download_site import HttpClient


CONFIG = {"province": "云南省", "name": "样本站"}


def parse(text, **kwargs):
    return parse_notice("test", CONFIG, "https://example.gov.cn/notice/1", "<pre>" + html.escape(text) + "</pre>", **kwargs)["record"]


class FieldTests(unittest.TestCase):
    def test_amount_unit_and_unknown(self):
        self.assertEqual(money("127.957333 万元"), 1279573.33)
        self.assertEqual(money("2000000 （元）"), 2000000)
        self.assertEqual(money("￥0 元"), 0)
        self.assertEqual(money("免费获取"), 0)
        self.assertIsNone(money("未公开"))
        self.assertIsNone(money("12345"))
        self.assertIsNone(money("标项1：20万元，标项2：30万元"))

    def test_dates(self):
        self.assertEqual(dates("2026年9月29日9时30分"), ["2026-09-29 09:30:00"])
        self.assertEqual(dates("20260803090800"), ["2026-08-03 09:08:00"])
        self.assertEqual(dates("2026-02-31"), [])

    def test_no_boilerplate_publication_date(self):
        record = parse("电子反拍须知\n平台从2012年12月1日起运行。\n一、电子反拍编号：A-001\n采购人：某大学")
        self.assertIsNone(record["发布时间"])

    def test_deadline_is_not_submission_start(self):
        r = parse("一、项目基本情况\n项目编号：A-1。\n四、响应文件提交\n1.递交时间：2026年9月29日9时00分至9时30分\n2.提交响应文件截止时间及谈判开始时间：2026年9月29日9时30分\n五、开启\n时间：2026年9月29日9时30分")
        self.assertEqual(r["投标截止时间"], "2026-09-29 09:30:00")
        self.assertEqual(r["项目编号"], "A-1")

    def test_opening_and_submission_addresses_are_distinct(self):
        r = parse("四、提交投标文件截止时间、开标时间和地点\n截止时间：2026-10-23 13:00\n投标地点：网上\n开标时间：2026-10-23 13:00\n开标地点：A208室")
        self.assertEqual(r["投标地点"], "网上")
        self.assertEqual(r["开标地点"], "A208室")

    def test_deadline_value_on_next_line(self):
        r = parse("四、响应文件提交\n截止时间：\n2026年10月09日 09时30分00秒\n（北京时间）\n地点：网上")
        self.assertEqual(r["投标截止时间"], "2026-10-09 09:30:00")

    def test_contacts_are_scoped(self):
        r = parse("七、联系方式\n1.采购人信息\n名 称：甲大学\n联系方式：010-12345678\n2.采购代理机构信息\n名称：乙公司\n联系方式：020-87654321\n3.项目联系方式\n项目联系人：张老师\n电 话：138****1234")
        self.assertEqual(r["采购人名称"], "甲大学")
        self.assertEqual(r["代理机构联系方式"], "020-87654321")
        self.assertEqual(r["项目联系电话"], "138****1234")

    def test_flatten_preserves_empty_values(self):
        self.assertEqual(dict(flatten({"zero": 0, "none": None, "items": [], "nested": [{"x": False}]})),
                         {"zero": 0, "none": None, "items": [], "nested[0].x": False})

    def test_pdf_wrapped_fields(self):
        text = pdf_paragraphs("一、项目基本情况\n合同履行期限：30天内完成供货、安装（供应\n商负责运输）。\n二、联系方式\n地址：吴井路183号商业广场B座\n709号\n电话：0871-12345678")
        self.assertIn("供应商负责运输", text)
        self.assertIn("B座709号", text)
        self.assertIn("\n电话", text)

    def test_api_time_conflict_is_exposed(self):
        r = parse("四、响应文件提交\n截止时间：2026年8月3日9时00分", data={"bidclosingtime": "20260803090800"})
        self.assertEqual(r["投标截止时间"], "2026-08-03 09:00:00")
        self.assertIn("09:08", r["字段备注"])

    def test_joint_bid_no(self):
        self.assertEqual(parse("本项目（不）接受联合体。 ")["是否接受联合体"], "否")

    def test_spaced_pdf_date_with_period(self):
        self.assertEqual(dates("2026 年 9 月 25 日下午 4:00。"), ["2026-09-25 16:00:00"])

    def test_attachments_outside_template(self):
        markup = '<div id="template-center-page">' + "项目情况" * 50 + '</div><a href="https://files.gov.cn/file.pdf">公告.pdf</a>'
        n = parse_notice("test", CONFIG, "https://example.gov.cn", markup)
        self.assertEqual(n["attachments"][0]["name"], "公告.pdf")

    def test_auction_times_are_not_file_obtain_times(self):
        r = parse("竞价采购公告", data={"startBidtime": "2026-09-23 09:00:00", "endBidtime": "2026-09-25 17:00:00"})
        self.assertIsNone(r["获取文件开始时间"])
        self.assertIsNone(r["发布时间"])
        self.assertEqual(r["竞价开始时间"], "2026-09-23 09:00:00")

    def test_curl_preserves_post_and_status(self):
        response = SimpleNamespace(returncode=0, stdout=b'blocked\nCODEX_HTTP_META:403\ttext/html\thttps://example.gov.cn/api', stderr=b'')
        with patch("download_site.subprocess.run", return_value=response) as run:
            result = HttpClient().fetch_with_curl("https://example.gov.cn/api", method="POST", body=b'{"page":1}', headers={"Content-Type": "application/json"})
        self.assertEqual(result.status, 403)
        self.assertEqual(result.content, b"blocked")
        self.assertIn("POST", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["input"], b'{"page":1}')


if __name__ == "__main__":
    unittest.main()
