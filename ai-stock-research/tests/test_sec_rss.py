import unittest
from pathlib import Path
import tempfile

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.sources.sec_rss import parse_current_filings, store_rss_events


ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <category term="8-K" />
    <title>8-K - Example Corp (0000123456) (Filer)</title>
    <updated>2026-05-31T12:00:00-04:00</updated>
    <link href="https://www.sec.gov/Archives/edgar/data/123456/000012345626000001/example-index.html" />
    <summary>Example filing summary CIK 0000123456</summary>
  </entry>
</feed>
"""


class SecRssTests(unittest.TestCase):
    def test_parse_current_filings(self) -> None:
        events = parse_current_filings(ATOM_SAMPLE)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].form, "8-K")
        self.assertEqual(events[0].cik, "0000123456")
        self.assertEqual(events[0].company_name, "Example Corp")
        self.assertEqual(events[0].accession_number, "0000123456-26-000001")

    def test_store_rss_events_maps_ticker_and_queues(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            events = parse_current_filings(ATOM_SAMPLE)
            with open_db(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO company_profile (ticker, cik, company_name, source, updated_at)
                    VALUES ('EXM', '0000123456', 'Example Corp', 'unit', 'now')
                    """
                )
                result = store_rss_events(conn, events, forms={"8-K"})
                event_row = conn.execute(
                    "SELECT ticker, accession_number FROM rss_filing_events"
                ).fetchone()
                queue_row = conn.execute(
                    "SELECT ticker, cik, form, accession_number, status FROM filing_queue"
                ).fetchone()
        self.assertEqual(result.events_seen, 1)
        self.assertEqual(result.queued, 1)
        self.assertEqual(event_row["ticker"], "EXM")
        self.assertEqual(event_row["accession_number"], "0000123456-26-000001")
        self.assertEqual(queue_row["ticker"], "EXM")
        self.assertEqual(queue_row["status"], "queued")


if __name__ == "__main__":
    unittest.main()
