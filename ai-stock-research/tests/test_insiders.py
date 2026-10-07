from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.cards import generate_card
from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.sources import insiders


class InsiderTransactionTests(unittest.TestCase):
    def test_parse_sec_ownership_xml_extracts_non_derivative_transaction(self) -> None:
        xml = """
        <ownershipDocument>
          <issuer>
            <issuerTradingSymbol>TEST</issuerTradingSymbol>
          </issuer>
          <reportingOwner>
            <reportingOwnerId>
              <rptOwnerName>Jane Doe</rptOwnerName>
            </reportingOwnerId>
            <reportingOwnerRelationship>
              <isDirector>1</isDirector>
              <isOfficer>1</isOfficer>
              <officerTitle>Chief Financial Officer</officerTitle>
            </reportingOwnerRelationship>
          </reportingOwner>
          <nonDerivativeTable>
            <nonDerivativeTransaction>
              <transactionDate><value>2026-05-31</value></transactionDate>
              <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
              <transactionAmounts>
                <transactionShares><value>1000</value></transactionShares>
                <transactionPricePerShare><value>12.50</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
              </transactionAmounts>
              <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>9000</value></sharesOwnedFollowingTransaction>
              </postTransactionAmounts>
            </nonDerivativeTransaction>
          </nonDerivativeTable>
        </ownershipDocument>
        """

        transactions = insiders.parse_ownership_xml(
            ticker=None,
            xml_text=xml,
            source_url="https://www.sec.gov/Archives/test-form4.xml",
            source_type="SEC 4",
        )

        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0].ticker, "TEST")
        self.assertEqual(transactions[0].owner_name, "Jane Doe")
        self.assertIn("Director", transactions[0].relationship or "")
        self.assertEqual(transactions[0].transaction_code, "S")
        self.assertEqual(transactions[0].acquired_disposed_code, "D")
        self.assertEqual(transactions[0].transaction_shares, 1000)

    def test_import_insider_transactions_csv_writes_evidence_without_risk_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            csv_path = base / "insiders.csv"
            csv_path.write_text(
                "ticker,owner_name,relationship,transaction_date,transaction_code,"
                "acquired_disposed_code,transaction_shares,transaction_price,"
                "shares_owned_following,source_type,source_name,source_url\n"
                "TEST,Jane Doe,Director,2026-05-31,S,D,1000,12.50,9000,"
                "SEC Form 4,SEC EDGAR,https://www.sec.gov/Archives/test-form4.xml\n",
                encoding="utf-8",
            )
            db_path = base / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                count = insiders.import_insider_transactions_csv(conn, csv_path)
                tx = insiders.list_insider_transactions(conn, ticker="TEST")[0]
                evidence = conn.execute(
                    """
                    SELECT related_module, related_ticker, summary
                    FROM evidence_items
                    WHERE related_module = 'insider_transactions'
                    """
                ).fetchone()
                risk_count = conn.execute("SELECT COUNT(*) AS count FROM risk_flags").fetchone()["count"]

        self.assertEqual(count, 1)
        self.assertEqual(tx["owner_name"], "Jane Doe")
        self.assertEqual(tx["transaction_shares"], 1000)
        self.assertEqual(evidence["related_ticker"], "TEST")
        self.assertIn("not automatically", evidence["summary"])
        self.assertEqual(risk_count, 0)

    def test_card_includes_insider_transactions_with_caveat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.sqlite"
            init_db(db_path)
            with open_db(db_path) as conn:
                insiders.upsert_insider_transactions(
                    conn,
                    [
                        insiders.InsiderTransaction(
                            ticker="TEST",
                            owner_name="Jane Doe",
                            relationship="Director",
                            transaction_date="2026-05-31",
                            transaction_code="S",
                            acquired_disposed_code="D",
                            transaction_shares=1000,
                            transaction_price=12.5,
                            shares_owned_following=9000,
                            source_type="SEC Form 4",
                            source_name="SEC EDGAR",
                            source_url="https://www.sec.gov/Archives/test-form4.xml",
                            content_hash="insider-hash",
                        )
                    ],
                )
                content = generate_card(conn, "TEST", output_dir=None)

        self.assertIn("Insider/Form 4 events", content)
        self.assertIn("Jane Doe", content)
        self.assertIn("not automatically a risk conclusion", content)


if __name__ == "__main__":
    unittest.main()
