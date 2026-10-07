from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.source_input_plan import (
    build_source_input_plan,
    write_source_input_plan_csv,
)
from ai_stock_discovery.source_input_templates import (
    select_source_input_templates,
    write_source_input_template_pack,
)


class SourceInputPlanTests(unittest.TestCase):
    def test_plan_guides_header_only_template_fill_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            input_dir = base / "input_templates"
            manifest = base / "source_input_templates.csv"
            output = base / "source_input_plan.csv"
            template = [
                item
                for item in select_source_input_templates()
                if item.template_name == "ai_industry_tags"
            ]
            write_source_input_template_pack(template, output_dir=input_dir, manifest_path=manifest)

            rows = build_source_input_plan(
                template,
                input_dir=input_dir,
                readiness_statuses={"table_ai_chain_tags": "needs_data"},
                audit_output=base / "source_input_audit.csv",
                import_output=base / "source_input_import.csv",
            )
            write_source_input_plan_csv(rows, output)
            text = output.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.template_name, "ai_industry_tags")
        self.assertEqual(row.readiness_statuses, "table_ai_chain_tags=needs_data")
        self.assertEqual(row.audit_status, "ready_to_fill")
        self.assertEqual(row.import_status, "not_ready")
        self.assertIn("--template ai_industry_tags", row.audit_command)
        self.assertIn("--apply", row.apply_command)
        self.assertIn("Fill", row.recommended_next_action)
        self.assertIn("template_name", text)
        self.assertIn("ready_to_fill", text)

    def test_plan_marks_source_backed_template_ready_for_dry_run_apply_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            input_dir = base / "input_templates"
            input_dir.mkdir()
            (input_dir / "expectation_gap_signals.csv").write_text(
                "ticker,source_url,signal_type,direction,description,confidence\n"
                "TEST,https://www.sec.gov/Archives/edgar/data/0000000000/source.htm,"
                "legacy_market_label,supports_gap,Source-backed row,0.8\n",
                encoding="utf-8",
            )
            template = [
                item
                for item in select_source_input_templates()
                if item.template_name == "expectation_gap_signals"
            ]

            rows = build_source_input_plan(
                template,
                input_dir=input_dir,
                readiness_statuses={
                    "source_expectation_gap": "needs_data",
                    "table_expectation_gap": "needs_data",
                },
            )

        self.assertEqual(rows[0].audit_status, "ready_to_import")
        self.assertEqual(rows[0].import_status, "dry_run_ready")
        self.assertEqual(rows[0].data_rows, 1)
        self.assertIn("Review the dry-run report", rows[0].recommended_next_action)
        self.assertIn("source_expectation_gap=needs_data", rows[0].readiness_statuses)
        self.assertIn("table_expectation_gap=needs_data", rows[0].readiness_statuses)


if __name__ == "__main__":
    unittest.main()
