from pathlib import Path
import tempfile
import unittest

from ai_stock_discovery.database import init_db, open_db
from ai_stock_discovery.source_input_import import (
    import_source_input_pack,
    write_source_input_import_csv,
)
from ai_stock_discovery.source_input_templates import (
    build_source_input_audit,
    filter_source_input_templates,
    select_source_input_templates,
    write_source_input_audit_csv,
    write_source_input_template_pack,
)


class SourceInputTemplateTests(unittest.TestCase):
    def test_selects_only_readiness_gap_templates(self) -> None:
        statuses = {
            "source_market_confirmation": "unavailable",
            "table_market_confirmation": "needs_data",
            "source_macro": "ready",
            "source_short_sale": "needs_data",
            "source_expectation_gap": "ready",
            "table_expectation_gap": "ready",
            "table_ai_chain_tags": "ready",
            "table_catalysts": "ready",
        }

        templates = select_source_input_templates(readiness_statuses=statuses, only_gaps=True)
        names = {template.template_name for template in templates}

        self.assertIn("market_confirmation_signals", names)
        self.assertIn("market_price_bars", names)
        self.assertIn("finra_short_sale_volume", names)
        self.assertNotIn("macro_indicators", names)
        self.assertNotIn("expectation_gap_signals", names)

    def test_filters_templates_by_name_csv_filename_and_path(self) -> None:
        templates = select_source_input_templates()

        selected = filter_source_input_templates(
            templates,
            ["ai_industry_tags,macro_indicators.csv", "data/input_templates/market_price_bars.csv"],
        )
        names = [template.template_name for template in selected]

        self.assertEqual(names, ["ai_industry_tags", "market_price_bars", "macro_indicators"])

    def test_filter_templates_rejects_unknown_name(self) -> None:
        templates = select_source_input_templates()

        with self.assertRaisesRegex(ValueError, "does_not_exist"):
            filter_source_input_templates(templates, ["does_not_exist"])

    def test_writes_header_only_templates_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            output_dir = base / "templates"
            manifest = base / "manifest.csv"
            templates = [
                template
                for template in select_source_input_templates()
                if template.template_name in {"ai_industry_tags", "macro_indicators"}
            ]

            rows = write_source_input_template_pack(
                templates,
                output_dir=output_dir,
                manifest_path=manifest,
            )

            ai_template = output_dir / "ai_industry_tags.csv"
            macro_template = output_dir / "macro_indicators.csv"
            ai_text = ai_template.read_text(encoding="utf-8")
            macro_text = macro_template.read_text(encoding="utf-8")
            manifest_text = manifest.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.write_status == "written" for row in rows))
        self.assertEqual(ai_text.splitlines(), ["ticker,tag,confidence,source_type,source_url,evidence_snippet"])
        self.assertIn("series_id,source_name,metric_name,category,period,value,source_url", macro_text)
        self.assertIn("python -m ai_stock_discovery.cli import-ai-tags", manifest_text)
        self.assertIn("Macro/electricity rows are background context only", manifest_text)

    def test_preserves_existing_template_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            output_dir = base / "templates"
            output_dir.mkdir()
            existing = output_dir / "ai_industry_tags.csv"
            existing.write_text("custom,data\n", encoding="utf-8")
            manifest = base / "manifest.csv"
            template = [
                item
                for item in select_source_input_templates()
                if item.template_name == "ai_industry_tags"
            ]

            rows = write_source_input_template_pack(
                template,
                output_dir=output_dir,
                manifest_path=manifest,
            )
            preserved = existing.read_text(encoding="utf-8")
            overwritten_rows = write_source_input_template_pack(
                template,
                output_dir=output_dir,
                manifest_path=manifest,
                overwrite=True,
            )
            overwritten = existing.read_text(encoding="utf-8")

        self.assertEqual(rows[0].write_status, "exists")
        self.assertEqual(preserved, "custom,data\n")
        self.assertEqual(overwritten_rows[0].write_status, "written")
        self.assertIn("ticker,tag", overwritten)

    def test_audit_marks_header_only_templates_ready_to_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            output_dir = base / "templates"
            manifest = base / "manifest.csv"
            template = [
                item
                for item in select_source_input_templates()
                if item.template_name == "market_price_bars"
            ]
            write_source_input_template_pack(template, output_dir=output_dir, manifest_path=manifest)

            rows = build_source_input_audit(template, input_dir=output_dir)

        self.assertEqual(rows[0].file_status, "header_only")
        self.assertEqual(rows[0].audit_status, "ready_to_fill")
        self.assertEqual(rows[0].data_rows, 0)
        self.assertEqual(rows[0].missing_required_columns, "")

    def test_audit_blocks_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            output_dir = base / "templates"
            output_dir.mkdir()
            (output_dir / "market_price_bars.csv").write_text("ticker,date,close\nTEST,2026-05-31,10\n", encoding="utf-8")
            template = [
                item
                for item in select_source_input_templates()
                if item.template_name == "market_price_bars"
            ]

            rows = build_source_input_audit(template, input_dir=output_dir)

        self.assertEqual(rows[0].audit_status, "missing_required_columns")
        self.assertEqual(rows[0].missing_required_columns, "volume")
        self.assertEqual(rows[0].data_rows, 1)

    def test_audit_blocks_missing_required_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            output_dir = base / "templates"
            output_dir.mkdir()
            (output_dir / "expectation_gap_signals.csv").write_text(
                "ticker,source_url,signal_type,direction,description,confidence\n"
                "TEST,https://www.sec.gov/Archives/edgar/data/0000000000/source.htm,legacy_market_label,,Source-backed row,0.8\n",
                encoding="utf-8",
            )
            template = [
                item
                for item in select_source_input_templates()
                if item.template_name == "expectation_gap_signals"
            ]

            rows = build_source_input_audit(template, input_dir=output_dir)

        self.assertEqual(rows[0].audit_status, "missing_required_values")
        self.assertEqual(rows[0].rows_missing_required_values, 1)
        self.assertEqual(rows[0].rows_with_source_reference, 1)
        self.assertIn("required field", rows[0].recommended_action)

    def test_audit_blocks_placeholder_source_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            output_dir = base / "templates"
            output_dir.mkdir()
            (output_dir / "expectation_gap_signals.csv").write_text(
                "ticker,source_url,signal_type,direction,description,confidence\n"
                "TEST,SOURCE_URL_OR_LOCAL_PATH,legacy_market_label,supports_gap,Source-backed row,0.8\n",
                encoding="utf-8",
            )
            template = [
                item
                for item in select_source_input_templates()
                if item.template_name == "expectation_gap_signals"
            ]

            rows = build_source_input_audit(template, input_dir=output_dir)

        self.assertEqual(rows[0].audit_status, "placeholder_source_reference")
        self.assertEqual(rows[0].rows_with_source_reference, 1)
        self.assertEqual(rows[0].rows_with_placeholder_source_reference, 1)

    def test_audit_flags_rows_without_source_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            output_dir = base / "templates"
            output_dir.mkdir()
            (output_dir / "ai_industry_tags.csv").write_text(
                "ticker,tag,confidence,source_type,source_url,evidence_snippet\n"
                "TEST,cooling,0.8,manual,,source-backed note\n",
                encoding="utf-8",
            )
            template = [
                item
                for item in select_source_input_templates()
                if item.template_name == "ai_industry_tags"
            ]

            rows = build_source_input_audit(template, input_dir=output_dir)

        self.assertEqual(rows[0].audit_status, "source_reference_gap")
        self.assertEqual(rows[0].source_reference_columns, "source_url")
        self.assertEqual(rows[0].rows_with_source_reference, 0)

    def test_audit_marks_source_backed_rows_ready_to_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            output_dir = base / "templates"
            output_dir.mkdir()
            audit_output = base / "audit.csv"
            (output_dir / "expectation_gap_signals.csv").write_text(
                "ticker,source_url,signal_type,direction,description,confidence,signal_date,source_type,source_name,magnitude\n"
                "TEST,https://www.sec.gov/Archives/edgar/data/0000000000/source.htm,legacy_market_label,supports_gap,Source-backed row,0.8,2026-05-31,manual,unit,\n",
                encoding="utf-8",
            )
            template = [
                item
                for item in select_source_input_templates()
                if item.template_name == "expectation_gap_signals"
            ]

            rows = build_source_input_audit(template, input_dir=output_dir)
            write_source_input_audit_csv(rows, audit_output)
            audit_text = audit_output.read_text(encoding="utf-8")

        self.assertEqual(rows[0].audit_status, "ready_to_import")
        self.assertEqual(rows[0].rows_with_source_reference, 1)
        self.assertIn("import-expectation-gap-csv", rows[0].recommended_action)
        self.assertIn("ready_to_import", audit_text)

    def test_source_input_import_pack_dry_run_does_not_modify_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            input_dir = base / "templates"
            output = base / "source_input_import.csv"
            input_dir.mkdir()
            (input_dir / "ai_industry_tags.csv").write_text(
                "ticker,tag,confidence,source_type,source_url,evidence_snippet\n"
                "TEST,cooling,0.8,manual,https://www.sec.gov/Archives/edgar/data/0000000000/source.htm,Source-backed cooling row\n",
                encoding="utf-8",
            )
            templates = [
                item
                for item in select_source_input_templates()
                if item.template_name == "ai_industry_tags"
            ]
            init_db(db_path)

            with open_db(db_path) as conn:
                rows = import_source_input_pack(conn, templates, input_dir=input_dir)
                write_source_input_import_csv(rows, output)
                tag_count = conn.execute("SELECT COUNT(*) FROM ai_industry_tags").fetchone()[0]
            import_text = output.read_text(encoding="utf-8")

        self.assertEqual(rows[0].import_status, "dry_run_ready")
        self.assertEqual(rows[0].rows_imported, 0)
        self.assertEqual(tag_count, 0)
        self.assertIn("dry_run_ready", import_text)

    def test_source_input_import_pack_apply_imports_ready_templates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            db_path = base / "test.sqlite"
            input_dir = base / "templates"
            input_dir.mkdir()
            (input_dir / "ai_industry_tags.csv").write_text(
                "ticker,tag,confidence,source_type,source_url,evidence_snippet\n"
                "TEST,cooling,0.8,manual,https://www.sec.gov/Archives/edgar/data/0000000000/source.htm,Source-backed cooling row\n",
                encoding="utf-8",
            )
            templates = [
                item
                for item in select_source_input_templates()
                if item.template_name == "ai_industry_tags"
            ]
            init_db(db_path)

            with open_db(db_path) as conn:
                rows = import_source_input_pack(conn, templates, input_dir=input_dir, apply=True)
                tag_count = conn.execute("SELECT COUNT(*) FROM ai_industry_tags WHERE ticker = 'TEST'").fetchone()[0]
                source_status = conn.execute(
                    """
                    SELECT status
                    FROM data_source_status
                    WHERE source_name = 'Manual AI industry tags CSV'
                    """
                ).fetchone()["status"]

        self.assertEqual(rows[0].import_status, "imported")
        self.assertEqual(rows[0].rows_imported, 1)
        self.assertEqual(tag_count, 1)
        self.assertEqual(source_status, "ok")


if __name__ == "__main__":
    unittest.main()
