import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class IcaoAppHelpersTest(unittest.TestCase):
    def test_forecast_audit_never_surfaces_dashboard_generated_provenance(self):
        from app import _forecast_audit_source

        generated = pd.DataFrame([{
            "+30 min source": "Dashboard-generated persistence forecast",
        }])
        official = pd.DataFrame([{
            "+30 min source": "SERENE official forecast",
        }])

        self.assertEqual(
            _forecast_audit_source(generated, "+30 min source"), "Unavailable"
        )
        self.assertEqual(
            _forecast_audit_source(official, "+30 min source"),
            "SERENE official forecast",
        )

    def test_forecast_helpers_restore_all_four_horizons(self):
        from app import (
            _available_primary_periods,
            _forecast_availability_message,
            _visible_summary_columns,
        )
        from data_loader import LoadStatus

        status = LoadStatus(metadata={
            "available_primary_forecast_periods": [30, 90, 180, 360],
            "forecast_request_audit": [
                {"forecast_parameter": 30, "display_role": "primary", "outcome": "available"},
                {"forecast_parameter": 90, "display_role": "primary", "outcome": "available"},
                {"forecast_parameter": 180, "display_role": "primary", "outcome": "available"},
                {"forecast_parameter": 360, "display_role": "primary", "outcome": "available"},
            ],
        })
        summary = pd.DataFrame(columns=[
            "Indicator", "Latest value", "Status",
            "+30 min forecast", "+30 min status", "+30 min source",
            "+90 min forecast", "+90 min status", "+90 min source",
            "+3h forecast", "+3h status", "+3h source",
            "+6h forecast", "+6h status", "+6h source",
        ])

        self.assertEqual(_available_primary_periods(status), [30, 90, 180, 360])
        visible = _visible_summary_columns(summary, status)
        for label in ("+30 min", "+90 min", "+3h", "+6h"):
            self.assertIn(f"{label} forecast", visible)
        message = _forecast_availability_message(status)
        self.assertIn("+30 min, +90 min, +3 h and +6 h retrieved", message)

    def test_forecast_helpers_keep_unavailable_horizon_groups_visible(self):
        from app import _available_primary_periods, _visible_summary_columns
        from data_loader import LoadStatus

        status = LoadStatus(metadata={
            "available_primary_forecast_periods": [30],
        })
        summary = pd.DataFrame(columns=[
            "Indicator", "Latest value", "Status",
            "+30 min forecast", "+30 min status", "+30 min source",
            "+90 min forecast", "+90 min status", "+90 min source",
            "+3h forecast", "+3h status", "+3h source",
            "+6h forecast", "+6h status", "+6h source",
        ])

        self.assertEqual(_available_primary_periods(status), [30])
        visible = _visible_summary_columns(summary, status)
        for label in ("+30 min", "+90 min", "+3h", "+6h"):
            self.assertIn(f"{label} forecast", visible)

    def test_kp_horizon_keeps_summary_group_visible_without_aida_forecast(self):
        from app import _visible_summary_columns
        from data_loader import LoadStatus

        status = LoadStatus(metadata={"available_primary_forecast_periods": []})
        summary = pd.DataFrame([{
            "Indicator": "Auroral Absorption",
            "+30 min forecast": 7.5,
            "+30 min status": "OK",
            "+30 min source": "GFZ official PAGER/SWIFT ensemble forecast",
            "+90 min forecast": "N/A",
            "+90 min status": "UNAVAILABLE",
            "+90 min source": "Unavailable",
            "+3h forecast": "N/A",
            "+3h status": "UNAVAILABLE",
            "+3h source": "Unavailable",
            "+6h forecast": "N/A",
            "+6h status": "UNAVAILABLE",
            "+6h source": "Unavailable",
        }])

        visible = _visible_summary_columns(summary, status)

        for label in ("+30 min", "+90 min", "+3h", "+6h"):
            self.assertIn(f"{label} forecast", visible)

    def test_kp_horizon_evidence_table_exposes_role_and_uncertainty(self):
        from app import _kp_horizon_evidence_table

        horizons = pd.DataFrame([
            {
                "horizon_minutes": 30,
                "target_time": "2026-08-12T13:30:00Z",
                "value": 7.5,
                "evidence_role": "official_forecast",
                "source": "GFZ official PAGER/SWIFT ensemble forecast",
                "ensemble_maximum": 8.4,
                "probability_kp_ge_8": 0.2,
                "issue_time": "2026-08-12T13:05:20Z",
                "data_status": "forecast",
            },
            {
                "horizon_minutes": 90,
                "target_time": "2026-07-01T07:25:00Z",
                "value": 3.0,
                "evidence_role": "observed_backtesting",
                "source": "GFZ observed outcome — backtesting only",
                "ensemble_maximum": float("nan"),
                "probability_kp_ge_8": float("nan"),
                "issue_time": pd.NaT,
                "data_status": "preliminary",
            },
            {
                "horizon_minutes": 180,
                "target_time": "2026-08-12T16:00:00Z",
                "value": 6.0,
                "evidence_role": "official_forecast",
                "source": "GFZ official PAGER/SWIFT ensemble forecast",
                "ensemble_maximum": 6.8,
                "probability_kp_ge_8": 0.1,
                "issue_time": "2026-08-12T13:05:20Z",
                "data_status": "forecast",
            },
            {
                "horizon_minutes": 360,
                "target_time": "2026-08-12T19:00:00Z",
                "value": 5.0,
                "evidence_role": "official_forecast",
                "source": "GFZ official PAGER/SWIFT ensemble forecast",
                "ensemble_maximum": 5.9,
                "probability_kp_ge_8": 0.0,
                "issue_time": "2026-08-12T13:05:20Z",
                "data_status": "forecast",
            },
        ])

        table = _kp_horizon_evidence_table(horizons)

        self.assertEqual(table["Evidence role"].tolist(), [
            "Official forecast", "Observed outcome (backtesting only)",
            "Official forecast", "Official forecast",
        ])
        self.assertEqual(table["Horizon"].tolist(), [
            "+30 min", "+90 min", "+3 h", "+6 h",
        ])
        self.assertEqual(table["Primary status"].tolist(), ["OK", "OK", "OK", "OK"])
        self.assertEqual(table.iloc[0]["Ensemble maximum"], 8.4)
        self.assertEqual(table.iloc[0]["P(Kp >= 8)"], "20%")
        self.assertEqual(table.iloc[1]["P(Kp >= 8)"], "N/A")

    def test_forecast_message_reports_all_available_horizons_without_audit_only_wording(self):
        from app import _forecast_availability_message
        from data_loader import LoadStatus

        status = LoadStatus(metadata={
            "available_primary_forecast_periods": [30, 90, 180, 360],
            "forecast_request_audit": [
                {"forecast_parameter": 30, "outcome": "available"},
                {"forecast_parameter": 90, "outcome": "available"},
                {"forecast_parameter": 180, "outcome": "available"},
                {"forecast_parameter": 360, "outcome": "available"},
            ],
        })

        message = _forecast_availability_message(status)

        self.assertIn("+30 min, +90 min, +3 h and +6 h retrieved", message)
        self.assertNotIn("audit only", message)

    def test_requested_window_rejects_reversed_range(self):
        from app_utils import validate_requested_window

        error = validate_requested_window(
            "2026-06-24T13:00:00Z",
            "2026-06-24T12:00:00Z",
            publication_safe_now=pd.Timestamp("2026-06-24T14:00:00Z"),
        )

        self.assertIn("before", error)

    def test_requested_window_rejects_unpublished_future(self):
        from app_utils import validate_requested_window

        error = validate_requested_window(
            "2026-06-24T10:00:00Z",
            "2026-06-24T13:50:00Z",
            publication_safe_now=pd.Timestamp("2026-06-24T13:45:00Z"),
        )

        self.assertIn("future", error)

    def test_advisory_metadata_is_stable_and_clears_on_failed_load(self):
        from app_utils import advisory_metadata_for_load

        generated = pd.Timestamp("2026-06-24T12:00:00Z")
        success = advisory_metadata_for_load(True, 4, generated)
        failure = advisory_metadata_for_load(False, success["sequence"], generated)

        self.assertEqual(success["sequence"], 5)
        self.assertEqual(success["number"], "2026/005")
        self.assertEqual(success["generated_time"], generated)
        self.assertEqual(failure, {
            "sequence": 5,
            "generated_time": None,
            "number": None,
        })

    def test_successful_live_load_is_not_described_as_api_not_tested(self):
        from app_utils import loaded_api_state
        from data_loader import LoadStatus

        status = LoadStatus(source="api", ok=True, message="Live AIDA loaded")

        level, text = loaded_api_state(status, None, "Not tested yet.")

        self.assertEqual(level, "success")
        self.assertIn("live load succeeded", text.lower())

    def test_provenance_metadata_exposes_full_utc_values(self):
        from app_utils import build_provenance_metadata

        rows = build_provenance_metadata(
            "2026-08-11T17:35:00Z",
            pd.Timestamp("2026-08-11T17:35:00Z"),
            pd.Timestamp("2026-08-11T17:36:00Z"),
            pd.Timestamp("2026-08-11T18:00:00Z"),
            3,
        )

        self.assertEqual(rows[0], {
            "label": "Requested analysis",
            "value": "2026-08-11 17:35 UTC",
        })
        self.assertEqual(rows[1]["value"], "2026-08-11 17:35 UTC")
        self.assertEqual(rows[2]["value"], "2026-08-11 17:36 UTC")
        self.assertEqual(rows[3]["value"], "25 min")
        self.assertEqual(rows[-1]["value"], "3 official")

    def test_display_data_keeps_rolling_products_for_time_series(self):
        from app import _build_display_data
        from data_loader import IcaoProductBundle, LoadStatus

        products = pd.DataFrame([
            {
                "time": "2025-01-01T17:50:00Z",
                "variable": "TEC",
                "value": 11.0,
                "product_kind": "rolling",
            },
            {
                "time": "2025-01-01T17:55:00Z",
                "variable": "TEC",
                "value": 12.0,
                "product_kind": "analysis",
            },
        ])
        indices = pd.DataFrame([{
            "time": "2025-01-01T17:55:00Z",
            "variable": "Kp",
            "value": 8.0,
        }])
        bundle = IcaoProductBundle(
            products=products,
            indices=indices,
            status=LoadStatus(source="api", ok=True, message="loaded"),
        )

        display = _build_display_data(bundle)

        self.assertEqual(len(display), 3)
        self.assertIn("rolling", set(display["product_kind"].dropna()))

    def test_streamlit_app_starts_without_exception(self):
        from streamlit.testing.v1 import AppTest

        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
        app = AppTest.from_file(app_path, default_timeout=30).run()

        self.assertFalse(app.exception, [item.value for item in app.exception])

    def test_loaded_trial_renders_evidence_first_sections(self):
        from streamlit.testing.v1 import AppTest
        from data_loader import IcaoProductBundle, LoadStatus
        from icao_risk import build_icao_summary

        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
        products = pd.DataFrame([
            {
                "time": "2026-08-11T17:35:00Z",
                "actual_output_time": "2026-08-11T17:35:00Z",
                "variable": "TEC",
                "product_kind": "analysis",
                "lat": 50.0,
                "lon": 0.0,
                "value": 100.0,
                "source": "SERENE AIDA analysis",
            },
            {
                "time": "2026-08-11T17:35:00Z",
                "actual_output_time": "2026-08-11T17:35:00Z",
                "variable": "MUF3000F2",
                "product_kind": "analysis",
                "lat": 50.0,
                "lon": 0.0,
                "value": 8.0,
                "psd_percent": pd.NA,
                "source": "SERENE AIDA analysis",
            },
        ])
        status = LoadStatus(
            source="api",
            ok=True,
            message="Loaded live AIDA data",
            metadata={
                "analysis_time": "2026-08-11T17:35:00Z",
                "actual_analysis_output_time": "2026-08-11T17:35:00Z",
                "forecast_downloads": 0,
            },
        )
        bundle = IcaoProductBundle(
            products=products,
            status=status,
            kp_storm_eligible=None,
        )
        summary = build_icao_summary(products, pd.DataFrame(), eligible=None)
        data = products.copy()
        app = AppTest.from_file(app_path, default_timeout=30).run()
        app.session_state["data"] = data
        app.session_state["status"] = bundle.status
        app.session_state["icao_bundle"] = bundle
        app.session_state["icao_summary"] = summary

        app = app.run(timeout=30)

        headings = [item.value for item in app.subheader]
        expander_labels = [item.label for item in app.expander]
        markdown = "\n".join(str(item.value) for item in app.markdown)
        self.assertIn("Evidence completeness", headings)
        self.assertIn("Standalone HF Communication Engineering Study", headings)
        self.assertIn("Engineering Impact: HF Communication Coverage", headings)
        self.assertNotIn("Open standalone study details", expander_labels)
        self.assertIn("How to interpret this HF case study", expander_labels)
        self.assertIn("Trace integration status", expander_labels)
        self.assertIn("Data Completeness", markdown)
        self.assertFalse(app.exception, [item.value for item in app.exception])

    def test_summary_table_normalises_mixed_values_before_arrow_serialisation(self):
        from app import _style_pecasus_table

        summary = pd.DataFrame({
            "Indicator": ["Vertical TEC", "Post-Storm Depression"],
            "Latest value": [100.0, "N/A"],
            "Status": ["OK", "UNAVAILABLE"],
        })

        styled = _style_pecasus_table(summary)

        self.assertTrue(
            styled.data.applymap(lambda value: isinstance(value, str)).all().all()
        )


if __name__ == "__main__":
    unittest.main()
