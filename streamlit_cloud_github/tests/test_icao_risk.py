import os
import sys
import unittest

import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class IcaoRiskTest(unittest.TestCase):
    def test_tec_threshold_boundaries(self):
        from icao_risk import classify_tec

        self.assertEqual(classify_tec(124.999), "OK")
        self.assertEqual(classify_tec(125), "MODERATE")
        self.assertEqual(classify_tec(174.999), "MODERATE")
        self.assertEqual(classify_tec(175), "SEVERE")

    def test_kp_auroral_absorption_proxy_boundaries(self):
        from icao_risk import classify_auroral_absorption

        self.assertEqual(classify_auroral_absorption(7.999), "OK")
        self.assertEqual(classify_auroral_absorption(8), "MODERATE")
        self.assertEqual(classify_auroral_absorption(8.999), "MODERATE")
        self.assertEqual(classify_auroral_absorption(9), "SEVERE")

    def test_post_storm_depression_percent_and_invalid_reference(self):
        from icao_risk import calculate_psd_percent

        self.assertEqual(calculate_psd_percent(60, 100), 40.0)
        self.assertEqual(calculate_psd_percent(120, 100), 0.0)
        self.assertIsNone(calculate_psd_percent(20, 0))
        self.assertIsNone(calculate_psd_percent(20, None))

    def test_psd_risk_requires_recent_storm_eligibility(self):
        from icao_risk import classify_psd

        self.assertEqual(classify_psd(60, kp_storm_eligible=False), "OK")
        self.assertEqual(classify_psd(60, kp_storm_eligible=None), "UNAVAILABLE")
        self.assertEqual(classify_psd(30, kp_storm_eligible=True), "MODERATE")
        self.assertEqual(classify_psd(50, kp_storm_eligible=True), "SEVERE")

    def test_invalid_classifications_and_worst_category(self):
        from icao_risk import (
            classify_auroral_absorption,
            classify_psd,
            classify_tec,
            worst_category,
        )

        self.assertEqual(classify_tec(float("nan")), "UNAVAILABLE")
        self.assertEqual(classify_auroral_absorption(float("inf")), "UNAVAILABLE")
        self.assertEqual(classify_psd(None, True), "UNAVAILABLE")
        self.assertEqual(worst_category(["OK", "SEVERE", "MODERATE"]), "SEVERE")
        self.assertEqual(worst_category(["UNAVAILABLE", "OK"]), "OK")
        self.assertEqual(worst_category(["unknown"]), "UNAVAILABLE")

    def test_categorical_cells_support_only_spatial_icao_products(self):
        from icao_risk import ICAO_COLORS, build_categorical_cells

        products = pd.DataFrame([
            {"indicator": "Vertical TEC", "horizon": "Latest", "time": "2026-06-24T12:00:00Z", "lat": 50, "lon": 1, "value": 180},
            {"indicator": "Vertical TEC", "horizon": "+30 min", "product_kind": "forecast_30", "time": "2026-06-24T12:30:00Z", "lat": 51, "lon": 2, "value": 130, "source": "SERENE AIDA forecast"},
            {"indicator": "Kp", "horizon": "Latest", "time": "2026-06-24T12:00:00Z", "lat": 50, "lon": 1, "value": 9},
        ])

        cells = build_categorical_cells(products, "Vertical TEC", "Latest")

        self.assertEqual(len(cells), 1)
        self.assertEqual(cells.iloc[0]["status"], "SEVERE")
        self.assertEqual(cells.iloc[0]["color"], ICAO_COLORS["SEVERE"])
        self.assertTrue(build_categorical_cells(products, "Kp", "Latest").empty)
        self.assertTrue(build_categorical_cells(products, "Vertical TEC", "+1h").empty)

    def test_summary_and_cells_support_three_and_six_hour_forecasts(self):
        from icao_risk import build_categorical_cells, build_icao_summary

        products = pd.DataFrame([
            {
                "variable": "TEC",
                "product_kind": "forecast_180",
                "time": "2026-08-12T15:00:00Z",
                "lat": 50,
                "lon": 1,
                "value": 160.0,
                "source": "SERENE AIDA forecast",
            },
            {
                "variable": "TEC",
                "product_kind": "forecast_360",
                "time": "2026-08-12T18:00:00Z",
                "lat": 50,
                "lon": 1,
                "value": 180.0,
                "source": "SERENE AIDA forecast",
            },
            {
                "variable": "MUF3000F2",
                "product_kind": "forecast_180",
                "time": "2026-08-12T15:00:00Z",
                "lat": 50,
                "lon": 1,
                "psd_percent": 35.0,
                "source": "SERENE AIDA forecast",
            },
            {
                "variable": "MUF3000F2",
                "product_kind": "forecast_360",
                "time": "2026-08-12T18:00:00Z",
                "lat": 50,
                "lon": 1,
                "psd_percent": 55.0,
                "source": "SERENE AIDA forecast",
            },
        ])
        horizons = pd.DataFrame([
            {
                "horizon_minutes": 180,
                "value": 8.2,
                "evidence_role": "official_forecast",
                "source": "GFZ official PAGER/SWIFT ensemble forecast",
            },
            {
                "horizon_minutes": 360,
                "value": 9.0,
                "evidence_role": "official_forecast",
                "source": "GFZ official PAGER/SWIFT ensemble forecast",
            },
        ])

        summary = build_icao_summary(
            products, pd.DataFrame(), eligible=True, kp_horizons=horizons
        )
        tec = summary.loc[summary["Indicator"] == "Vertical TEC"].iloc[0]
        psd = summary.loc[
            summary["Indicator"] == "Post-Storm Depression"
        ].iloc[0]
        kp = summary.loc[summary["Indicator"] == "Auroral Absorption"].iloc[0]
        cells = build_categorical_cells(products, "Vertical TEC", "+6h")

        self.assertEqual(tec["+3h forecast"], 160.0)
        self.assertEqual(tec["+3h status"], "MODERATE")
        self.assertEqual(tec["+3h source"], "SERENE official forecast")
        self.assertEqual(psd["+6h forecast"], 55.0)
        self.assertEqual(psd["+6h status"], "SEVERE")
        self.assertEqual(psd["+6h source"], "SERENE official forecast")
        self.assertEqual(kp["+3h forecast"], 8.2)
        self.assertEqual(kp["+3h status"], "MODERATE")
        self.assertEqual(kp["+6h forecast"], 9.0)
        self.assertEqual(kp["+6h status"], "SEVERE")
        self.assertEqual(
            kp["+6h source"], "GFZ official PAGER/SWIFT ensemble forecast"
        )
        self.assertEqual(cells.iloc[0]["category"], "SEVERE")

    def test_three_and_six_hour_aliases_accept_internal_whitespace(self):
        from icao_risk import build_categorical_cells

        for raw_horizon, canonical_horizon, value in (
            ("+3 h", "+3h", 160.0),
            ("+6 h", "+6h", 180.0),
            ("+3\t h", "+3h", 150.0),
        ):
            products = pd.DataFrame([{
                "indicator": "Vertical TEC",
                "horizon": raw_horizon,
                "product_kind": f"forecast_{180 if canonical_horizon == '+3h' else 360}",
                "lat": 50,
                "lon": 1,
                "value": value,
                "source": "SERENE AIDA forecast",
            }])

            cells = build_categorical_cells(
                products, "Vertical TEC", canonical_horizon
            )

            self.assertEqual(len(cells), 1)
            self.assertEqual(cells.iloc[0]["horizon"], canonical_horizon)

    def test_new_horizons_keep_official_and_missing_evidence_independent(self):
        from icao_risk import build_categorical_cells, build_icao_summary

        products = pd.DataFrame([
            {
                "variable": "TEC",
                "product_kind": "analysis",
                "time": "2026-08-12T12:00:00Z",
                "lat": 50,
                "lon": 1,
                "value": 130.0,
                "source": "SERENE AIDA analysis",
            },
            {
                "variable": "TEC",
                "product_kind": "forecast_180",
                "time": "2026-08-12T15:00:00Z",
                "lat": 50,
                "lon": 1,
                "value": 160.0,
                "source": "SERENE AIDA forecast",
            },
        ])
        horizons = pd.DataFrame([{
            "horizon_minutes": 180,
            "value": 8.2,
            "evidence_role": "official_forecast",
            "source": "GFZ official PAGER/SWIFT ensemble forecast",
        }])

        summary = build_icao_summary(
            products, pd.DataFrame(), eligible=False, kp_horizons=horizons
        )
        tec = summary.loc[summary["Indicator"] == "Vertical TEC"].iloc[0]
        kp = summary.loc[summary["Indicator"] == "Auroral Absorption"].iloc[0]

        self.assertEqual(tec["+3h forecast"], 160.0)
        self.assertEqual(tec["+3h source"], "SERENE official forecast")
        self.assertIsNone(tec["+6h forecast"])
        self.assertEqual(tec["+6h status"], "UNAVAILABLE")
        self.assertEqual(tec["+6h source"], "Unavailable")
        self.assertTrue(
            build_categorical_cells(products, "Vertical TEC", "+6h").empty
        )
        self.assertEqual(kp["+3h forecast"], 8.2)
        self.assertEqual(kp["+3h status"], "MODERATE")
        self.assertEqual(kp["+6h forecast"], "N/A")
        self.assertEqual(kp["+6h status"], "UNAVAILABLE")
        self.assertEqual(kp["+6h source"], "Unavailable")

    def test_post_storm_cells_apply_eligibility_gate(self):
        from icao_risk import build_categorical_cells

        products = pd.DataFrame([
            {"indicator": "Post-Storm Depression", "horizon": "+30 min", "product_kind": "forecast_30", "lat": 50, "lon": 1, "reference": 100, "current": 40, "source": "SERENE AIDA forecast"},
        ])

        gated = build_categorical_cells(products, "Post-Storm Depression", "+30 min")
        eligible = build_categorical_cells(
            products, "Post-Storm Depression", "+30 min", kp_storm_eligible=True
        )

        self.assertEqual(gated.iloc[0]["display_value"], 60.0)
        self.assertEqual(gated.iloc[0]["status"], "OK")
        self.assertEqual(eligible.iloc[0]["status"], "SEVERE")

    def test_latest_cells_exclude_older_product_times(self):
        from icao_risk import build_categorical_cells

        products = pd.DataFrame([
            {"variable": "TEC", "time": "2026-06-24T11:00:00Z", "lat": 50, "lon": 1, "value": 180},
            {"variable": "TEC", "time": "2026-06-24T12:00:00Z", "lat": 50, "lon": 1, "value": 130},
        ])

        cells = build_categorical_cells(products, "Vertical TEC", "Latest")

        self.assertEqual(len(cells), 1)
        self.assertEqual(cells.iloc[0]["display_value"], 130)

    def test_invalid_spatial_values_are_retained_as_unavailable(self):
        from icao_risk import ICAO_COLORS, build_categorical_cells

        products = pd.DataFrame([
            {"variable": "TEC", "product_kind": "analysis", "time": "2026-06-24T12:00:00Z", "lat": 50, "lon": 1, "value": float("nan")},
            {"variable": "TEC", "product_kind": "analysis", "time": "2026-06-24T12:00:00Z", "lat": 51, "lon": 2, "value": float("inf")},
        ])

        cells = build_categorical_cells(products, "Vertical TEC", "Latest")

        self.assertEqual(len(cells), 2)
        self.assertTrue((cells["display_value"] == "N/A").all())
        self.assertTrue((cells["category"] == "UNAVAILABLE").all())
        self.assertTrue((cells["color"] == ICAO_COLORS["UNAVAILABLE"]).all())

    def test_cells_include_threshold_and_product_state_for_hover(self):
        from icao_risk import build_categorical_cells

        products = pd.DataFrame([
            {"variable": "TEC", "product_kind": "analysis", "time": "2026-06-24T12:00:00Z", "lat": 50, "lon": 1, "value": 130},
            {"variable": "TEC", "product_kind": "forecast_30", "time": "2026-06-24T12:30:00Z", "lat": 50, "lon": 1, "value": 150, "source": "SERENE AIDA forecast"},
        ])

        latest = build_categorical_cells(products, "Vertical TEC", "Latest")
        forecast = build_categorical_cells(products, "Vertical TEC", "+30 min")

        self.assertIn("threshold_explanation", latest.columns)
        self.assertIn("product_state", latest.columns)
        self.assertIn("125", latest.iloc[0]["threshold_explanation"])
        self.assertEqual(latest.iloc[0]["product_state"], "analysis")
        self.assertEqual(forecast.iloc[0]["product_state"], "serene official forecast")

    def test_summary_uses_regional_max_and_keeps_missing_values_na(self):
        from icao_risk import build_icao_summary

        products = pd.DataFrame([
            {"indicator": "Vertical TEC", "horizon": "Latest", "time": "2026-06-24T12:00:00Z", "lat": 50, "lon": 1, "value": 120},
            {"indicator": "Vertical TEC", "horizon": "Latest", "time": "2026-06-24T12:00:00Z", "lat": 51, "lon": 2, "value": 180},
            {"indicator": "Vertical TEC", "horizon": "Max3h", "lat": 50, "lon": 1, "value": 160},
            {"indicator": "Vertical TEC", "horizon": "+30 min", "product_kind": "forecast_30", "lat": 50, "lon": 1, "value": 130, "source": "SERENE AIDA forecast"},
        ])
        indices = pd.DataFrame([
            {"variable": "Kp", "time": "2026-06-24T12:00:00Z", "value": 8.5},
            {"variable": "Kp", "time": "2026-06-24T09:00:00Z", "value": 7.0},
        ])

        summary = build_icao_summary(products, indices, eligible=False)
        tec = summary.loc[summary["Indicator"] == "Vertical TEC"].iloc[0]
        kp = summary.loc[summary["Indicator"] == "Auroral Absorption"].iloc[0]

        self.assertEqual(tec["Latest value"], 180)
        self.assertEqual(tec["Status"], "SEVERE")
        self.assertEqual(tec["Max-3h value"], 160)
        self.assertEqual(tec["Max-3h status"], "MODERATE")
        self.assertEqual(tec["+30 min forecast"], 130)
        self.assertEqual(tec["+30 min status"], "MODERATE")
        self.assertEqual(tec["+30 min source"], "SERENE official forecast")
        self.assertEqual(tec["+90 min source"], "Unavailable")
        self.assertEqual(kp["Latest value"], 8.5)
        self.assertEqual(kp["Status"], "MODERATE")
        self.assertEqual(kp["Max-3h value"], 8.5)
        self.assertEqual(kp["Max-3h status"], "MODERATE")
        self.assertEqual(kp["+30 min forecast"], "N/A")
        self.assertEqual(kp["+90 min forecast"], "N/A")

    def test_latest_summary_uses_latest_timestamp_before_regional_max(self):
        from icao_risk import build_icao_summary

        products = pd.DataFrame([
            {"variable": "TEC", "time": "2026-06-24T11:00:00Z", "lat": 50, "lon": 1, "value": 190},
            {"variable": "TEC", "time": "2026-06-24T12:00:00Z", "lat": 50, "lon": 1, "value": 140},
            {"variable": "TEC", "time": "2026-06-24T12:00:00Z", "lat": 51, "lon": 2, "value": 150},
        ])

        summary = build_icao_summary(products, pd.DataFrame(), eligible=False)
        tec = summary.loc[summary["Indicator"] == "Vertical TEC"].iloc[0]

        self.assertEqual(tec["Latest value"], 150)
        self.assertEqual(tec["Time UTC"], "2026-06-24 12:00 UTC")

    def test_kp_max3h_is_inclusive_window_ending_at_latest_kp(self):
        from icao_risk import build_icao_summary

        indices = pd.DataFrame([
            {"variable": "Kp", "time": "2026-06-24T08:59:59Z", "value": 9.7},
            {"variable": "Kp", "time": "2026-06-24T09:00:00Z", "value": 9.0},
            {"variable": "Kp", "time": "2026-06-24T11:00:00Z", "value": 8.8},
            {"variable": "Kp", "time": "2026-06-24T12:00:00Z", "value": 8.5},
        ])

        summary = build_icao_summary(pd.DataFrame(), indices, eligible=False)
        kp = summary.loc[summary["Indicator"] == "Auroral Absorption"].iloc[0]

        self.assertEqual(kp["Latest value"], 8.5)
        self.assertEqual(kp["Max-3h value"], 9.0)
        self.assertEqual(kp["Max-3h status"], "SEVERE")
        self.assertEqual(kp["+30 min forecast"], "N/A")
        self.assertEqual(kp["+90 min forecast"], "N/A")

    def test_kp_official_horizons_use_median_without_hiding_ensemble_tail(self):
        from icao_risk import build_icao_summary

        indices = pd.DataFrame([{
            "variable": "Kp",
            "time": "2026-08-12T12:00:00Z",
            "value": 2.0,
            "source": "GFZ Kp/ap JSON service",
        }])
        horizons = pd.DataFrame([
            {
                "horizon_minutes": 30,
                "value": 7.5,
                "evidence_role": "official_forecast",
                "source": "GFZ official PAGER/SWIFT ensemble forecast",
                "ensemble_maximum": 8.4,
                "probability_kp_ge_8": 0.2,
                "data_status": "forecast",
                "issue_time": "2026-08-12T13:05:20Z",
            },
            {
                "horizon_minutes": 90,
                "value": 8.2,
                "evidence_role": "official_forecast",
                "source": "GFZ official PAGER/SWIFT ensemble forecast",
                "ensemble_maximum": 9.0,
                "probability_kp_ge_8": 0.7,
                "data_status": "forecast",
                "issue_time": "2026-08-12T13:05:20Z",
            },
        ])

        summary = build_icao_summary(
            pd.DataFrame(), indices, eligible=False, kp_horizons=horizons
        )
        kp = summary.loc[summary["Indicator"] == "Auroral Absorption"].iloc[0]

        self.assertEqual(kp["Latest value"], 2.0)
        self.assertEqual(kp["Status"], "OK")
        self.assertEqual(kp["+30 min forecast"], 7.5)
        self.assertEqual(kp["+30 min status"], "OK")
        self.assertEqual(kp["+90 min forecast"], 8.2)
        self.assertEqual(kp["+90 min status"], "MODERATE")
        self.assertEqual(
            kp["+30 min source"],
            "GFZ official PAGER/SWIFT ensemble forecast",
        )
        self.assertIn("ensemble maximum Kp 8.4", kp["Source / Availability"])
        self.assertIn("P(Kp >= 8) 20%", kp["Source / Availability"])

    def test_kp_observed_horizon_is_labelled_backtesting_not_forecast(self):
        from icao_risk import build_icao_summary

        horizons = pd.DataFrame([{
            "horizon_minutes": 30,
            "value": 9.0,
            "evidence_role": "observed_backtesting",
            "source": "GFZ observed outcome — backtesting only",
            "ensemble_maximum": float("nan"),
            "probability_kp_ge_8": float("nan"),
            "data_status": "preliminary",
        }])

        summary = build_icao_summary(
            pd.DataFrame(), pd.DataFrame(), kp_horizons=horizons
        )
        kp = summary.loc[summary["Indicator"] == "Auroral Absorption"].iloc[0]

        self.assertEqual(kp["+30 min forecast"], 9.0)
        self.assertEqual(kp["+30 min status"], "SEVERE")
        self.assertEqual(
            kp["+30 min source"],
            "GFZ observed outcome — backtesting only",
        )
        self.assertIn("not a forecast", kp["Source / Availability"])
        self.assertEqual(kp["+90 min status"], "UNAVAILABLE")

    def test_loader_product_kind_and_variables_map_to_icao_products(self):
        from icao_risk import build_categorical_cells, build_icao_summary

        products = pd.DataFrame([
            {"variable": "TEC", "product_kind": "analysis", "time": "2026-06-24T12:00:00Z", "lat": 50, "lon": 1, "value": 130, "source": "SERENE AIDA"},
            {"variable": "TEC", "product_kind": "rolling", "time": "2026-06-24T11:00:00Z", "lat": 50, "lon": 1, "value": 180, "source": "SERENE AIDA"},
            {"variable": "TEC", "product_kind": "forecast_30", "time": "2026-06-24T12:30:00Z", "lat": 50, "lon": 1, "value": 150, "source": "SERENE AIDA forecast"},
            {"variable": "MUF3000F2", "product_kind": "analysis", "time": "2026-06-24T12:00:00Z", "lat": 50, "lon": 1, "value": 8, "psd_percent": 40, "source": "SERENE AIDA"},
        ])

        tec_forecast = build_categorical_cells(products, "Vertical TEC", "+30 min")
        psd_latest = build_categorical_cells(
            products, "Post-Storm Depression", "Latest", kp_storm_eligible=True
        )
        summary = build_icao_summary(products, pd.DataFrame(), eligible=True)
        tec = summary.loc[summary["Indicator"] == "Vertical TEC"].iloc[0]

        self.assertEqual(tec_forecast.iloc[0]["category"], "MODERATE")
        self.assertEqual(psd_latest.iloc[0]["category"], "MODERATE")
        self.assertEqual(tec["Max-3h value"], 180)
        self.assertEqual(tec["+30 min forecast"], 150)
        self.assertEqual(tec["+30 min source"], "SERENE official forecast")
        self.assertEqual(tec["+90 min source"], "Unavailable")

    def test_missing_psd_baseline_never_treats_muf_mhz_as_percent(self):
        from icao_risk import build_categorical_cells, build_icao_summary

        products = pd.DataFrame([{
            "variable": "MUF3000F2",
            "product_kind": "analysis",
            "time": "2026-06-24T12:00:00Z",
            "lat": 50,
            "lon": 1,
            "value": 8.0,
            "reference_value": pd.NA,
            "psd_percent": pd.NA,
            "source": "SERENE AIDA",
        }])

        cells = build_categorical_cells(
            products, "Post-Storm Depression", "Latest", kp_storm_eligible=True
        )
        summary = build_icao_summary(products, pd.DataFrame(), eligible=True)
        psd = summary.loc[summary["Indicator"] == "Post-Storm Depression"].iloc[0]

        self.assertEqual(cells.iloc[0]["display_value"], "N/A")
        self.assertEqual(cells.iloc[0]["category"], "UNAVAILABLE")
        self.assertEqual(psd["Latest value"], "N/A")
        self.assertEqual(psd["Status"], "UNAVAILABLE")

    def test_summary_table_contains_only_serene_supported_indicators(self):
        from icao_risk import build_icao_summary

        products = pd.DataFrame([
            {
                "variable": "TEC",
                "product_kind": "analysis",
                "time": "2026-06-24T12:00:00Z",
                "lat": 50,
                "lon": 1,
                "value": 130,
                "source": "SERENE AIDA TEC",
            },
            {
                "variable": "MUF3000F2",
                "product_kind": "analysis",
                "time": "2026-06-24T12:00:00Z",
                "lat": 50,
                "lon": 1,
                "value": 8,
                "psd_percent": 35,
                "source": "SERENE AIDA MUF3000F2",
            },
        ])
        indices = pd.DataFrame([
            {
                "variable": "Kp",
                "time": "2026-06-24T12:00:00Z",
                "value": 8.2,
                "source": "SERENE Kp/ap",
            }
        ])

        summary = build_icao_summary(products, indices, eligible=True)

        self.assertEqual(list(summary.columns), [
            "Domain",
            "Indicator",
            "Moderate threshold",
            "Severe threshold",
            "Time UTC",
            "Latest value",
            "Latest status",
            "Status",
            "Alert",
            "Max-3h value",
            "Max-3h status",
            "+30 min forecast",
            "+30 min status",
            "+30 min source",
            "+90 min forecast",
            "+90 min status",
            "+90 min source",
            "+3h forecast",
            "+3h status",
            "+3h source",
            "+6h forecast",
            "+6h status",
            "+6h source",
            "Source / Availability",
        ])
        self.assertEqual(set(summary["Indicator"]), {
            "Vertical TEC",
            "Auroral Absorption",
            "Post-Storm Depression",
        })
        tec = summary.loc[summary["Indicator"] == "Vertical TEC"].iloc[0]
        psd = summary.loc[summary["Indicator"] == "Post-Storm Depression"].iloc[0]
        kp = summary.loc[summary["Indicator"] == "Auroral Absorption"].iloc[0]

        self.assertNotIn("Amplitude Scintillation", set(summary["Indicator"]))
        self.assertNotIn("Phase Scintillation", set(summary["Indicator"]))
        self.assertNotIn("Polar Cap Absorption", set(summary["Indicator"]))
        self.assertNotIn("Shortwave Fadeout", set(summary["Indicator"]))
        self.assertEqual(set(summary["Domain"]), {"GNSS", "HF COM"})
        self.assertEqual(tec["Status"], "MODERATE")
        self.assertEqual(psd["Status"], "MODERATE")
        self.assertEqual(kp["Status"], "MODERATE")
        self.assertEqual(kp["+30 min status"], "UNAVAILABLE")
        self.assertEqual(kp["+90 min source"], "Unavailable")
        self.assertEqual(kp["+30 min source"], "Unavailable")

    def test_spatial_forecast_matrix_is_official_or_unavailable(self):
        from icao_risk import build_categorical_cells, build_icao_summary

        observations = pd.DataFrame([
            {
                "variable": "TEC",
                "product_kind": "rolling",
                "time": "2026-06-24T09:00:00Z",
                "lat": 50,
                "lon": 1,
                "value": 100,
                "source": "SERENE AIDA analysis",
            },
            {
                "variable": "TEC",
                "product_kind": "analysis",
                "time": "2026-06-24T12:00:00Z",
                "lat": 50,
                "lon": 1,
                "value": 130,
                "source": "SERENE AIDA analysis",
            },
            {
                "variable": "MUF3000F2",
                "product_kind": "rolling",
                "time": "2026-06-24T09:00:00Z",
                "lat": 50,
                "lon": 1,
                "value": 12,
                "psd_percent": 20,
                "source": "SERENE AIDA analysis",
            },
            {
                "variable": "MUF3000F2",
                "product_kind": "analysis",
                "time": "2026-06-24T12:00:00Z",
                "lat": 50,
                "lon": 1,
                "value": 8,
                "psd_percent": 35,
                "source": "SERENE AIDA analysis",
            },
        ])
        horizon_cases = (
            ("+30 min", 30, 130.0, 35.0),
            ("+90 min", 90, 140.0, 40.0),
            ("+3h", 180, 160.0, 45.0),
            ("+6h", 360, 180.0, 55.0),
        )
        forecast_rows = []
        for label, minutes, tec_value, psd_value in horizon_cases:
            forecast_rows.extend([
                {
                    "variable": "TEC",
                    "product_kind": f"forecast_{minutes}",
                    "time": "2026-06-24T12:30:00Z",
                    "lat": 50,
                    "lon": 1,
                    "value": tec_value,
                    "source": "SERENE AIDA forecast",
                },
                {
                    "variable": "MUF3000F2",
                    "product_kind": f"forecast_{minutes}",
                    "time": "2026-06-24T12:30:00Z",
                    "lat": 50,
                    "lon": 1,
                    "value": 8,
                    "psd_percent": psd_value,
                    "source": "SERENE AIDA forecast",
                },
            ])

        official_products = pd.concat(
            [observations, pd.DataFrame(forecast_rows)], ignore_index=True
        )
        official_summary = build_icao_summary(
            official_products, pd.DataFrame(), eligible=True
        )
        missing_summary = build_icao_summary(
            observations, pd.DataFrame(), eligible=True
        )

        for indicator, expected_values in (
            ("Vertical TEC", [130.0, 140.0, 160.0, 180.0]),
            ("Post-Storm Depression", [35.0, 40.0, 45.0, 55.0]),
        ):
            official_row = official_summary.loc[
                official_summary["Indicator"] == indicator
            ].iloc[0]
            missing_row = missing_summary.loc[
                missing_summary["Indicator"] == indicator
            ].iloc[0]
            for (label, _minutes, _tec, _psd), expected in zip(
                horizon_cases, expected_values
            ):
                with self.subTest(indicator=indicator, horizon=label, state="official"):
                    self.assertEqual(official_row[f"{label} forecast"], expected)
                    self.assertEqual(
                        official_row[f"{label} source"], "SERENE official forecast"
                    )
                    self.assertFalse(
                        build_categorical_cells(
                            official_products,
                            indicator,
                            label,
                            kp_storm_eligible=True,
                        ).empty
                    )
                with self.subTest(indicator=indicator, horizon=label, state="missing"):
                    self.assertIsNone(missing_row[f"{label} forecast"])
                    self.assertEqual(missing_row[f"{label} status"], "UNAVAILABLE")
                    self.assertEqual(missing_row[f"{label} source"], "Unavailable")
                    self.assertTrue(
                        build_categorical_cells(
                            observations,
                            indicator,
                            label,
                            kp_storm_eligible=True,
                        ).empty
                    )

    def test_spatial_forecasts_reject_non_official_or_mismatched_provenance(self):
        from icao_risk import build_categorical_cells, build_icao_summary

        invalid_products = pd.DataFrame([
            {
                "indicator": "Vertical TEC",
                "horizon": "+30 min",
                "product_kind": "analysis",
                "lat": 50,
                "lon": 1,
                "value": 180.0,
                "source": "SERENE AIDA analysis",
            },
            {
                "indicator": "Post-Storm Depression",
                "horizon": "+90 min",
                "product_kind": "forecast_90",
                "lat": 50,
                "lon": 1,
                "psd_percent": 55.0,
                "source": "Dashboard-generated persistence forecast",
            },
            {
                "indicator": "Vertical TEC",
                "horizon": "+3h",
                "product_kind": "forecast_180",
                "lat": 50,
                "lon": 1,
                "value": 180.0,
                "source": "SERENE-looking synthetic forecast",
            },
        ])

        summary = build_icao_summary(
            invalid_products, pd.DataFrame(), eligible=True
        )

        for indicator, horizon in (
            ("Vertical TEC", "+30 min"),
            ("Post-Storm Depression", "+90 min"),
            ("Vertical TEC", "+3h"),
        ):
            row = summary.loc[summary["Indicator"] == indicator].iloc[0]
            self.assertIsNone(row[f"{horizon} forecast"])
            self.assertEqual(row[f"{horizon} status"], "UNAVAILABLE")
            self.assertEqual(row[f"{horizon} source"], "Unavailable")
            self.assertTrue(
                build_categorical_cells(
                    invalid_products,
                    indicator,
                    horizon,
                    kp_storm_eligible=True,
                ).empty
            )

    def test_official_psd_forecast_without_baseline_is_unavailable_and_map_empty(self):
        from icao_risk import build_categorical_cells, build_icao_summary

        products = pd.DataFrame([{
            "variable": "MUF3000F2",
            "product_kind": "forecast_360",
            "time": "2026-06-24T18:00:00Z",
            "lat": 50,
            "lon": 1,
            "value": 8.0,
            "reference_value": pd.NA,
            "psd_percent": pd.NA,
            "source": "SERENE AIDA forecast",
        }])

        summary = build_icao_summary(products, pd.DataFrame(), eligible=True)
        psd = summary.loc[
            summary["Indicator"] == "Post-Storm Depression"
        ].iloc[0]

        self.assertIsNone(psd["+6h forecast"])
        self.assertEqual(psd["+6h status"], "UNAVAILABLE")
        self.assertEqual(psd["+6h source"], "Unavailable")
        self.assertTrue(
            build_categorical_cells(
                products,
                "Post-Storm Depression",
                "+6h",
                kp_storm_eligible=True,
            ).empty
        )

    def test_overall_risk_cards_use_worst_available_status(self):
        from icao_risk import build_overall_risk_cards

        summary = pd.DataFrame([
            {"Domain": "GNSS", "Status": "UNAVAILABLE"},
            {"Domain": "GNSS", "Status": "MODERATE"},
            {"Domain": "HF COM", "Status": "OK"},
            {"Domain": "HF COM", "Status": "SEVERE"},
        ])

        cards = build_overall_risk_cards(summary)

        self.assertEqual(cards["GNSS Risk"], "MODERATE")
        self.assertEqual(cards["HF COM Risk"], "SEVERE")
        self.assertEqual(
            set(cards),
            {"GNSS Risk", "HF COM Risk", "Overall Risk", "Data Completeness"},
        )
        self.assertEqual(cards["Overall Risk"], "SEVERE + PARTIAL DATA")
        self.assertEqual(cards["Data Completeness"], "PARTIAL")

    def test_partial_inputs_do_not_produce_unqualified_overall_ok(self):
        from icao_risk import build_overall_risk_cards, build_evidence_completeness

        summary = pd.DataFrame([
            {"Domain": "GNSS", "Indicator": "Vertical TEC", "Status": "OK"},
            {
                "Domain": "HF COM",
                "Indicator": "Post-Storm Depression",
                "Status": "UNAVAILABLE",
            },
            {
                "Domain": "HF COM",
                "Indicator": "Auroral Absorption",
                "Status": "UNAVAILABLE",
            },
        ])

        cards = build_overall_risk_cards(summary)
        completeness = build_evidence_completeness(summary)

        self.assertEqual(cards["Overall Risk"], "PARTIAL DATA")
        self.assertEqual(cards["Data Completeness"], "PARTIAL")
        self.assertEqual(completeness["available"], 1)
        self.assertEqual(completeness["required"], 3)
        self.assertEqual(completeness["percent"], 33)
        self.assertEqual(
            completeness["missing"],
            ["Post-Storm Depression", "Auroral Absorption"],
        )

    def test_severe_risk_is_preserved_when_other_evidence_is_missing(self):
        from icao_risk import build_overall_risk_cards

        summary = pd.DataFrame([
            {"Domain": "GNSS", "Indicator": "Vertical TEC", "Status": "SEVERE"},
            {
                "Domain": "HF COM",
                "Indicator": "Post-Storm Depression",
                "Status": "UNAVAILABLE",
            },
        ])

        cards = build_overall_risk_cards(summary)

        self.assertEqual(cards["Overall Risk"], "SEVERE + PARTIAL DATA")


if __name__ == "__main__":
    unittest.main()
