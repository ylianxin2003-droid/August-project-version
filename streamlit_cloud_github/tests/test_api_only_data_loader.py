import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


GLOBAL_REGION = {"lat_min": -90, "lat_max": 90, "lon_min": -180, "lon_max": 180}


def _fake_calculation(_payload, region, step, variables):
    variable = (variables or ["TEC"])[0]
    return pd.DataFrame([{
        "time": pd.Timestamp("2026-06-21T20:00:00Z"),
        "lat": float(region["lat_min"]),
        "lon": float(region["lon_min"]),
        "variable": variable,
        "value": float(step),
        "model": "AIDA",
        "source": "test upstream adapter",
    }])


class FakeRawClient:
    def __init__(self):
        self.download_requests = []
        self.index_requests = []

    def download_aida_raw_output(self, requested_time, latency):
        self.download_requests.append((requested_time, latency))
        return True, f"downloaded {requested_time or 'latest'}", b"raw-state"

    def download_aida_forecast(self, requested_time, latency, period_minutes):
        self.forecast_requests = getattr(self, "forecast_requests", [])
        self.forecast_requests.append((requested_time, latency, period_minutes))
        return True, f"forecast {period_minutes}", b"forecast-state"

    def fetch_kp_ap_indices(self, **kwargs):
        self.index_requests.append(kwargs)
        return False, "indices unavailable", pd.DataFrame()

    def fetch_gfz_kp_forecast(self):
        self.kp_forecast_requests = getattr(self, "kp_forecast_requests", 0) + 1
        return False, "Kp ensemble forecast unavailable", pd.DataFrame()


class ApiOnlyDataLoaderTest(unittest.TestCase):
    def test_kp_completeness_allows_one_interval_plus_publication_delay(self):
        import data_loader

        analysis = pd.Timestamp("2026-08-12T12:10:00Z")
        complete_times = pd.date_range(
            start="2026-08-08T12:00:00Z", periods=32, freq="3h", tz="UTC"
        )
        complete = pd.DataFrame({
            "time": complete_times,
            "variable": "Kp",
            "value": 2.0,
        })
        stale = complete.copy()
        stale["time"] = stale["time"] - pd.Timedelta(hours=3)

        self.assertTrue(data_loader._kp_history_is_complete(complete, analysis))
        self.assertFalse(data_loader._kp_history_is_complete(stale, analysis))

    def test_kp_horizon_resolver_uses_observed_outcomes_for_backtesting(self):
        import data_loader

        analysis = pd.Timestamp("2026-07-01T05:55:00Z")
        observed = pd.DataFrame([{
            "time": pd.Timestamp("2026-07-01T06:00:00Z"),
            "variable": "Kp",
            "value": 1.333,
            "source": "GFZ Kp/ap JSON service",
            "data_status": "preliminary",
        }, {
            "time": pd.Timestamp("2026-07-01T09:00:00Z"),
            "variable": "Kp",
            "value": 2.333,
            "source": "GFZ Kp/ap JSON service",
            "data_status": "preliminary",
        }])

        result = data_loader._resolve_kp_horizons(
            analysis,
            observed,
            pd.DataFrame(),
            now=pd.Timestamp("2026-07-02T00:00:00Z"),
        )

        self.assertEqual(result["horizon_minutes"].tolist(), [30, 90, 180, 360])
        self.assertEqual(result["interval_start"].tolist(), [
            pd.Timestamp("2026-07-01T06:00:00Z"),
            pd.Timestamp("2026-07-01T06:00:00Z"),
            pd.Timestamp("2026-07-01T06:00:00Z"),
            pd.Timestamp("2026-07-01T09:00:00Z"),
        ])
        self.assertEqual(result["value"].tolist(), [1.333, 1.333, 1.333, 2.333])
        self.assertEqual(
            result["evidence_role"].tolist(),
            ["observed_backtesting"] * 4,
        )
        self.assertTrue((
            result["source"] == "GFZ observed outcome — backtesting only"
        ).all())
        self.assertTrue(result["ensemble_maximum"].isna().all())

    def test_kp_horizon_resolver_uses_median_and_retains_uncertainty(self):
        import data_loader

        analysis = pd.Timestamp("2026-08-12T12:55:00Z")
        forecast = pd.DataFrame([{
            "interval_start": pd.Timestamp("2026-08-12T12:00:00Z"),
            "median": 7.5,
            "maximum": 8.4,
            "probability_kp_ge_8": 0.2,
            "source": "GFZ official PAGER/SWIFT ensemble forecast",
            "issue_time": pd.Timestamp("2026-08-12T12:45:00Z"),
        }, {
            "interval_start": pd.Timestamp("2026-08-12T15:00:00Z"),
            "median": 6.5,
            "maximum": 7.4,
            "probability_kp_ge_8": 0.1,
            "source": "GFZ official PAGER/SWIFT ensemble forecast",
            "issue_time": pd.Timestamp("2026-08-12T12:45:00Z"),
        }, {
            "interval_start": pd.Timestamp("2026-08-12T18:00:00Z"),
            "median": 5.5,
            "maximum": 6.4,
            "probability_kp_ge_8": 0.05,
            "source": "GFZ official PAGER/SWIFT ensemble forecast",
            "issue_time": pd.Timestamp("2026-08-12T12:45:00Z"),
        }])

        result = data_loader._resolve_kp_horizons(
            analysis,
            pd.DataFrame(),
            forecast,
            now=pd.Timestamp("2026-08-12T13:00:00Z"),
        )

        self.assertEqual(result["evidence_role"].tolist(), [
            "official_forecast", "official_forecast",
            "official_forecast", "official_forecast",
        ])
        self.assertEqual(result["horizon_minutes"].tolist(), [30, 90, 180, 360])
        self.assertEqual(result["interval_start"].tolist(), [
            pd.Timestamp("2026-08-12T12:00:00Z"),
            pd.Timestamp("2026-08-12T12:00:00Z"),
            pd.Timestamp("2026-08-12T15:00:00Z"),
            pd.Timestamp("2026-08-12T18:00:00Z"),
        ])
        self.assertEqual(result["value"].tolist(), [7.5, 7.5, 6.5, 5.5])
        self.assertEqual(result["ensemble_maximum"].tolist(), [8.4, 8.4, 7.4, 6.4])
        self.assertEqual(result["probability_kp_ge_8"].tolist(), [0.2, 0.2, 0.1, 0.05])

    def test_kp_horizons_resolve_independently_and_reject_stale_forecast(self):
        import data_loader

        analysis = pd.Timestamp("2026-08-12T13:00:00Z")
        observed = pd.DataFrame([{
            "time": pd.Timestamp("2026-08-12T12:00:00Z"),
            "variable": "Kp",
            "value": 3.0,
            "source": "GFZ Kp/ap JSON service",
            "data_status": "preliminary",
        }])
        fresh = pd.DataFrame([{
            "interval_start": pd.Timestamp("2026-08-12T12:00:00Z"),
            "median": 4.0,
            "maximum": 5.0,
            "probability_kp_ge_8": 0.01,
            "source": "GFZ official PAGER/SWIFT ensemble forecast",
            "issue_time": pd.Timestamp("2026-08-12T13:05:00Z"),
        }])

        mixed = data_loader._resolve_kp_horizons(
            analysis,
            observed,
            fresh,
            now=pd.Timestamp("2026-08-12T13:45:00Z"),
        )
        self.assertEqual(
            mixed["evidence_role"].tolist()[:2],
            ["observed_backtesting", "official_forecast"],
        )
        self.assertEqual(mixed["value"].tolist()[:2], [3.0, 4.0])

        stale = fresh.copy()
        stale["issue_time"] = pd.Timestamp("2026-08-12T08:00:00Z")
        rejected = data_loader._resolve_kp_horizons(
            analysis,
            observed,
            stale,
            now=pd.Timestamp("2026-08-12T13:45:00Z"),
        )
        plus_90 = rejected[rejected["horizon_minutes"] == 90].iloc[0]
        self.assertEqual(plus_90["evidence_role"], "unavailable")
        self.assertEqual(plus_90["data_status"], "unavailable")
        self.assertIn("fresh", plus_90["availability_reason"].lower())

    def test_loader_keeps_horizon_outcomes_separate_from_prior_96h_indices(self):
        import data_loader

        analysis = pd.Timestamp("2026-07-01T05:55:00Z")
        history_times = pd.date_range(
            end="2026-07-01T03:00:00Z", periods=32, freq="3h", tz="UTC"
        )
        history = pd.DataFrame([{
            "time": timestamp,
            "variable": "Kp",
            "value": 6.0 if position == 5 else 2.0,
            "source": "GFZ Kp/ap JSON service",
            "data_status": "definitive",
        } for position, timestamp in enumerate(history_times)])
        outcome = pd.DataFrame([{
            "time": pd.Timestamp("2026-07-01T06:00:00Z"),
            "variable": "Kp",
            "value": 9.0,
            "source": "GFZ Kp/ap JSON service",
            "data_status": "preliminary",
        }, {
            "time": pd.Timestamp("2026-07-01T09:00:00Z"),
            "variable": "Kp",
            "value": 9.0,
            "source": "GFZ Kp/ap JSON service",
            "data_status": "preliminary",
        }])

        class HistoricalHorizonClient(FakeRawClient):
            kp_ap_source_latest_time = pd.Timestamp("2026-07-01T03:00:00Z")
            kp_ap_data_statuses = ["definitive"]
            kp_ap_missing_indices = ["ap"]

            def fetch_kp_ap_indices(self, **kwargs):
                self.index_requests.append(kwargs)
                if kwargs["start_time"] == "2026-07-01T06:00:00+00:00":
                    return True, "target Kp loaded", outcome
                return True, "history Kp loaded", history

            def fetch_gfz_kp_forecast(self):
                self.kp_forecast_requests = getattr(
                    self, "kp_forecast_requests", 0
                ) + 1
                return True, "should not be used", pd.DataFrame([{
                    "interval_start": pd.Timestamp("2026-07-01T06:00:00Z"),
                    "median": 1.0,
                    "maximum": 1.0,
                    "probability_kp_ge_8": 0.0,
                    "issue_time": pd.Timestamp("2026-08-12T13:00:00Z"),
                }])

        client = HistoricalHorizonClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(
                data_loader, "calculate_aida_grid", side_effect=_fake_calculation
            ),
        ):
            bundle = data_loader.load_icao_products(
                analysis_time=analysis.isoformat(),
                variables=["TEC"],
                region=GLOBAL_REGION,
                grid_step=30,
                include_three_hour_window=False,
                include_psd_baseline=False,
            )

        self.assertEqual(bundle.indices["time"].max(), history_times.max())
        self.assertNotIn(9.0, bundle.indices["value"].tolist())
        self.assertEqual(bundle.kp_horizons["value"].tolist(), [9.0, 9.0, 9.0, 9.0])
        self.assertTrue(bundle.kp_storm_eligible)
        self.assertEqual(getattr(client, "kp_forecast_requests", 0), 0)
        self.assertEqual(
            bundle.status.metadata["kp_horizon_message"],
            "Kp +30/+90/+180/+360 minute horizon evidence resolved.",
        )

    def test_follow_latest_anchors_forecasts_to_time_inside_latest_state(self):
        import data_loader

        client = FakeRawClient()
        latest_cycle = pd.Timestamp("2026-08-12T10:35:00Z")
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "read_aida_state_time", return_value=latest_cycle),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            bundle = data_loader.load_icao_products(
                analysis_time="2026-08-12T10:20:00Z",
                variables=["TEC"],
                region=GLOBAL_REGION,
                grid_step=30,
                include_three_hour_window=False,
                include_psd_baseline=False,
                follow_latest=True,
            )

        self.assertEqual(client.download_requests, [(None, "ultra")])
        self.assertEqual(client.index_requests, [
            {
                "start_time": "2026-08-08T09:00:00+00:00",
                "end_time": "2026-08-12T10:35:00+00:00",
            },
            {
                "start_time": "2026-08-12T09:00:00+00:00",
                "end_time": "2026-08-12T15:00:00+00:00",
            },
        ])
        self.assertTrue(all(
            request[0] == "2026-08-12T10:35:00+00:00"
            for request in client.forecast_requests
        ))
        self.assertEqual(
            bundle.status.metadata["analysis_time"],
            "2026-08-12T10:35:00+00:00",
        )
        self.assertEqual(
            bundle.status.metadata["requested_analysis_time"],
            "2026-08-12T10:20:00+00:00",
        )
        self.assertEqual(
            bundle.status.metadata["analysis_anchor_source"],
            "latest_serene_state",
        )

    def test_historical_gfz_window_uses_selected_serene_analysis_time(self):
        import data_loader

        analysis = pd.Timestamp("2026-07-01T05:55:00Z")
        kp_times = pd.date_range(
            end="2026-07-01T03:00:00Z", periods=32, freq="3h", tz="UTC"
        )
        indices = pd.DataFrame([{
            "time": timestamp,
            "lat": None,
            "lon": None,
            "alt": None,
            "variable": "Kp",
            "value": 6.0 if position == 5 else 2.0,
            "model": "GFZ Geomagnetic Indices",
            "source": "GFZ Kp/ap JSON service",
            "data_status": "definitive",
        } for position, timestamp in enumerate(kp_times)])

        class HistoricalClient(FakeRawClient):
            kp_ap_source_latest_time = pd.Timestamp("2026-07-01T03:00:00Z")
            kp_ap_data_statuses = ["definitive"]
            kp_ap_missing_indices = ["ap"]

            def fetch_kp_ap_indices(self, **kwargs):
                self.index_requests.append(kwargs)
                return True, "Kp loaded; ap unavailable", indices

        client = HistoricalClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(
                data_loader, "calculate_aida_grid", side_effect=_fake_calculation
            ),
        ):
            bundle = data_loader.load_icao_products(
                analysis_time=analysis.isoformat(),
                variables=["TEC"],
                region=GLOBAL_REGION,
                grid_step=30,
                include_three_hour_window=False,
                include_psd_baseline=False,
            )

        self.assertEqual(client.index_requests, [
            {
                "start_time": "2026-06-27T03:00:00+00:00",
                "end_time": "2026-07-01T05:55:00+00:00",
            },
            {
                "start_time": "2026-07-01T06:00:00+00:00",
                "end_time": "2026-07-01T09:00:00+00:00",
            },
        ])
        self.assertTrue(bundle.kp_storm_eligible)
        self.assertEqual(
            bundle.status.metadata["kp_ap_missing_indices"], ["ap"]
        )
        self.assertEqual(
            set(bundle.products["product_kind"]),
            {
                "analysis", "rolling", "forecast_30", "forecast_90",
                "forecast_180", "forecast_360",
            },
        )

    def test_three_hour_schedule_has_37_five_minute_states(self):
        import data_loader

        times = data_loader.three_hour_aida_times("2026-06-21T20:00:00Z")

        self.assertEqual(len(times), 37)
        self.assertEqual(times[0], pd.Timestamp("2026-06-21T17:00:00Z"))
        self.assertEqual(times[-1], pd.Timestamp("2026-06-21T20:00:00Z"))
        self.assertTrue(all(
            right - left == pd.Timedelta(minutes=5)
            for left, right in zip(times, times[1:])
        ))

    def test_psd_reference_schedule_uses_previous_30_days_at_same_utc(self):
        import data_loader

        times = data_loader.psd_reference_times("2026-06-21T20:00:00Z")

        self.assertEqual(len(times), 30)
        self.assertEqual(times[0], pd.Timestamp("2026-05-22T20:00:00Z"))
        self.assertEqual(times[-1], pd.Timestamp("2026-06-20T20:00:00Z"))

    def test_product_frame_uses_requested_time_for_time_series(self):
        import data_loader

        upstream_time = pd.Timestamp("2000-01-01T00:00:00Z")
        requested = pd.Timestamp("2025-01-01T17:55:00Z")
        with patch.object(
            data_loader,
            "calculate_aida_grid",
            return_value=pd.DataFrame([{
                "time": upstream_time,
                "lat": 50.0,
                "lon": 1.0,
                "variable": "TEC",
                "value": 12.0,
            }]),
        ):
            frame = data_loader._calculate_product_frame(
                b"raw-state",
                GLOBAL_REGION,
                30,
                ["TEC"],
                product_kind="rolling",
                requested_time=requested,
            )

        self.assertEqual(frame.iloc[0]["time"], requested)
        self.assertEqual(frame.iloc[0]["requested_time"], requested)

    def test_forecast_product_frame_uses_valid_time_for_time_series(self):
        import data_loader

        upstream_time = pd.Timestamp("2000-01-01T00:00:00Z")
        analysis = pd.Timestamp("2025-01-01T17:55:00Z")
        with patch.object(
            data_loader,
            "calculate_aida_grid",
            return_value=pd.DataFrame([{
                "time": upstream_time,
                "lat": 50.0,
                "lon": 1.0,
                "variable": "TEC",
                "value": 12.0,
            }]),
        ):
            frame = data_loader._calculate_product_frame(
                b"raw-state",
                GLOBAL_REGION,
                30,
                ["TEC"],
                product_kind="forecast_180",
                requested_time=analysis,
                forecast_minutes=180,
            )

        self.assertEqual(frame.iloc[0]["time"], analysis + pd.Timedelta(minutes=180))
        self.assertIn("valid_time", frame.columns)
        self.assertEqual(
            frame.iloc[0]["valid_time"],
            analysis + pd.Timedelta(minutes=180),
        )
        self.assertIn("actual_output_time", frame.columns)
        self.assertEqual(frame.iloc[0]["actual_output_time"], upstream_time)
        self.assertEqual(frame.iloc[0]["requested_time"], analysis)

    def test_icao_products_use_one_download_per_time_and_official_forecasts(self):
        import data_loader

        client = FakeRawClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            bundle = data_loader.load_icao_products(
                analysis_time="2026-06-21T20:00:00Z",
                variables=["TEC"],
                region=GLOBAL_REGION,
                grid_step=30,
                include_psd_baseline=False,
            )

        self.assertEqual(len(client.download_requests), 37)
        self.assertEqual(len(set(client.download_requests)), 37)
        self.assertEqual(client.forecast_requests, [
            ("2026-06-21T20:00:00+00:00", "ultra", 30),
            ("2026-06-21T20:00:00+00:00", "ultra", 90),
            ("2026-06-21T20:00:00+00:00", "ultra", 180),
            ("2026-06-21T20:00:00+00:00", "ultra", 360),
        ])
        self.assertEqual(set(bundle.products["product_kind"]), {
            "analysis", "rolling", "forecast_30", "forecast_90",
            "forecast_180", "forecast_360",
        })
        self.assertEqual(bundle.status.metadata["analysis_downloads"], 37)
        self.assertEqual(bundle.status.metadata["rolling_analysis_downloads"], 37)
        self.assertEqual(bundle.status.metadata["forecast_downloads"], 4)
        self.assertEqual(bundle.status.metadata["primary_forecast_states"], 4)
        self.assertEqual(
            bundle.status.metadata["available_primary_forecast_periods"],
            [30, 90, 180, 360],
        )
        self.assertEqual(
            bundle.status.metadata.get("actual_analysis_output_time"),
            "2026-06-21T20:00:00+00:00",
        )
        self.assertEqual(
            [row["forecast_parameter"] for row in bundle.status.metadata["forecast_request_audit"]],
            [30, 90, 180, 360],
        )
        self.assertEqual(
            [row["display_role"] for row in bundle.status.metadata["forecast_request_audit"]],
            ["primary", "primary", "primary", "primary"],
        )
        self.assertEqual(
            [row["outcome"] for row in bundle.status.metadata["forecast_request_audit"]],
            ["available", "available", "available", "available"],
        )
        self.assertEqual(
            [row["valid_time"] for row in bundle.status.metadata["forecast_request_audit"]],
            [
                "2026-06-21T20:30:00+00:00",
                "2026-06-21T21:30:00+00:00",
                "2026-06-21T23:00:00+00:00",
                "2026-06-22T02:00:00+00:00",
            ],
        )

    def test_forecast_audit_distinguishes_not_published_and_authentication(self):
        import data_loader

        class PartiallyAvailableClient(FakeRawClient):
            def download_aida_forecast(self, requested_time, latency, period_minutes):
                self.forecast_requests = getattr(self, "forecast_requests", [])
                self.forecast_requests.append((requested_time, latency, period_minutes))
                if period_minutes in {30, 90}:
                    return True, f"forecast {period_minutes}", b"forecast-state"
                if period_minutes == 180:
                    return False, "SERENE AIDA forecast API returned status 404", None
                return False, "SERENE rejected the API token for AIDA forecast", None

        client = PartiallyAvailableClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            bundle = data_loader.load_icao_products(
                analysis_time="2026-06-21T20:00:00Z",
                variables=["TEC"],
                region=GLOBAL_REGION,
                grid_step=30,
                include_three_hour_window=False,
                include_psd_baseline=False,
            )

        audit = bundle.status.metadata["forecast_request_audit"]
        self.assertEqual(
            [row["outcome"] for row in audit],
            ["available", "available", "not_published", "authentication_failed"],
        )
        self.assertEqual(
            bundle.status.metadata["available_primary_forecast_periods"],
            [30, 90],
        )

    def test_icao_products_keep_observations_when_forecasts_fail(self):
        import data_loader

        class ForecastFailingClient(FakeRawClient):
            def download_aida_forecast(self, requested_time, latency, period_minutes):
                return False, f"forecast {period_minutes} unavailable", None

        client = ForecastFailingClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            bundle = data_loader.load_icao_products(
                analysis_time="2026-06-21T20:00:00Z",
                variables=["TEC"],
                region=GLOBAL_REGION,
                grid_step=30,
                include_three_hour_window=False,
                include_psd_baseline=False,
            )

        self.assertFalse(bundle.products.empty)
        self.assertEqual(set(bundle.products["product_kind"]), {"analysis", "rolling"})
        self.assertTrue(any("forecast 90 unavailable" in item for item in bundle.status.warnings))
        self.assertTrue(any("forecast 180 unavailable" in item for item in bundle.status.warnings))

    def test_forecast_decode_failure_stays_unavailable_in_summary_and_map(self):
        import data_loader
        from icao_risk import build_categorical_cells, build_icao_summary

        class DecodeFailingClient(FakeRawClient):
            def download_aida_forecast(self, requested_time, latency, period_minutes):
                self.forecast_requests = getattr(self, "forecast_requests", [])
                self.forecast_requests.append(
                    (requested_time, latency, period_minutes)
                )
                return True, f"forecast {period_minutes}", f"forecast-{period_minutes}".encode()

        def decode(payload, region, step, variables):
            if payload == b"forecast-90":
                raise data_loader.AidaGridError("controlled forecast decode failure")
            return _fake_calculation(payload, region, step, variables)

        client = DecodeFailingClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=decode),
        ):
            bundle = data_loader.load_icao_products(
                analysis_time="2026-06-21T20:00:00Z",
                variables=["TEC"],
                region=GLOBAL_REGION,
                grid_step=30,
                include_three_hour_window=False,
                include_psd_baseline=False,
            )

        summary = build_icao_summary(
            bundle.products,
            bundle.indices,
            eligible=bundle.kp_storm_eligible,
            kp_horizons=bundle.kp_horizons,
        )
        tec = summary.loc[summary["Indicator"] == "Vertical TEC"].iloc[0]

        self.assertNotIn("forecast_90", set(bundle.products["product_kind"]))
        self.assertIsNone(tec["+90 min forecast"])
        self.assertEqual(tec["+90 min status"], "UNAVAILABLE")
        self.assertEqual(tec["+90 min source"], "Unavailable")
        self.assertTrue(
            build_categorical_cells(
                bundle.products, "Vertical TEC", "+90 min"
            ).empty
        )
        audit = bundle.status.metadata["forecast_request_audit"]
        plus90 = next(row for row in audit if row["forecast_parameter"] == 90)
        self.assertEqual(plus90["outcome"], "decode_failed")

    def test_psd_reference_tolerates_two_missing_daily_states(self):
        import data_loader

        products = pd.DataFrame([
            {
                "product_kind": "baseline",
                "requested_time": pd.Timestamp("2026-05-01T12:00:00Z") + pd.Timedelta(days=index),
                "lat": 50.0,
                "lon": 1.0,
                "variable": "MUF3000F2",
                "value": 10.0,
            }
            for index in range(28)
        ] + [{
            "product_kind": "analysis",
            "requested_time": pd.Timestamp("2026-06-01T12:00:00Z"),
            "lat": 50.0,
            "lon": 1.0,
            "variable": "MUF3000F2",
            "value": 7.0,
        }])

        result = data_loader._attach_psd_reference(products)

        analysis = result[result["product_kind"] == "analysis"].iloc[0]
        self.assertEqual(float(analysis["reference_value"]), 10.0)
        self.assertAlmostEqual(float(analysis["psd_percent"]), 30.0)

    def test_psd_reference_tolerates_three_missing_daily_states(self):
        import data_loader

        products = pd.DataFrame([
            {
                "product_kind": "baseline",
                "requested_time": pd.Timestamp("2026-05-01T12:00:00Z") + pd.Timedelta(days=index),
                "lat": 50.0,
                "lon": 1.0,
                "variable": "MUF3000F2",
                "value": 10.0,
            }
            for index in range(27)
        ] + [{
            "product_kind": "analysis",
            "requested_time": pd.Timestamp("2026-06-01T12:00:00Z"),
            "lat": 50.0,
            "lon": 1.0,
            "variable": "MUF3000F2",
            "value": 7.0,
        }])

        result = data_loader._attach_psd_reference(products)

        analysis = result[result["product_kind"] == "analysis"].iloc[0]
        self.assertEqual(float(analysis["reference_value"]), 10.0)
        self.assertAlmostEqual(float(analysis["psd_percent"]), 30.0)

    def test_psd_reference_uses_30_day_median(self):
        import data_loader

        products = pd.DataFrame([
            {
                "product_kind": "baseline",
                "requested_time": pd.Timestamp("2026-05-01T12:00:00Z") + pd.Timedelta(days=index),
                "lat": 50.0,
                "lon": 1.0,
                "variable": "MUF3000F2",
                "value": 10.0,
            }
            for index in range(30)
        ] + [{
            "product_kind": "analysis",
            "requested_time": pd.Timestamp("2026-06-01T12:00:00Z"),
            "lat": 50.0,
            "lon": 1.0,
            "variable": "MUF3000F2",
            "value": 7.0,
        }])

        result = data_loader._attach_psd_reference(products)

        analysis = result[result["product_kind"] == "analysis"].iloc[0]
        self.assertEqual(float(analysis["reference_value"]), 10.0)
        self.assertAlmostEqual(float(analysis["psd_percent"]), 30.0)

    def test_loader_aggregates_baseline_before_building_product_table(self):
        import data_loader

        client = FakeRawClient()
        captured = {}

        def capture_reference(products, reference=None):
            captured["product_kinds"] = set(products["product_kind"])
            captured["reference"] = reference
            return products

        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
            patch.object(data_loader, "_attach_psd_reference", side_effect=capture_reference),
        ):
            data_loader.load_icao_products(
                analysis_time="2026-06-21T20:00:00Z",
                variables=["MUF3000F2"],
                region=GLOBAL_REGION,
                grid_step=30,
                include_three_hour_window=False,
                include_psd_baseline=True,
            )

        self.assertNotIn("baseline", captured["product_kinds"])
        self.assertIsInstance(captured["reference"], pd.DataFrame)
        self.assertEqual(float(captured["reference"].iloc[0]["reference_value"]), 30.0)

    def test_loader_summarises_missing_psd_baseline_files(self):
        import data_loader

        class PartlyMissingBaselineClient(FakeRawClient):
            def download_aida_raw_output(self, requested_time, latency):
                parsed = pd.Timestamp(requested_time)
                if parsed.date().isoformat() in {"2026-05-22", "2026-05-23"}:
                    return (
                        False,
                        f"SERENE AIDA raw-output API returned status 404 for "
                        f"product={latency}, file_type=raw, file_time={parsed:%Y-%m-%dT%H:%M:%S}. "
                        '"Requested file is not available for download."',
                        None,
                    )
                return super().download_aida_raw_output(requested_time, latency)

        client = PartlyMissingBaselineClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            bundle = data_loader.load_icao_products(
                analysis_time="2026-06-21T20:00:00Z",
                variables=["MUF3000F2"],
                region=GLOBAL_REGION,
                grid_step=30,
                include_three_hour_window=False,
                include_psd_baseline=True,
            )

        warnings = "\n".join(bundle.status.warnings)
        self.assertIn("PSD reference used 28/30 available SERENE AIDA states", warnings)
        self.assertNotIn("raw-output API returned status 404", warnings)
        self.assertEqual(bundle.status.metadata["baseline_download_failures"], 2)
        self.assertFalse(bundle.products["psd_percent"].dropna().empty)

    def test_loader_uses_psd_baseline_with_twenty_seven_reference_states(self):
        import data_loader

        class ThreeMissingBaselineClient(FakeRawClient):
            def download_aida_raw_output(self, requested_time, latency):
                parsed = pd.Timestamp(requested_time)
                if parsed.date().isoformat() in {
                    "2026-05-22", "2026-05-23", "2026-05-24",
                }:
                    return (
                        False,
                        f"SERENE AIDA raw-output API returned status 404 for "
                        f"product={latency}, file_type=raw, file_time={parsed:%Y-%m-%dT%H:%M:%S}. "
                        '"Requested file is not available for download."',
                        None,
                    )
                return super().download_aida_raw_output(requested_time, latency)

        client = ThreeMissingBaselineClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            bundle = data_loader.load_icao_products(
                analysis_time="2026-06-21T20:00:00Z",
                variables=["MUF3000F2"],
                region=GLOBAL_REGION,
                grid_step=30,
                include_three_hour_window=False,
                include_psd_baseline=True,
            )

        warnings = "\n".join(bundle.status.warnings)
        self.assertIn("PSD reference used 27/30 available SERENE AIDA states", warnings)
        self.assertEqual(bundle.status.metadata["baseline_download_failures"], 3)
        self.assertEqual(bundle.status.metadata["baseline_reference_states_used"], 27)
        self.assertFalse(bundle.products["psd_percent"].dropna().empty)

    def test_early_archive_window_skips_psd_and_summarises_missing_forecasts(self):
        import data_loader

        class MissingForecastClient(FakeRawClient):
            def download_aida_forecast(self, requested_time, latency, period_minutes):
                return (
                    False,
                    f"SERENE AIDA forecast API returned status 404 for "
                    f"product={latency}, file_type=raw, file_time=2024-10-11T02:55:00, "
                    f"forecast_period={period_minutes} min. "
                    '"Requested file is not available for download."',
                    None,
                )

        client = MissingForecastClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            bundle = data_loader.load_icao_products(
                analysis_time="2024-10-11T02:55:00Z",
                variables=["MUF3000F2"],
                region=GLOBAL_REGION,
                grid_step=30,
                include_three_hour_window=False,
                include_psd_baseline=True,
            )

        warnings = "\n".join(bundle.status.warnings)
        self.assertIn("PSD unavailable", warnings)
        self.assertIn("archive boundary", warnings)
        self.assertIn("Official AIDA +3h forecast unavailable", warnings)
        self.assertIn("Official AIDA +6h forecast unavailable", warnings)
        self.assertNotIn("only 13/30", warnings)
        self.assertNotIn("forecast API returned status 404", warnings)
        self.assertEqual(bundle.status.metadata["baseline_state_count"], 0)

    def test_api_failure_does_not_fall_back_to_local_file(self):
        import data_loader

        class FailingClient:
            def download_aida_raw_output(self, *_args):
                return False, "raw download failed", None

            def fetch_kp_ap_indices(self, **_kwargs):
                return False, "indices failed", pd.DataFrame()

        with patch.object(data_loader, "SereneClient", return_value=FailingClient()):
            frame, status = data_loader.load_data(source="api")

        self.assertTrue(frame.empty)
        self.assertEqual(status.source, "none")
        self.assertFalse(status.ok)
        self.assertNotIn("fallback", status.message.lower())

    def test_grid_density_does_not_change_raw_download_count(self):
        import data_loader

        coarse_client = FakeRawClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=coarse_client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            _coarse_frame, coarse = data_loader.load_data(
                start_time="2026-06-21T20:00:00Z",
                end_time="2026-06-21T21:00:00Z",
                variables=["TEC"],
                region=GLOBAL_REGION,
                grid_step=30,
            )

        dense_client = FakeRawClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=dense_client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            _dense_frame, dense = data_loader.load_data(
                start_time="2026-06-21T20:00:00Z",
                end_time="2026-06-21T21:00:00Z",
                variables=["TEC"],
                region=GLOBAL_REGION,
                grid_step=2,
            )

        self.assertEqual(len(coarse_client.download_requests), 2)
        self.assertEqual(len(dense_client.download_requests), 2)
        self.assertEqual(coarse.metadata["aida_dataset_downloads"], 2)
        self.assertEqual(dense.metadata["aida_dataset_downloads"], 2)
        self.assertEqual(coarse.metadata["local_map_points"], 91)
        self.assertEqual(dense.metadata["local_map_points"], 16471)

    def test_duplicate_time_and_latency_download_once(self):
        import data_loader

        client = FakeRawClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            _frame, status = data_loader.load_data(
                start_time="2026-06-21T20:00:00Z",
                end_time="2026-06-21T20:00:00Z",
            )

        self.assertEqual(len(client.download_requests), 1)
        self.assertEqual(status.metadata["aida_dataset_downloads"], 1)

    def test_times_rounded_to_same_five_minute_output_download_once(self):
        import data_loader

        client = FakeRawClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            _frame, status = data_loader.load_data(
                start_time="2026-06-21T20:00:01Z",
                end_time="2026-06-21T20:02:29Z",
            )

        self.assertEqual(len(client.download_requests), 1)
        self.assertEqual(
            client.download_requests[0],
            ("2026-06-21T20:00:00+00:00", "ultra"),
        )
        self.assertEqual(status.metadata["aida_dataset_downloads"], 1)

    def test_archive_time_uses_rapid_product_for_five_minute_raw_states(self):
        import data_loader

        client = FakeRawClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            data_loader.load_data(start_time="2024-05-10T21:00:00Z")

        self.assertEqual(client.download_requests[0][1], "rapid")

    def test_indices_only_result_does_not_claim_aida_success(self):
        import data_loader

        indices = pd.DataFrame([{
            "time": pd.Timestamp("2026-06-21T21:00:00Z"),
            "lat": None,
            "lon": None,
            "variable": "Kp",
            "value": 5.0,
            "model": "SERENE Indices",
            "source": "SERENE API Kp/ap",
        }])

        class IndicesOnlyClient:
            def download_aida_raw_output(self, *_args):
                return False, "raw download failed", None

            def fetch_kp_ap_indices(self, **_kwargs):
                return True, "indices ok", indices

        with patch.object(data_loader, "SereneClient", return_value=IndicesOnlyClient()):
            frame, status = data_loader.load_data(
                start_time="2026-06-21T20:00:00Z",
                end_time="2026-06-21T21:00:00Z",
            )

        self.assertEqual(set(frame["variable"]), {"Kp"})
        self.assertFalse(status.ok)
        self.assertEqual(status.source, "indices")
        self.assertIn("regional AIDA", status.message)

    def test_unavailable_kp_ap_propagates_source_freshness_without_psd_eligibility(self):
        import data_loader

        class StaleIndicesClient(FakeRawClient):
            kp_ap_source_latest_time = pd.Timestamp("2026-07-07T03:00:00Z")
            kp_ap_data_statuses = []

            def fetch_kp_ap_indices(self, **_kwargs):
                return False, "indices unavailable", pd.DataFrame()

        client = StaleIndicesClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            bundle = data_loader.load_icao_products(
                analysis_time="2026-08-10T08:50:00Z",
                variables=["MUF3000F2"],
                region=GLOBAL_REGION,
                grid_step=30,
                include_three_hour_window=False,
                include_psd_baseline=False,
            )

        self.assertEqual(
            bundle.status.metadata.get("kp_ap_source_latest_time"),
            "2026-07-07T03:00:00+00:00",
        )
        self.assertIsNone(bundle.kp_storm_eligible)
        self.assertIn(
            "Complete 96-hour GFZ Kp history is unavailable",
            "\n".join(bundle.status.warnings),
        )

    def test_gfz_metadata_and_96h_gate_preserve_aida_product_types(self):
        import data_loader

        analysis = pd.Timestamp("2026-08-12T09:00:00Z")
        times = pd.date_range(end=analysis, periods=32, freq="3h", tz="UTC")
        indices = pd.DataFrame([{
            "time": timestamp,
            "lat": None,
            "lon": None,
            "alt": None,
            "variable": "Kp",
            "value": 6.0 if timestamp == times[5] else 2.0,
            "model": "GFZ Geomagnetic Indices",
            "source": "GFZ Kp/ap nowcast",
            "data_status": "preliminary",
        } for timestamp in times])

        class CurrentGfzClient(FakeRawClient):
            kp_ap_source_latest_time = analysis
            kp_ap_data_statuses = ["preliminary"]

            def fetch_kp_ap_indices(self, **_kwargs):
                return True, "Loaded GFZ Kp/ap rows", indices

        client = CurrentGfzClient()
        with (
            patch.object(data_loader, "SereneClient", return_value=client),
            patch.object(data_loader, "calculate_aida_grid", side_effect=_fake_calculation),
        ):
            bundle = data_loader.load_icao_products(
                analysis_time=analysis.isoformat(),
                variables=["TEC"],
                region=GLOBAL_REGION,
                grid_step=30,
                include_three_hour_window=False,
                include_psd_baseline=False,
            )

        self.assertTrue(bundle.kp_storm_eligible)
        self.assertEqual(
            bundle.status.metadata["kp_ap_source"],
            "GFZ Helmholtz Centre for Geosciences",
        )
        self.assertEqual(
            bundle.status.metadata["kp_ap_data_statuses"], ["preliminary"]
        )
        self.assertEqual(
            set(bundle.products["product_kind"]),
            {
                "analysis", "rolling", "forecast_30", "forecast_90",
                "forecast_180", "forecast_360",
            },
        )

    def test_kp_ap_caption_reports_gfz_time_and_loaded_data_status(self):
        import app
        from data_loader import LoadStatus

        formatter = getattr(app, "_kp_ap_source_freshness_caption", None)
        self.assertIsNotNone(formatter)

        unavailable = LoadStatus(metadata={
            "kp_ap_index_status": "unavailable",
            "kp_ap_source_latest_time": "2026-07-07T03:00:00+00:00",
            "kp_ap_source": "GFZ Helmholtz Centre for Geosciences",
            "kp_ap_data_statuses": [],
        })
        loaded = LoadStatus(metadata={
            "kp_ap_index_status": "loaded",
            "kp_ap_source_latest_time": "2026-08-12T09:00:00+00:00",
            "kp_ap_source": "GFZ Helmholtz Centre for Geosciences",
            "kp_ap_data_statuses": ["preliminary"],
        })
        partial = LoadStatus(metadata={
            "kp_ap_index_status": "loaded",
            "kp_ap_source_latest_time": "2026-08-12T09:00:00+00:00",
            "kp_ap_source": "GFZ Helmholtz Centre for Geosciences",
            "kp_ap_data_statuses": ["preliminary"],
            "kp_ap_missing_indices": ["ap"],
        })
        malformed = LoadStatus(metadata={
            "kp_ap_index_status": "unavailable",
            "kp_ap_source_latest_time": "<script>private-token</script>",
        })

        self.assertEqual(
            formatter(unavailable),
            "GFZ Kp/ap unavailable — latest source timestamp: "
            "2026-07-07 03:00 UTC",
        )
        self.assertEqual(
            formatter(loaded),
            "GFZ Kp/ap — latest source timestamp: 2026-08-12 09:00 UTC; "
            "loaded status: preliminary",
        )
        self.assertEqual(
            formatter(partial),
            "GFZ Kp loaded; ap unavailable — latest source timestamp: "
            "2026-08-12 09:00 UTC; loaded status: preliminary",
        )
        self.assertIsNone(formatter(malformed))
        for parseable_but_invalid in ("now", "today", 0):
            with self.subTest(value=parseable_but_invalid):
                invalid = LoadStatus(metadata={
                    "kp_ap_index_status": "unavailable",
                    "kp_ap_source_latest_time": parseable_but_invalid,
                })
                self.assertIsNone(formatter(invalid))


if __name__ == "__main__":
    unittest.main()
