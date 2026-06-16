"""
TrendForecaster: extracts numeric time-series from DataPoints and forecasts
market trends using Prophet (primary) or ARIMA (fallback).
Computes CAGR and 80% confidence intervals.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    metric: str
    unit: str
    historical_years: list[int]
    historical_values: list[float]
    forecast_years: list[int]
    forecast_values: list[float]
    ci_80_upper: list[float]
    ci_80_lower: list[float]
    cagr_historical: float
    cagr_projected: float
    method: str
    confidence: float
    notes: str = ""


NUMBER_PATTERNS = [
    r"\$?\s*(\d[\d,\.]*)\s*(?:trillion|T\b)",
    r"\$?\s*(\d[\d,\.]*)\s*(?:billion|B\b)",
    r"\$?\s*(\d[\d,\.]*)\s*(?:million|M\b)",
    r"(?:growth|grew|increase)\s+(?:of|by)\s+(\d+(?:\.\d+)?)\s*%",
    r"(\d+(?:\.\d+)?)\s*%\s*(?:CAGR|compound)",
    r"(\d+(?:\.\d+)?)\s*%\s*(?:growth|increase)",
    r"market\s+(?:size|value)\s+(?:of|at|reached?)\s+\$?\s*(\d[\d,\.]*)",
]

MULTIPLIERS = {
    "trillion": 1_000_000,
    "t": 1_000_000,
    "billion": 1_000,
    "b": 1_000,
    "million": 1,
    "m": 1,
}

METRIC_EXTRACTION_PROMPT = """Extract all numeric market size, growth rate, or financial metric values from this text.
Return JSON: {{"metrics": [{{"year": 2023, "value": 45.2, "unit": "billion USD", "description": "market size"}}]}}
Text: {text}
Return only the JSON, no other text."""


class TrendForecaster:
    def __init__(self, llm_client=None):
        self._llm = llm_client

    async def forecast(self, data_points: list) -> Optional[ForecastResult]:
        series = await self._extract_time_series(data_points)
        if not series or len(series) < 2:
            logger.info("Insufficient time-series data for forecasting (%d points)", len(series))
            return self._qualitative_trend(data_points)

        years = sorted(series.keys())
        values = [series[y] for y in years]
        unit = self._infer_unit(data_points)
        metric = self._infer_metric_name(data_points)

        forecast_result = self._prophet_forecast(years, values, metric, unit)
        if forecast_result is None:
            forecast_result = self._arima_forecast(years, values, metric, unit)
        if forecast_result is None:
            forecast_result = self._linear_forecast(years, values, metric, unit)

        return forecast_result

    # ── Time-series extraction ────────────────────────────────────────────────

    async def _extract_time_series(self, data_points: list) -> dict[int, float]:
        series: dict[int, float] = {}

        for dp in data_points:
            if dp.numeric_value is not None and dp.date:
                try:
                    year = int(dp.date[:4])
                    if 1990 <= year <= datetime.utcnow().year + 1:
                        val = dp.numeric_value
                        if dp.numeric_unit and "%" not in dp.numeric_unit:
                            if year not in series or val > series[year]:
                                series[year] = val
                except (ValueError, TypeError):
                    pass

        for dp in data_points:
            text = f"{dp.title} {dp.snippet}"
            extracted = self._parse_numbers_from_text(text, dp.date)
            for year, value in extracted.items():
                if year not in series:
                    series[year] = value

        if not series and self._llm:
            snippets = " ".join(dp.snippet for dp in data_points[:5])
            if snippets.strip():
                try:
                    result = await self._llm.complete(
                        METRIC_EXTRACTION_PROMPT.format(text=snippets[:1500]),
                        max_tokens=400,
                        task="metric_extraction",
                    )
                    import json, re as re2
                    raw = re2.search(r"\{[\s\S]+\}", result.text)
                    if raw:
                        data = json.loads(raw.group())
                        for m in data.get("metrics", []):
                            year = m.get("year")
                            value = m.get("value")
                            if year and value and 1990 <= year <= datetime.utcnow().year + 5:
                                series[int(year)] = float(value)
                except Exception as e:
                    logger.debug("LLM metric extraction failed: %s", e)

        return series

    def _parse_numbers_from_text(self, text: str, date_str: str = "") -> dict[int, float]:
        series = {}
        year = None
        if date_str and len(date_str) >= 4:
            try:
                year = int(date_str[:4])
            except ValueError:
                pass

        year_matches = re.findall(r"\b(20\d{2}|19\d{2})\b", text)

        for pattern in NUMBER_PATTERNS[:3]:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                raw_num = m.group(1).replace(",", "")
                try:
                    value = float(raw_num)
                except ValueError:
                    continue
                suffix_match = re.search(r"(trillion|billion|million|T\b|B\b|M\b)", text[m.end(): m.end() + 20], re.IGNORECASE)
                if suffix_match:
                    mult = MULTIPLIERS.get(suffix_match.group(1).lower(), 1)
                    value *= mult

                if value > 0 and year:
                    near_years = [
                        int(y) for y in year_matches
                        if abs(text.find(y) - m.start()) < 100
                    ] if year_matches else [year]
                    target_year = near_years[0] if near_years else year
                    if 1990 <= target_year <= datetime.utcnow().year + 2:
                        series[target_year] = value

        return series

    # ── Forecasting methods ───────────────────────────────────────────────────

    def _prophet_forecast(self, years: list[int], values: list[float], metric: str, unit: str) -> Optional[ForecastResult]:
        try:
            import pandas as pd
            from prophet import Prophet

            df = pd.DataFrame({
                "ds": pd.to_datetime([f"{y}-06-01" for y in years]),
                "y": values,
            })
            if len(df) < 2:
                return None

            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                            interval_width=0.80)
            model.fit(df)

            last_year = max(years)
            future_years = list(range(last_year + 1, last_year + 4))
            future = pd.DataFrame({"ds": pd.to_datetime([f"{y}-06-01" for y in future_years])})
            forecast = model.predict(future)

            forecast_values = forecast["yhat"].tolist()
            ci_upper = forecast["yhat_upper"].tolist()
            ci_lower = forecast["yhat_lower"].tolist()

            cagr_hist = self._compute_cagr(values[0], values[-1], len(values) - 1)
            cagr_proj = self._compute_cagr(values[-1], forecast_values[-1], len(future_years))

            return ForecastResult(
                metric=metric,
                unit=unit,
                historical_years=years,
                historical_values=values,
                forecast_years=future_years,
                forecast_values=[round(v, 2) for v in forecast_values],
                ci_80_upper=[round(v, 2) for v in ci_upper],
                ci_80_lower=[round(v, 2) for v in ci_lower],
                cagr_historical=round(cagr_hist, 4),
                cagr_projected=round(cagr_proj, 4),
                method="Prophet",
                confidence=0.80,
            )
        except ImportError:
            logger.info("Prophet not installed — trying ARIMA")
            return None
        except Exception as e:
            logger.warning("Prophet forecast failed: %s", e)
            return None

    def _arima_forecast(self, years: list[int], values: list[float], metric: str, unit: str) -> Optional[ForecastResult]:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            import numpy as np

            model = ARIMA(values, order=(min(2, len(values) - 1), 1, min(2, len(values) - 1)))
            fit = model.fit()
            steps = 3
            forecast_obj = fit.get_forecast(steps=steps)
            forecast_values = forecast_obj.predicted_mean.tolist()
            ci = forecast_obj.conf_int(alpha=0.20)
            ci_upper = ci.iloc[:, 1].tolist()
            ci_lower = ci.iloc[:, 0].tolist()

            last_year = max(years)
            future_years = list(range(last_year + 1, last_year + steps + 1))
            cagr_hist = self._compute_cagr(values[0], values[-1], len(values) - 1)
            cagr_proj = self._compute_cagr(values[-1], forecast_values[-1], steps)

            return ForecastResult(
                metric=metric,
                unit=unit,
                historical_years=years,
                historical_values=values,
                forecast_years=future_years,
                forecast_values=[round(v, 2) for v in forecast_values],
                ci_80_upper=[round(v, 2) for v in ci_upper],
                ci_80_lower=[round(v, 2) for v in ci_lower],
                cagr_historical=round(cagr_hist, 4),
                cagr_projected=round(cagr_proj, 4),
                method="ARIMA(2,1,2)",
                confidence=0.75,
            )
        except ImportError:
            logger.info("statsmodels not installed — using linear fallback")
            return None
        except Exception as e:
            logger.warning("ARIMA forecast failed: %s", e)
            return None

    def _linear_forecast(self, years: list[int], values: list[float], metric: str, unit: str) -> ForecastResult:
        n = len(years)
        if n < 2:
            avg = values[0] if values else 0
            last_year = years[0] if years else datetime.utcnow().year
            future_years = [last_year + 1, last_year + 2, last_year + 3]
            forecast_values = [avg * 1.05, avg * 1.10, avg * 1.16]
            return ForecastResult(
                metric=metric, unit=unit,
                historical_years=years, historical_values=values,
                forecast_years=future_years, forecast_values=[round(v, 2) for v in forecast_values],
                ci_80_upper=[round(v * 1.2, 2) for v in forecast_values],
                ci_80_lower=[round(v * 0.8, 2) for v in forecast_values],
                cagr_historical=0.05, cagr_projected=0.05,
                method="Linear (insufficient data)", confidence=0.50,
                notes="Insufficient historical data for statistical forecast",
            )

        x_mean = sum(years) / n
        y_mean = sum(values) / n
        numerator = sum((years[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((years[i] - x_mean) ** 2 for i in range(n)) + 1e-9
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        last_year = max(years)
        future_years = [last_year + 1, last_year + 2, last_year + 3]
        forecast_values = [slope * y + intercept for y in future_years]
        std_err = (sum((values[i] - (slope * years[i] + intercept)) ** 2 for i in range(n)) / max(n - 2, 1)) ** 0.5 * 1.28

        cagr_hist = self._compute_cagr(values[0], values[-1], n - 1)
        cagr_proj = self._compute_cagr(values[-1], forecast_values[-1], 3)

        return ForecastResult(
            metric=metric, unit=unit,
            historical_years=years, historical_values=values,
            forecast_years=future_years,
            forecast_values=[round(max(0, v), 2) for v in forecast_values],
            ci_80_upper=[round(max(0, v + std_err), 2) for v in forecast_values],
            ci_80_lower=[round(max(0, v - std_err), 2) for v in forecast_values],
            cagr_historical=round(cagr_hist, 4),
            cagr_projected=round(cagr_proj, 4),
            method="Linear regression",
            confidence=0.60,
        )

    def _qualitative_trend(self, data_points: list) -> Optional[ForecastResult]:
        text = " ".join(dp.snippet for dp in data_points[:5]).lower()
        growth_signals = sum(1 for w in ["grow", "increas", "expan", "boom", "surge"] if w in text)
        decline_signals = sum(1 for w in ["declin", "shrink", "slow", "contract", "fall"] if w in text)
        current_year = datetime.utcnow().year
        trend_cagr = 0.08 if growth_signals > decline_signals else (-0.03 if decline_signals > growth_signals else 0.04)
        return ForecastResult(
            metric="Qualitative market trend",
            unit="index",
            historical_years=[current_year - 2, current_year - 1, current_year],
            historical_values=[100.0, 100 * (1 + trend_cagr), 100 * (1 + trend_cagr) ** 2],
            forecast_years=[current_year + 1, current_year + 2, current_year + 3],
            forecast_values=[100 * (1 + trend_cagr) ** 3, 100 * (1 + trend_cagr) ** 4, 100 * (1 + trend_cagr) ** 5],
            ci_80_upper=[v * 1.15 for v in [100 * (1 + trend_cagr) ** i for i in range(3, 6)]],
            ci_80_lower=[v * 0.85 for v in [100 * (1 + trend_cagr) ** i for i in range(3, 6)]],
            cagr_historical=trend_cagr,
            cagr_projected=trend_cagr,
            method="Qualitative sentiment analysis",
            confidence=0.45,
            notes="Insufficient numeric data — trend estimated from text sentiment",
        )

    @staticmethod
    def _compute_cagr(start: float, end: float, years: int) -> float:
        if years <= 0 or start <= 0 or end <= 0:
            return 0.0
        return (end / start) ** (1.0 / years) - 1.0

    @staticmethod
    def _infer_unit(data_points: list) -> str:
        for dp in data_points:
            text = f"{dp.title} {dp.snippet}".lower()
            if "billion usd" in text or "billion $" in text or "usd billion" in text:
                return "billion USD"
            if "trillion usd" in text or "usd trillion" in text:
                return "trillion USD"
            if "million usd" in text:
                return "million USD"
        return "USD (units unspecified)"

    @staticmethod
    def _infer_metric_name(data_points: list) -> str:
        for dp in data_points:
            text = f"{dp.title} {dp.snippet}".lower()
            if "market size" in text:
                return "Market Size"
            if "market value" in text:
                return "Market Value"
            if "revenue" in text:
                return "Total Revenue"
            if "gdp" in text:
                return "GDP"
        return "Market Indicator"
