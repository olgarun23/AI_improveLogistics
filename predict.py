from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np

HORIZON_HOURS = 72
HOURS_PER_WEEK = 24 * 7


def _parse_timestamp(timestamp: str) -> datetime:
    if not isinstance(timestamp, str):
        raise ValueError("timestamp must be an ISO-format string")

    normalized = timestamp.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unrecognized timestamp format: {timestamp}")


def _hour_of_week_series(start: datetime, count: int) -> np.ndarray:
    values = np.empty(count, dtype=int)
    current = start
    for idx in range(count):
        values[idx] = current.weekday() * 24 + current.hour
        current += timedelta(hours=1)
    return values


def _compute_hourly_baseline(
    sensor_history: np.ndarray, history_hours: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n_sensors = sensor_history.shape[1]
    baseline = np.full((HOURS_PER_WEEK, n_sensors), np.nan, dtype=float)

    for hour in range(HOURS_PER_WEEK):
        mask = history_hours == hour
        if np.any(mask):
            baseline[hour] = np.nanmean(sensor_history[mask], axis=0)

    overall = np.nanmean(sensor_history, axis=0)
    overall = np.where(np.isnan(overall), 0.0, overall)
    baseline = np.where(np.isnan(baseline), overall, baseline)
    return baseline, overall


def _fill_sensor_history(
    sensor_history: np.ndarray,
    baseline_history: np.ndarray,
    overall: np.ndarray,
) -> np.ndarray:
    if not np.isnan(sensor_history).any():
        return sensor_history

    filled = sensor_history.copy()
    fallback = np.where(np.isnan(baseline_history), overall, baseline_history)
    mask = np.isnan(filled)
    filled[mask] = fallback[mask]
    return filled


def _prepare_weather_arrays(
    weather_history: Optional[np.ndarray],
    weather_forecast: Optional[np.ndarray],
    history_len: int,
    horizon: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if weather_history is None or weather_forecast is None:
        return None

    history = np.asarray(weather_history, dtype=float)
    forecast = np.asarray(weather_forecast, dtype=float)

    if history.ndim == 1:
        history = history.reshape(-1, 1)
    if forecast.ndim == 1:
        forecast = forecast.reshape(-1, 1)

    if history.shape[0] < history_len or forecast.shape[0] < horizon:
        return None
    if history.shape[1] != forecast.shape[1]:
        return None

    history = history[-history_len:]
    forecast = forecast[:horizon]
    return history, forecast


def _fill_nan_with_mean(
    data: np.ndarray, means: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if means is None:
        means = np.nanmean(data, axis=0)
    means = np.where(np.isnan(means), 0.0, means)

    if np.isnan(data).any():
        filled = data.copy()
        row_idx, col_idx = np.where(np.isnan(filled))
        filled[row_idx, col_idx] = means[col_idx]
        return filled, means

    return data, means


def _ridge_residual_predict(
    weather_history: np.ndarray,
    weather_forecast: np.ndarray,
    residuals: np.ndarray,
    alpha: float = 1e-3,
) -> Optional[np.ndarray]:
    if weather_history.size == 0 or weather_forecast.size == 0:
        return None

    history, means = _fill_nan_with_mean(weather_history)
    forecast, _ = _fill_nan_with_mean(weather_forecast, means)

    stds = np.nanstd(history, axis=0)
    stds = np.where(stds == 0, 1.0, stds)

    history_scaled = (history - means) / stds
    forecast_scaled = (forecast - means) / stds

    ones_history = np.ones((history_scaled.shape[0], 1))
    ones_forecast = np.ones((forecast_scaled.shape[0], 1))
    history_design = np.concatenate([history_scaled, ones_history], axis=1)
    forecast_design = np.concatenate([forecast_scaled, ones_forecast], axis=1)

    reg = np.eye(history_design.shape[1])
    reg[-1, -1] = 0.0
    lhs = history_design.T @ history_design + alpha * reg
    rhs = history_design.T @ residuals

    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients, _, _, _ = np.linalg.lstsq(history_design, residuals, rcond=None)

    residual_pred = forecast_design @ coefficients
    if not np.all(np.isfinite(residual_pred)):
        return None
    return residual_pred


def predict(
    sensor_history: np.ndarray,
    timestamp: str,
    weather_forecast: Optional[np.ndarray] = None,
    weather_history: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Predict hot water demand for all sensors, 72 hours ahead.

    Args:
        sensor_history: (672, 45) array of past sensor readings.
        timestamp: ISO format datetime string for the first forecast hour.
        weather_forecast: (72, n) array of weather forecasts (optional).
        weather_history: (672, n) array of weather observations (optional).

    Returns:
        (72, 45) array of predictions.
    """
    history = np.asarray(sensor_history, dtype=float)
    if history.ndim != 2:
        raise ValueError("sensor_history must be a 2D array")

    history_len, n_sensors = history.shape
    start_time = _parse_timestamp(timestamp)

    history_start = start_time - timedelta(hours=history_len)
    history_hours = _hour_of_week_series(history_start, history_len)
    forecast_hours = _hour_of_week_series(start_time, HORIZON_HOURS)

    baseline, overall = _compute_hourly_baseline(history, history_hours)
    baseline_history = baseline[history_hours]
    baseline_forecast = baseline[forecast_hours]

    history_filled = _fill_sensor_history(history, baseline_history, overall)
    predictions = baseline_forecast.copy()

    weather_arrays = _prepare_weather_arrays(
        weather_history, weather_forecast, history_len, HORIZON_HOURS
    )
    if weather_arrays is not None:
        history_weather, forecast_weather = weather_arrays
        residuals = history_filled - baseline_history
        residual_pred = _ridge_residual_predict(
            history_weather, forecast_weather, residuals
        )
        if residual_pred is not None:
            predictions = baseline_forecast + residual_pred

    predictions = np.where(np.isfinite(predictions), predictions, overall)
    predictions = np.maximum(predictions, 0.0)
    if predictions.shape != (HORIZON_HOURS, n_sensors):
        raise ValueError(
            f"Unexpected prediction shape {predictions.shape}, "
            f"expected {(HORIZON_HOURS, n_sensors)}"
        )
    return predictions
