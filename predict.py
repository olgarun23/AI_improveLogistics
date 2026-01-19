from __future__ import annotations

import os
import pickle
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np

HORIZON_HOURS = 72
HOURS_PER_WEEK = 24 * 7
LOOK_BACK = 24

FEATURE_COLUMNS = (
    "temperature",
    "windspeed",
    "cloud_coverage",
    "dewpoint",
    "rain_accumulated",
    "f",
    "fg",
    "fsdev",
    "d",
    "dsdev",
    "t",
    "rh",
    "td",
    "p",
    "r",
)

MODEL_PATH = os.getenv("MODEL_PATH", "best_lstm_model.keras")
FEATURE_SCALER_PATH = os.getenv("FEATURE_SCALER_PATH", "scaler_features.pkl")
TARGET_SCALER_PATH = os.getenv("TARGET_SCALER_PATH", "scaler_targets.pkl")


class _ArrayMinMaxScaler:
    def __init__(self, min_: np.ndarray, scale_: np.ndarray) -> None:
        self.min_ = np.asarray(min_, dtype=float)
        self.scale_ = np.asarray(scale_, dtype=float)
        if self.min_.ndim != 1 or self.scale_.ndim != 1:
            raise ValueError("Scaler arrays must be 1D")
        if self.min_.shape != self.scale_.shape:
            raise ValueError("Scaler arrays must have the same shape")
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)

    @property
    def n_features_in_(self) -> int:
        return self.min_.shape[0]

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.scale_ + self.min_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min_) / self.scale_


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


def _forward_fill_then_mean(data: np.ndarray) -> np.ndarray:
    filled = data.copy()
    rows, cols = filled.shape
    for col in range(cols):
        col_values = filled[:, col]
        last = np.nan
        for idx in range(rows):
            if np.isnan(col_values[idx]):
                if not np.isnan(last):
                    col_values[idx] = last
            else:
                last = col_values[idx]
        if np.isnan(col_values).any():
            mean = np.nanmean(col_values)
            if np.isnan(mean):
                mean = 0.0
            col_values[np.isnan(col_values)] = mean
        filled[:, col] = col_values
    return filled


def _load_scaler(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found: {path}")

    if path.endswith(".npz"):
        data = np.load(path)
        if "min_" in data and "scale_" in data:
            return _ArrayMinMaxScaler(data["min_"], data["scale_"])
        if "data_min_" in data and "data_max_" in data:
            data_min = np.asarray(data["data_min_"], dtype=float)
            data_max = np.asarray(data["data_max_"], dtype=float)
            denom = data_max - data_min
            scale = np.divide(1.0, denom, out=np.ones_like(denom), where=denom != 0)
            min_ = -data_min * scale
            return _ArrayMinMaxScaler(min_, scale)
        raise ValueError(f"Unsupported scaler npz format: {path}")

    with open(path, "rb") as handle:
        scaler = pickle.load(handle)
    if not hasattr(scaler, "transform"):
        raise ValueError(f"Scaler in {path} missing transform()")
    return scaler


def _load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to load the LSTM model") from exc

    return tf.keras.models.load_model(path)


def _expected_feature_count(scaler) -> int:
    if hasattr(scaler, "n_features_in_"):
        return int(scaler.n_features_in_)
    return len(FEATURE_COLUMNS)


def _build_feature_sequences(
    weather_history: np.ndarray,
    weather_forecast: np.ndarray,
    feature_scaler,
) -> np.ndarray:
    history = np.asarray(weather_history, dtype=float)
    forecast = np.asarray(weather_forecast, dtype=float)

    if history.ndim == 1:
        history = history.reshape(-1, 1)
    if forecast.ndim == 1:
        forecast = forecast.reshape(-1, 1)

    if history.shape[1] != forecast.shape[1]:
        raise ValueError("weather_history and weather_forecast feature counts differ")

    expected_features = _expected_feature_count(feature_scaler)
    if history.shape[1] != expected_features:
        raise ValueError(
            "weather feature count mismatch: "
            f"expected {expected_features}, got {history.shape[1]}"
        )

    if history.shape[0] < LOOK_BACK:
        raise ValueError(
            f"weather_history must contain at least {LOOK_BACK} rows"
        )
    if forecast.shape[0] < HORIZON_HOURS:
        raise ValueError(
            f"weather_forecast must contain at least {HORIZON_HOURS} rows"
        )

    combined = np.concatenate([history, forecast[:HORIZON_HOURS]], axis=0)
    combined = _forward_fill_then_mean(combined)

    combined_scaled = feature_scaler.transform(combined)

    start = history.shape[0] - LOOK_BACK
    sequences = np.empty((HORIZON_HOURS, LOOK_BACK, combined_scaled.shape[1]))
    for idx in range(HORIZON_HOURS):
        sequences[idx] = combined_scaled[start + idx : start + idx + LOOK_BACK]

    return sequences


def _predict_with_model(
    weather_history: np.ndarray, weather_forecast: np.ndarray
) -> np.ndarray:
    model = _load_model(MODEL_PATH)
    feature_scaler = _load_scaler(FEATURE_SCALER_PATH)
    target_scaler = _load_scaler(TARGET_SCALER_PATH)

    sequences = _build_feature_sequences(
        weather_history, weather_forecast, feature_scaler
    )
    predictions_scaled = model.predict(sequences, verbose=0)

    if predictions_scaled.ndim != 2:
        predictions_scaled = predictions_scaled.reshape(predictions_scaled.shape[0], -1)

    if hasattr(target_scaler, "inverse_transform"):
        return target_scaler.inverse_transform(predictions_scaled)

    raise ValueError("target scaler missing inverse_transform()")


def _merge_predictions(
    model_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    n_sensors: int,
) -> np.ndarray:
    if model_predictions.shape[0] != HORIZON_HOURS:
        raise ValueError(
            f"Model predictions must have {HORIZON_HOURS} rows, "
            f"got {model_predictions.shape[0]}"
        )

    if model_predictions.shape[1] == n_sensors:
        return model_predictions

    if model_predictions.shape[1] > n_sensors:
        return model_predictions[:, :n_sensors]

    merged = baseline_predictions.copy()
    merged[:, : model_predictions.shape[1]] = model_predictions
    return merged


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
            Weather feature order must match training:
            temperature, windspeed, cloud_coverage, dewpoint, rain_accumulated,
            f, fg, fsdev, d, dsdev, t, rh, td, p, r.

    Returns:
        (72, 45) array of predictions.
    """
    history = np.asarray(sensor_history, dtype=float)
    if history.ndim != 2:
        raise ValueError("sensor_history must be a 2D array")
    if history.shape[0] == 0:
        raise ValueError("sensor_history must not be empty")

    history_len, n_sensors = history.shape
    start_time = _parse_timestamp(timestamp)

    history_start = start_time - timedelta(hours=history_len)
    history_hours = _hour_of_week_series(history_start, history_len)
    forecast_hours = _hour_of_week_series(start_time, HORIZON_HOURS)

    baseline, overall = _compute_hourly_baseline(history, history_hours)
    baseline_forecast = baseline[forecast_hours]
    predictions = baseline_forecast.copy()

    if weather_history is not None and weather_forecast is not None:
        model_predictions = _predict_with_model(weather_history, weather_forecast)
        predictions = _merge_predictions(
            model_predictions, baseline_forecast, n_sensors
        )

    predictions = np.where(np.isfinite(predictions), predictions, overall)
    predictions = np.maximum(predictions, 0.0)
    if predictions.shape != (HORIZON_HOURS, n_sensors):
        raise ValueError(
            f"Unexpected prediction shape {predictions.shape}, "
            f"expected {(HORIZON_HOURS, n_sensors)}"
        )
    return predictions
