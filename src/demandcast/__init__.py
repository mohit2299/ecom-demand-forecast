"""demandcast: a leakage-safe e-commerce demand forecast (traffic x conversion x order value) with paid-media effects."""
from .evaluate import back_forecast, bias, monthly_view, rolling_backtest, seasonal_naive, summarize_backtest, wape
from .features import adstock, detect_level_shifts, event_calendar, fourier, thanksgiving
from .model import DemandModel, ModelConfig, spend_response
from .synth import Truth, make_dataset

__all__ = ["DemandModel", "ModelConfig", "spend_response", "make_dataset", "Truth", "rolling_backtest",
           "summarize_backtest", "back_forecast", "monthly_view", "seasonal_naive", "wape", "bias", "adstock",
           "detect_level_shifts", "event_calendar", "fourier", "thanksgiving"]
