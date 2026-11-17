"""Honest evaluation: rolling-origin backtest, benchmarks, a final holdout, and a back-forecast from a past date.

The rule throughout: a forecast made on day D is trained on days before D only. That includes everything that is
learned from data (spend effects, the spend reference level, tracking breaks), not just the regression fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import DemandModel, ModelConfig

INPUT_COLS = ["event", "promo_share", "discount_depth", "spend_search", "spend_social"]


def wape(actual, forecast) -> float:
    """Weighted absolute percentage error: sum |F - A| / sum A."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.abs(f - a).sum() / a.sum())


def bias(actual, forecast) -> float:
    """sum F / sum A - 1. Positive = forecast too high."""
    return float(np.asarray(forecast, float).sum() / np.asarray(actual, float).sum() - 1)


def seasonal_naive(train: pd.DataFrame, future: pd.DataFrame, col: str = "demand") -> np.ndarray:
    """Benchmark: same weekday last year (364 days back) x the trailing 28-day year-on-year ratio, per channel.
    If last year's day is itself in the future (horizon > 1 year), it goes back two years."""
    out = np.full(len(future), np.nan)
    for ch, f in future.groupby("channel"):
        h = train[train["channel"] == ch].set_index("date")[col]
        last = h.index.max()
        recent = h.loc[last - pd.Timedelta(days=27):last].sum()
        ly = h.loc[last - pd.Timedelta(days=364 + 27):last - pd.Timedelta(days=364)].sum()
        ratio = recent / ly if ly > 0 else 1.0
        back = f["date"] - pd.Timedelta(days=364)
        back = back.where(back <= last, f["date"] - pd.Timedelta(days=728))
        out[f.index.values] = h.reindex(back.values).values * ratio
    return out


def rolling_backtest(daily: pd.DataFrame, origins, horizon: int = 90, config: ModelConfig | None = None) -> pd.DataFrame:
    """Refits the model at every origin on earlier days only and forecasts the next `horizon` days.

    The forecast days get their *actual* promotions and spend: this scores the model given its inputs
    (a best case). back_forecast() scores it with only what was known at the time.
    """
    rows = []
    for o in pd.to_datetime(origins):
        train = daily[daily["date"] < o]
        test = daily[(daily["date"] >= o) & (daily["date"] < o + pd.Timedelta(days=horizon))].reset_index(drop=True)
        if test.empty:
            continue
        m = DemandModel(config).fit(train)
        p = m.predict(test[["date", "channel"] + INPUT_COLS]).rename(columns={"demand": "model"})
        t = test[["date", "channel", "demand"]].merge(p[["date", "channel", "model"]], on=["date", "channel"])
        t["naive"] = seasonal_naive(train, t)
        t["origin"] = o
        t["days_ahead"] = (t["date"] - o).dt.days + 1
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def summarize_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    """WAPE on daily values and on 30-day totals, and bias, for the model and the benchmark, per channel."""
    out = []
    for ch, g in list(bt.groupby("channel")) + [("Total", bt.groupby(["origin", "date"], as_index=False)[["demand", "model", "naive"]].sum())]:
        g = g.assign(block=((g["date"] - g["origin"]).dt.days // 30) if "origin" in g else 0)
        blocks = g.groupby(["origin", "block"])[["demand", "model", "naive"]].sum()
        for name in ("model", "naive"):
            out.append({"channel": ch, "forecast": name, "daily WAPE": wape(g["demand"], g[name]),
                        "30-day WAPE": wape(blocks["demand"], blocks[name]), "bias": bias(g["demand"], g[name])})
    return pd.DataFrame(out)


def _known_inputs(train: pd.DataFrame, days: pd.DatetimeIndex, ch: str, promo: str, spend: str,
                  actual: pd.DataFrame | None) -> pd.DataFrame:
    """Inputs for forecast days, built from what was known on the forecast date.

    promo / spend = "last_year": same weekday last year from the training days (two years back for days whose
    last-year date is itself after the forecast date). "actual": what really happened (only for comparison).
    The event calendar is always known in advance.
    """
    from .features import event_calendar
    h = train[train["channel"] == ch].set_index("date")
    f = pd.DataFrame({"date": days, "channel": ch, "event": event_calendar(days).values})
    last = h.index.max()
    back = pd.Series(days - pd.Timedelta(days=364))
    back = back.where(back <= last, pd.Series(days - pd.Timedelta(days=728)))
    ly = h.reindex(back.values)
    for cols, mode in ((["promo_share", "discount_depth"], promo), (["spend_search", "spend_social"], spend)):
        if mode == "actual":
            a = actual[actual["channel"] == ch].set_index("date").reindex(days)
            f[cols] = a[cols].values
        elif mode == "last_year":
            f[cols] = ly[cols].values
        else:
            raise ValueError(f"unknown input mode {mode!r}")
    return f


def back_forecast(daily: pd.DataFrame, as_of: str, period_end: str, promo: str = "last_year",
                  spend: str = "last_year", config: ModelConfig | None = None) -> pd.DataFrame:
    """What the model would have forecast for as_of..period_end if it had run on as_of.

    The model sees only rows before as_of. Forecast inputs come from _known_inputs; actual outcomes are joined
    after predicting, for scoring only.
    """
    as_of, end = pd.Timestamp(as_of), pd.Timestamp(period_end)
    train = daily[daily["date"] < as_of].copy()
    days = pd.date_range(as_of, end)
    actual_inputs = daily[(daily["date"] >= as_of) & (daily["date"] <= end)][["date", "channel"] + INPUT_COLS] \
        if "actual" in (promo, spend) else None
    future = pd.concat([_known_inputs(train, days, ch, promo, spend, actual_inputs) for ch in sorted(train["channel"].unique())],
                       ignore_index=True)
    m = DemandModel(config).fit(train)
    p = m.predict(future)
    act = daily[(daily["date"] >= as_of) & (daily["date"] <= end)][["date", "channel", "dau", "orders", "demand"]]
    return p.merge(act.rename(columns={"dau": "actual_dau", "orders": "actual_orders", "demand": "actual_demand"}),
                   on=["date", "channel"], how="left")


def monthly_view(bf: pd.DataFrame) -> pd.DataFrame:
    """Back-forecast vs actual by month (App + Web), with the traffic / conversion / order-value split of the miss."""
    g = bf.dropna(subset=["actual_demand"]).assign(month=lambda x: x["date"].dt.to_period("M").astype(str)) \
          .groupby("month")[["dau", "orders", "demand", "actual_dau", "actual_orders", "actual_demand"]].sum()
    return pd.DataFrame({
        "actual $M": g["actual_demand"] / 1e6,
        "forecast $M": g["demand"] / 1e6,
        "miss %": 100 * (g["demand"] / g["actual_demand"] - 1),
        "traffic miss %": 100 * (g["dau"] / g["actual_dau"] - 1),
        "CVR miss %": 100 * ((g["orders"] / g["dau"]) / (g["actual_orders"] / g["actual_dau"]) - 1),
        "AOV miss %": 100 * ((g["demand"] / g["orders"]) / (g["actual_demand"] / g["actual_orders"]) - 1),
    }).round(2)
