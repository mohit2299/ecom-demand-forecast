"""Synthetic daily e-commerce data with known ground truth.

The generator plays the role of a retailer with two channels (App, Web). Every effect the model is meant to
learn is set here, so the model can be checked against the truth:

- traffic (DAU) has a trend, weekly and yearly seasonality, sale-event lifts and paid-media effects with carry-over
  (adstock); paid media is the only thing the "true" spend elasticity acts on
- conversion (CVR) rises with the share of orders on promotion and falls slightly with paid social
  (paid traffic converts worse)
- order value (AOV) rises with discount depth (bigger discounts on pricier items)
- the App's DAU counting changes on one day (a tracking break): measured DAU jumps, real customers don't
- spend is higher during sale events, as in real life, so a model without event controls over-credits spend
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import adstock, event_calendar

CHANNELS = ("App", "Web")
MEDIA = ("search", "social")


@dataclass
class Truth:
    """The parameters the data is generated with."""
    dau_elasticity: dict = field(default_factory=lambda: {"App": {"search": 0.05, "social": 0.07},
                                                         "Web": {"search": 0.08, "social": 0.04}})
    cvr_social: dict = field(default_factory=lambda: {"App": -0.03, "Web": -0.02})   # logit-scale effect of log spend
    adstock_decay: dict = field(default_factory=lambda: {"search": 0.3, "social": 0.6})
    channel_share: dict = field(default_factory=lambda: {"App": 0.45, "Web": 0.55})  # share of each media's spend
    promo_cvr: float = 1.6          # logit CVR per unit of promo share
    depth_aov: float = 0.8          # log AOV per unit of discount depth
    break_channel: str = "App"
    break_date: str = "2025-06-15"
    break_shift: float = 0.18       # measured DAU is 18% higher from the break on (counting change only)


def _spend(dates: pd.DatetimeIndex, events: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """Daily spend per media: a base level, a Q4 ramp, flighted campaign bursts and more spend in sale events."""
    t = np.arange(len(dates))
    doy = dates.day_of_year.values
    out = {}
    for m, base in [("search", 40_000.0), ("social", 25_000.0)]:
        level = base * (1 + 0.15 * t / 365)                                   # budgets grow ~15% a year
        q4 = 1 + 0.5 * np.exp(-0.5 * ((doy - 330) / 18) ** 2)                 # peak season ramp
        bursts = np.ones(len(dates))
        for start in rng.choice(len(dates) - 30, size=len(dates) // 45, replace=False):   # ~8 bursts a year
            bursts[start:start + rng.integers(7, 21)] *= rng.uniform(1.4, 2.2)
        dark = np.ones(len(dates))
        for start in rng.choice(len(dates) - 30, size=len(dates) // 180, replace=False):  # occasional pauses
            dark[start:start + rng.integers(5, 12)] *= rng.uniform(0.2, 0.5)
        event_boost = np.where(events.values != "None", 1.35, 1.0)
        noise = np.exp(rng.normal(0, 0.12, len(dates)))
        out[m] = level * q4 * bursts * dark * event_boost * noise
    return pd.DataFrame(out, index=dates)


def make_dataset(start: str = "2023-01-01", end: str = "2026-06-30", seed: int = 7,
                 truth: Truth | None = None) -> tuple[pd.DataFrame, pd.DataFrame, Truth]:
    """Returns (daily, spend, truth).

    daily: one row per date x channel with dau, orders, demand, promo_share, discount_depth, event, channel spend
    spend: total daily spend per media (date index)
    """
    truth = truth or Truth()
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    events = event_calendar(dates)
    spend = _spend(dates, events, rng)
    t = np.arange(len(dates)) / 365.0
    doy = dates.day_of_year.values
    dow = dates.day_of_week.values

    lift = {"None": 0.0, "Spring Sale": 0.25, "Summer Sale": 0.30, "Back to School": 0.15,
            "Black Friday": 0.70, "Holiday": 0.35, "Clearance": 0.20}
    promo_lift = {"None": 0.0, "Spring Sale": 0.20, "Summer Sale": 0.25, "Back to School": 0.10,
                  "Black Friday": 0.35, "Holiday": 0.20, "Clearance": 0.40}
    ev = events.values

    rows = []
    for ch in CHANNELS:
        share = truth.channel_share[ch]
        base = {"App": np.log(180_000), "Web": np.log(260_000)}[ch]
        growth = {"App": 0.12, "Web": 0.03}[ch]
        weekly = {"App": [0.00, -0.02, -0.03, -0.02, 0.02, 0.06, 0.05],
                  "Web": [0.04, 0.02, 0.00, -0.01, -0.02, -0.06, -0.01]}[ch]
        # paid media: log of adstocked spend relative to its median
        sp_terms = {}
        for m in MEDIA:
            a = adstock(spend[m].values * share, truth.adstock_decay[m])
            sp_terms[m] = np.log(a / np.median(a))
        # AR(1) noise so day-to-day errors are correlated, as in real traffic
        e = np.zeros(len(dates))
        shocks = rng.normal(0, 0.05, len(dates))
        for i in range(1, len(dates)):
            e[i] = 0.5 * e[i - 1] + shocks[i]
        log_dau = (base + growth * t + np.array(weekly)[dow]
                   + 0.12 * np.sin(2 * np.pi * doy / 365.25) + 0.06 * np.cos(2 * np.pi * doy / 365.25)
                   + np.array([lift[x] for x in ev])
                   + sum(truth.dau_elasticity[ch][m] * sp_terms[m] for m in MEDIA) + e)
        true_dau = np.exp(log_dau)

        promo_share = np.clip(0.25 + np.array([promo_lift[x] for x in ev]) + rng.normal(0, 0.03, len(dates)), 0.02, 0.95)
        depth = np.clip(0.22 + 0.5 * np.array([promo_lift[x] for x in ev]) + rng.normal(0, 0.02, len(dates)), 0.05, 0.7)
        logit_cvr = ({"App": -3.2, "Web": -3.6}[ch] + truth.promo_cvr * promo_share
                     + truth.cvr_social[ch] * sp_terms["social"] + 0.03 * np.cos(2 * np.pi * dow / 7)
                     + rng.normal(0, 0.04, len(dates)))
        cvr = 1 / (1 + np.exp(-logit_cvr))
        aov = np.exp({"App": np.log(95), "Web": np.log(110)}[ch] + truth.depth_aov * depth + rng.normal(0, 0.03, len(dates)))

        orders = rng.poisson(true_dau * cvr).astype(float)
        demand = orders * aov
        measured_dau = true_dau.copy()
        if ch == truth.break_channel:                                       # tracking break: counting changes
            measured_dau[dates >= pd.Timestamp(truth.break_date)] *= np.exp(truth.break_shift)
        rows.append(pd.DataFrame({
            "date": dates, "channel": ch, "dau": np.round(measured_dau), "orders": orders, "demand": demand,
            "promo_share": promo_share, "discount_depth": depth, "event": ev,
            **{f"spend_{m}": spend[m].values * share for m in MEDIA},
        }))
    daily = pd.concat(rows, ignore_index=True)
    return daily, spend, truth
