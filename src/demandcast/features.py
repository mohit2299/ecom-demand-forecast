"""Feature building: event calendar, seasonality, adstock, tracking breaks, spend terms."""
from __future__ import annotations

import numpy as np
import pandas as pd


def thanksgiving(year: int) -> pd.Timestamp:
    """US Thanksgiving: the fourth Thursday of November."""
    first = pd.Timestamp(year, 11, 1)
    return first + pd.Timedelta(days=(3 - first.day_of_week) % 7 + 21)


def event_calendar(dates: pd.DatetimeIndex) -> pd.Series:
    """A generic retail sale calendar: one event name per day, "None" on normal days."""
    out = pd.Series("None", index=dates, dtype=object)
    for y in sorted(set(dates.year)):
        tg = thanksgiving(y)
        windows = [("Spring Sale", f"{y}-03-15", f"{y}-03-21"), ("Summer Sale", f"{y}-07-01", f"{y}-07-10"),
                   ("Back to School", f"{y}-08-10", f"{y}-08-20"),
                   ("Black Friday", tg, tg + pd.Timedelta(days=4)),            # Thursday to Cyber Monday
                   ("Holiday", f"{y}-12-15", f"{y}-12-24"), ("Clearance", f"{y}-12-26", f"{y + 1}-01-05")]
        for name, a, b in windows:
            out[(out.index >= pd.Timestamp(a)) & (out.index <= pd.Timestamp(b))] = name
    return out


def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Geometric carry-over: a_t = x_t + decay * a_(t-1). decay = 0 means no carry-over."""
    x = np.nan_to_num(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = v + decay * acc
        out[i] = acc
    return out


def fourier(dates: pd.DatetimeIndex, period: float = 365.25, k: int = 3, prefix: str = "yr") -> pd.DataFrame:
    """Sine / cosine pairs for smooth yearly seasonality."""
    t = dates.day_of_year.values if period > 300 else dates.day_of_week.values
    cols = {}
    for i in range(1, k + 1):
        cols[f"{prefix}_sin{i}"] = np.sin(2 * np.pi * i * t / period)
        cols[f"{prefix}_cos{i}"] = np.cos(2 * np.pi * i * t / period)
    return pd.DataFrame(cols, index=dates)


def detect_level_shifts(series: pd.Series, min_segment: int = 60, threshold: float = 0.08,
                        max_breaks: int = 3) -> list[pd.Timestamp]:
    """Finds step changes in a daily series by binary segmentation on a 7-day rolling median.

    Used on log(DAU / orders): customers change slowly, but a tracking change moves the ratio overnight.
    A break is kept when the medians either side differ by more than `threshold` (in log points) and both
    sides have at least `min_segment` days.
    """
    s = series.rolling(7, center=True, min_periods=4).median().dropna()
    breaks: list[pd.Timestamp] = []

    def split(seg: pd.Series):
        if len(breaks) >= max_breaks or len(seg) < 2 * min_segment:
            return
        v = seg.values
        best, best_i = 0.0, None
        for i in range(min_segment, len(v) - min_segment):
            gap = abs(np.median(v[i:]) - np.median(v[:i]))
            if gap > best:
                best, best_i = gap, i
        if best_i is None or best < threshold:
            return
        breaks.append(seg.index[best_i])
        split(seg.iloc[:best_i])
        split(seg.iloc[best_i:])

    split(s)
    return sorted(breaks)


def regimes_from_breaks(dates: pd.Series, breaks: list[pd.Timestamp]) -> np.ndarray:
    """Tracking period number per day: 0 before the first break, 1 after it, and so on."""
    d = pd.to_datetime(dates).values
    return np.searchsorted(np.array(breaks, dtype="datetime64[ns]"), d, side="right")


def spend_terms(spend: pd.DataFrame, decay: dict, ref: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """log(adstocked spend / reference) per media. The reference (median, from training days) centres the term,
    so a coefficient reads as the % change in the outcome per 1% more spend (an elasticity)."""
    ads = pd.DataFrame({m: adstock(spend[m].values, decay[m]) for m in decay}, index=spend.index)
    ref = ref or {m: float(np.median(ads[m][ads[m] > 0])) for m in decay}
    terms = pd.DataFrame({f"sp_{m}": np.log(np.maximum(ads[m], 1.0) / ref[m]) for m in decay}, index=spend.index)
    return terms, ref
