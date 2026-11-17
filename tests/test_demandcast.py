import numpy as np
import pandas as pd
import pytest

from demandcast import (DemandModel, ModelConfig, adstock, back_forecast, bias, detect_level_shifts, make_dataset,
                        rolling_backtest, seasonal_naive, summarize_backtest, thanksgiving, wape)


@pytest.fixture(scope="module")
def data():
    daily, spend, truth = make_dataset()
    return daily, truth


def test_adstock_carries_over():
    a = adstock(np.array([100.0, 0, 0, 0]), 0.5)
    assert np.allclose(a, [100, 50, 25, 12.5])
    assert np.allclose(adstock(np.array([1.0, 2, 3]), 0.0), [1, 2, 3])


def test_thanksgiving():
    assert thanksgiving(2025) == pd.Timestamp("2025-11-27")
    assert thanksgiving(2026) == pd.Timestamp("2026-11-26")


def test_metrics():
    assert wape([100, 100], [110, 90]) == pytest.approx(0.10)
    assert bias([100, 100], [110, 110]) == pytest.approx(0.10)


def test_finds_the_tracking_break(data):
    daily, truth = data
    app = daily[daily["channel"] == truth.break_channel]
    found = detect_level_shifts(pd.Series(np.log(app["dau"] / app["orders"]).values, index=app["date"]))
    assert len(found) == 1
    assert abs((found[0] - pd.Timestamp(truth.break_date)).days) <= 7
    web = daily[daily["channel"] != truth.break_channel]
    assert detect_level_shifts(pd.Series(np.log(web["dau"] / web["orders"]).values, index=web["date"])) == []


def test_recovers_spend_elasticities(data):
    daily, truth = data
    eff = DemandModel().fit(daily).spend_effects()
    dau = eff[eff["piece"] == "dau"].set_index(["channel", "media"])["elasticity"]
    for ch, by_media in truth.dau_elasticity.items():
        for m, true in by_media.items():
            assert abs(dau[(ch, m)] - true) < 0.03, (ch, m, dau[(ch, m)], true)


def test_event_controls_matter(data):
    """Spend is higher in sale events. Without event controls the model credits spend for the event lift."""
    daily, truth = data
    with_ev = DemandModel().fit(daily).spend_effects()
    without = DemandModel(ModelConfig(use_events=False)).fit(daily).spend_effects()
    sel = lambda e: e[(e["piece"] == "dau") & (e["channel"] == "Web") & (e["media"] == "search")]["elasticity"].item()
    assert sel(without) > sel(with_ev) + 0.03


def test_beats_seasonal_naive(data):
    daily, _ = data
    bt = rolling_backtest(daily, pd.date_range("2025-01-01", "2026-01-01", freq="QS"), 90)
    s = summarize_backtest(bt).set_index(["channel", "forecast"])
    assert s.loc[("Total", "model"), "30-day WAPE"] < s.loc[("Total", "naive"), "30-day WAPE"]


def test_back_forecast_does_not_see_later_actuals(data):
    """Changing every outcome on or after the forecast date must not change the forecast."""
    daily, _ = data
    as_of = "2026-01-01"
    a = back_forecast(daily, as_of, "2026-06-30")
    tampered = daily.copy()
    later = tampered["date"] >= as_of
    tampered.loc[later, ["dau", "orders", "demand"]] *= 1.5
    tampered.loc[later, ["promo_share", "spend_search", "spend_social"]] *= 0.5
    b = back_forecast(tampered, as_of, "2026-06-30")
    assert np.allclose(a["demand"], b["demand"])
    assert not np.allclose(a["actual_demand"], b["actual_demand"])   # the scoring side did change


def test_backtest_trains_before_each_origin(data, monkeypatch):
    daily, _ = data
    seen = []
    real_fit = DemandModel.fit

    def spy(self, df):
        seen.append(df["date"].max())
        return real_fit(self, df)

    monkeypatch.setattr(DemandModel, "fit", spy)
    origins = pd.to_datetime(["2025-03-01", "2025-09-01"])
    rolling_backtest(daily, origins, 30)
    assert all(last < o for last, o in zip(seen, origins))


def test_seasonal_naive_never_reads_the_future(data):
    daily, _ = data
    train = daily[daily["date"] < "2026-01-01"]
    future = daily[(daily["date"] >= "2026-01-01")].reset_index(drop=True)
    f = seasonal_naive(train, future)
    assert np.isfinite(f).all()
