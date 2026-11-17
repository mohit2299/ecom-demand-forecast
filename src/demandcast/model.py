"""Demand = traffic x conversion x order value, one set of regressions per channel.

    log(DAU)        = trend + weekday + yearly seasonality + events + tracking period + paid media
    logit(CVR)      = trend + weekday + yearly seasonality + events + tracking period + promo share + paid media
    log(AOV)        = trend + weekday + yearly seasonality + events + discount depth
    orders = DAU x CVR,  demand = orders x AOV

Paid-media terms are log(adstocked spend / training median), so each coefficient is an elasticity. A spend
effect is kept only if it is significant with HAC standard errors (daily errors are autocorrelated) and has the
expected sign: more spend brings more traffic (> 0) and, if anything, lower-intent traffic (CVR effect <= 0).
Effects that fail are dropped and the regression is refitted without them.

Tracking breaks are found on the training days only (log DAU / orders steps), so a backtest never uses a break
it couldn't have seen. Forecast days get the last tracking period.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .features import detect_level_shifts, fourier, regimes_from_breaks, spend_terms

MEDIA = ("search", "social")


@dataclass
class ModelConfig:
    adstock_decay: dict = field(default_factory=lambda: {"search": 0.3, "social": 0.6})
    fourier_k: int = 3
    t_min: float = 2.0              # |t| needed to keep a spend effect
    hac_lags: int = 7
    use_spend: bool = True
    use_events: bool = True
    detect_breaks: bool = True


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class DemandModel:
    def __init__(self, config: ModelConfig | None = None):
        self.cfg = config or ModelConfig()
        self.fits: dict = {}

    # ---------- design ----------
    def _design(self, df: pd.DataFrame, ch: str, sp: pd.DataFrame | None) -> pd.DataFrame:
        st = self.fits[ch]
        d = pd.DatetimeIndex(df["date"])
        X = pd.DataFrame(index=df.index)
        X["const"] = 1.0
        X["trend"] = (d - st["t0"]).days.values / 365.0
        for k in range(1, 7):
            X[f"dow{k}"] = (d.day_of_week == k).astype(float)
        X = X.join(fourier(d, k=self.cfg.fourier_k).set_index(df.index))
        if self.cfg.use_events:
            for e in st["events"]:
                X[f"ev_{e}"] = (df["event"].values == e).astype(float)
        for r in range(1, st["n_regimes"]):
            X[f"regime{r}"] = (df["regime"].values == r).astype(float)
        X["promo_share"] = df["promo_share"].values
        X["discount_depth"] = df["discount_depth"].values
        if sp is not None:
            X = X.join(sp.set_index(df.index))
        return X

    def _spend_terms(self, ch: str, rows: pd.DataFrame) -> pd.DataFrame | None:
        """Adstock runs over training spend followed by these rows' spend, so carry-over flows into the forecast."""
        if not self.cfg.use_spend:
            return None
        st = self.fits[ch]
        s = pd.concat([st["spend_hist"], rows.set_index("date")[[f"spend_{m}" for m in MEDIA]]])
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.columns = list(MEDIA)
        terms, _ = spend_terms(s, self.cfg.adstock_decay, st["spend_ref"])
        return terms.reindex(pd.DatetimeIndex(rows["date"])).reset_index(drop=True)

    # ---------- fit ----------
    def fit(self, daily: pd.DataFrame) -> "DemandModel":
        for ch, g in daily.groupby("channel"):
            g = g.sort_values("date").reset_index(drop=True)
            st = {"t0": pd.Timestamp(g["date"].min()), "events": sorted(set(g["event"]) - {"None"})}
            breaks = detect_level_shifts(pd.Series(np.log(g["dau"] / g["orders"]).values, index=g["date"])) \
                if self.cfg.detect_breaks else []
            g["regime"] = regimes_from_breaks(g["date"], breaks)
            st.update(breaks=breaks, n_regimes=len(breaks) + 1, last_regime=int(g["regime"].iloc[-1]))
            hist = g.set_index("date")[[f"spend_{m}" for m in MEDIA]]
            st["spend_hist"] = hist
            if self.cfg.use_spend:
                s = hist.copy(); s.columns = list(MEDIA)
                _, st["spend_ref"] = spend_terms(s, self.cfg.adstock_decay)
            self.fits[ch] = st
            sp = self._spend_terms(ch, g)
            X = self._design(g, ch, sp)
            y = {"dau": np.log(g["dau"]), "cvr": _logit(g["orders"] / g["dau"]), "aov": np.log(g["demand"] / g["orders"])}
            cols = {
                "dau": [c for c in X if c not in ("promo_share", "discount_depth")],
                "cvr": [c for c in X if c != "discount_depth"],
                "aov": [c for c in X if not c.startswith(("sp_", "regime")) and c != "promo_share"],
            }
            st["models"], st["spend_effects"] = {}, []
            for piece in ("dau", "cvr", "aov"):
                use = cols[piece]
                for _ in range(3):                                # drop failing spend effects, refit
                    res = sm.OLS(y[piece], X[use]).fit(cov_type="HAC", cov_kwds={"maxlags": self.cfg.hac_lags})
                    bad = []
                    for c in [c for c in use if c.startswith("sp_")]:
                        ok_sign = res.params[c] > 0 if piece == "dau" else res.params[c] <= 0
                        if not (ok_sign and abs(res.tvalues[c]) >= self.cfg.t_min):
                            bad.append(c)
                    if not bad:
                        break
                    use = [c for c in use if c not in bad]
                st["models"][piece] = (res, use)
                for c in [c for c in cols[piece] if c.startswith("sp_")]:
                    kept = c in use
                    st["spend_effects"].append({"channel": ch, "piece": piece, "media": c[3:],
                                                "elasticity": float(res.params[c]) if kept else 0.0,
                                                "t": float(res.tvalues[c]) if kept else np.nan, "kept": kept})
        return self

    # ---------- predict ----------
    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """future: date, channel, event, promo_share, discount_depth, spend_search, spend_social."""
        out = []
        for ch, f in future.groupby("channel"):
            st = self.fits[ch]
            f = f.sort_values("date").reset_index(drop=True).copy()
            f["regime"] = st["last_regime"]
            X = self._design(f, ch, self._spend_terms(ch, f))
            pred = {}
            for piece, (res, use) in st["models"].items():
                pred[piece] = X[use].values @ res.params[use].values
            dau, cvr, aov = np.exp(pred["dau"]), 1 / (1 + np.exp(-pred["cvr"])), np.exp(pred["aov"])
            out.append(pd.DataFrame({"date": f["date"].values, "channel": ch, "dau": dau, "cvr": cvr, "aov": aov,
                                     "orders": dau * cvr, "demand": dau * cvr * aov}))
        return pd.concat(out, ignore_index=True)

    def spend_effects(self) -> pd.DataFrame:
        return pd.DataFrame([r for st in self.fits.values() for r in st["spend_effects"]])

    def breaks(self) -> dict:
        return {ch: st["breaks"] for ch, st in self.fits.items()}


def spend_response(model: DemandModel, future: pd.DataFrame, media: str, change: float = 0.10) -> dict:
    """Extra demand and marginal ROAS if `media` spend is `change` higher on every future day."""
    base = model.predict(future)
    f2 = future.copy()
    f2[f"spend_{media}"] *= 1 + change
    alt = model.predict(f2)
    extra_demand = alt["demand"].sum() - base["demand"].sum()
    extra_spend = future[f"spend_{media}"].sum() * change
    return {"media": media, "extra_spend": extra_spend, "extra_demand": extra_demand,
            "marginal_roas": extra_demand / extra_spend if extra_spend else np.nan}
