
"""
forecast_evaluation_simple.py
==============================
Zjednodusena verzia bez tried/dataclass - vsetky funkcie vracaju
obycajny dict. Funkcne identicke s forecast_evaluation.py,
len menej "objektovo" organizovane.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def _loss(e, loss_type="se"):
    if loss_type == "se":
        return e ** 2
    elif loss_type == "ae":
        return np.abs(e)
    raise ValueError("loss_type musi byt 'se' alebo 'ae'")


def _long_run_var_mean(d, h):
    n = len(d)
    d_dm = d - d.mean()
    gamma0 = np.dot(d_dm, d_dm) / n
    var_d = gamma0
    for lag in range(1, max(h - 1, 0) + 1):
        gamma_lag = np.dot(d_dm[lag:], d_dm[:-lag]) / n
        var_d += 2 * gamma_lag
    return var_d / n


def dm_test(y_true, yhat1, yhat2, h=1, loss_type="se", alternative="two-sided"):
    """Diebold-Mariano test s HLN korekciou pre male vzorky. Vracia dict."""
    y_true, yhat1, yhat2 = map(lambda a: np.asarray(a, dtype=float), (y_true, yhat1, yhat2))
    e1, e2 = y_true - yhat1, y_true - yhat2
    d = _loss(e1, loss_type) - _loss(e2, loss_type)
    n = len(d)

    d_bar = d.mean()
    lrv = _long_run_var_mean(d, h)
    if lrv <= 0:
        lrv = d.var(ddof=1) / n

    dm_stat = d_bar / np.sqrt(lrv)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    hln_stat = dm_stat * correction
    df = n - 1

    if alternative == "two-sided":
        p_value = 2 * (1 - stats.t.cdf(abs(hln_stat), df=df))
    elif alternative == "less":
        p_value = stats.t.cdf(hln_stat, df=df)
    else:
        p_value = 1 - stats.t.cdf(hln_stat, df=df)

    return {
        "dm_stat": dm_stat, "hln_stat": hln_stat, "p_value": p_value,
        "df": df, "mean_loss_diff": d_bar, "n_obs": n,
    }


def encompassing_reg_test(y_true, yhat1, yhat2, hac_lags=None):
    """ENC-REG test (Chong-Hendry/Ericsson). Vracia dict."""
    y_true, yhat1, yhat2 = map(lambda a: np.asarray(a, dtype=float), (y_true, yhat1, yhat2))
    n = len(y_true)
    e1 = y_true - yhat1
    x = yhat2 - yhat1
    X = sm.add_constant(x)

    if hac_lags is None:
        hac_lags = max(int(np.floor(4 * (n / 100) ** (2 / 9))), 1)

    model = sm.OLS(e1, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    return {
        "alpha": model.params[1], "se_alpha": model.bse[1],
        "t_stat": model.tvalues[1], "p_value": model.pvalues[1],
        "df": int(model.df_resid), "hac_lags": hac_lags,
    }


def hln_encompassing_test(y_true, yhat1, yhat2, h=1, alternative="one-sided"):
    """ENC-T test (Harvey-Leybourne-Newbold, 1998). Vracia dict."""
    y_true, yhat1, yhat2 = map(lambda a: np.asarray(a, dtype=float), (y_true, yhat1, yhat2))
    e1, e2 = y_true - yhat1, y_true - yhat2
    c = e1 * (e1 - e2)
    n = len(c)

    c_bar = c.mean()
    lrv = _long_run_var_mean(c, h)
    if lrv <= 0:
        lrv = c.var(ddof=1) / n

    raw_stat = c_bar / np.sqrt(lrv)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    enc_t_stat = raw_stat * correction
    df = n - 1

    if alternative == "one-sided":
        p_value = 1 - stats.t.cdf(enc_t_stat, df=df)
    else:
        p_value = 2 * (1 - stats.t.cdf(abs(enc_t_stat), df=df))

    return {"enc_t_stat": enc_t_stat, "p_value": p_value, "df": df, "mean_cov": c_bar, "n_obs": n}


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 120
    y = np.cumsum(rng.normal(0, 1, n)) + 100
    arimax_pred = y + rng.normal(0, 1.5, n)
    xgboost_pred = y + rng.normal(0, 1.0, n)

    print("DM test:", dm_test(y, arimax_pred, xgboost_pred, h=1))
    print("ENC-REG:", encompassing_reg_test(y, arimax_pred, xgboost_pred))
    print("ENC-T:", hln_encompassing_test(y, arimax_pred, xgboost_pred, h=1))
