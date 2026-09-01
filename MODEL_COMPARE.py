"""
forecast_evaluation_simple.py
==============================
Zjednodusena verzia bez tried/dataclass - vsetky funkcie vracaju
obycajny dict. Vystupy funkcii su formatovane pomocou pomocnej
funkcie print_result() pre prehladnost priamo v konzole / notebooku.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


# ----------------------------------------------------------------------
# Pomocne funkcie
# ----------------------------------------------------------------------

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


def _stars(p_value):
    """Hviezdicky vyznamnosti: *** p<0.01, ** p<0.05, * p<0.10."""
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.10:
        return "*"
    return ""


# ----------------------------------------------------------------------
# Formatovany vypis vysledkov (namiesto .summary() metody v triede)
# ----------------------------------------------------------------------

# Popisky pre kazdy typ testu - nazov kluca -> (popisny text, format cisla)
_LABELS = {
    "dm_test": {
        "title": "Diebold-Mariano test (HLN korekcia pre male vzorky)",
        "fields": [
            ("dm_stat",        "DM statistika (asymptoticka)", "{:.4f}"),
            ("hln_stat",       "HLN statistika (male vzorky)", "{:.4f}"),
            ("p_value",        "p-hodnota",                    "{:.4f}"),
            ("df",             "Stupne volnosti",               "{:d}"),
            ("mean_loss_diff", "Priemerny rozdiel strat d_bar", "{:.6f}"),
            ("n_obs",          "Pocet pozorovani (n)",          "{:d}"),
        ],
    },
    "encompassing_reg_test": {
        "title": "Regresny test encompassingu (ENC-REG)",
        "fields": [
            ("alpha",    "Alpha (vaha modelu 2)", "{:.4f}"),
            ("se_alpha", "SE(alpha), HAC",         "{:.4f}"),
            ("t_stat",   "t-statistika",           "{:.4f}"),
            ("p_value",  "p-hodnota",              "{:.4f}"),
            ("df",       "Stupne volnosti",        "{:d}"),
            ("hac_lags", "Pocet HAC lagov",        "{:d}"),
        ],
    },
    "hln_encompassing_test": {
        "title": "Harvey-Leybourne-Newbold test encompassingu (ENC-T)",
        "fields": [
            ("enc_t_stat", "ENC-T statistika",        "{:.4f}"),
            ("p_value",    "p-hodnota",               "{:.4f}"),
            ("df",         "Stupne volnosti",         "{:d}"),
            ("mean_cov",   "Priemer c_t = e1*(e1-e2)", "{:.6f}"),
            ("n_obs",      "Pocet pozorovani (n)",     "{:d}"),
        ],
    },
}


def print_result(result: dict, test_name: str, label: str = "") -> None:
    """
    Prehladny konzolovy vypis vysledku testu.

    Parametre
    ---------
    result : dict vrateny z dm_test(), encompassing_reg_test()
        alebo hln_encompassing_test()
    test_name : "dm_test", "encompassing_reg_test" alebo "hln_encompassing_test"
    label : volitelny popis (napr. "ARIMAX vs XGBoost"), zobrazi sa v hlavicke
    """
    spec = _LABELS[test_name]
    width = 62

    header = spec["title"]
    if label:
        header += f"  [{label}]"

    print("=" * width)
    print(header)
    print("-" * width)
    for key, description, fmt in spec["fields"]:
        value = result[key]
        formatted_value = fmt.format(value)
        if key == "p_value":
            formatted_value += f"  {_stars(value)}"
        print(f"  {description:<32}: {formatted_value}")
    print("=" * width)


def format_pvalue_table(df_pvalues: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """
    Prida hviezdicky vyznamnosti k matici p-hodnot (napr. z pairwise_dm_matrix)
    pre prehladnejsie zobrazenie v konzole / exporte do tabulky.
    """
    def fmt(v):
        if pd.isna(v):
            return ""
        return f"{v:.{decimals}f}{_stars(v)}"

    return df_pvalues.map(fmt)


# ----------------------------------------------------------------------
# Diebold-Mariano test
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# ENC-REG test
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# ENC-T test (HLN)
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Porovnanie viacerych modelov naraz (pairwise matica p-hodnot)
# ----------------------------------------------------------------------

def pairwise_dm_matrix(y_true, forecasts: dict, h=1, loss_type="se") -> pd.DataFrame:
    """Matica p-hodnot DM testu (two-sided) pre vsetky dvojice modelov."""
    names = list(forecasts.keys())
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i >= j:
                continue
            res = dm_test(y_true, forecasts[name_i], forecasts[name_j], h=h, loss_type=loss_type)
            mat.loc[name_i, name_j] = res["p_value"]
            mat.loc[name_j, name_i] = res["p_value"]
    return mat


# ----------------------------------------------------------------------
# DEMONSTRACIA
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 120
    y = np.cumsum(rng.normal(0, 1, n)) + 100
    arimax_pred = y + rng.normal(0, 1.5, n)
    elasticnet_pred = y + rng.normal(0.2, 1.3, n)
    xgboost_pred = y + rng.normal(0, 1.0, n)

    dm_res = dm_test(y, arimax_pred, xgboost_pred, h=1)
    print_result(dm_res, "dm_test", label="ARIMAX vs XGBoost")
    print()

    enc_reg_res = encompassing_reg_test(y, arimax_pred, xgboost_pred)
    print_result(enc_reg_res, "encompassing_reg_test", label="ARIMAX vs XGBoost")
    print()

    enc_t_res = hln_encompassing_test(y, arimax_pred, xgboost_pred, h=1)
    print_result(enc_t_res, "hln_encompassing_test", label="ARIMAX vs XGBoost")
    print()

    forecasts = {"ARIMAX": arimax_pred, "ElasticNet": elasticnet_pred, "XGBoost": xgboost_pred}
    p_matrix = pairwise_dm_matrix(y, forecasts, h=1)
    print("Matica p-hodnot DM testu (s hviezdickami vyznamnosti):")
    print(format_pvalue_table(p_matrix).to_string())
