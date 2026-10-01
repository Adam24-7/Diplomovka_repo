# functions.py
from __future__ import annotations
import pandas as pd
import numpy as np
import pmdarima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import functions_general as fg
from statsmodels.stats.diagnostic import acorr_ljungbox

def _extract_arimax_diagnostics(refit_diagnostics, _step_diagnostics=None):
    residual_records = []
    coefficient_records = []

    date_keys = ("date", "refit_date", "forecast_date", "current_date")

    for entry in refit_diagnostics:
        if isinstance(entry, tuple) and len(entry) == 2:
            refit_date, fit_diag = entry
        elif isinstance(entry, dict):
            date_key = next((k for k in date_keys if k in entry), None)
            refit_date = entry.get(date_key) if date_key else None
            fit_diag = entry
        else:
            raise TypeError(
                f"Neocakavany format prvku v refit_diagnostics: {type(entry)}"
            )

        residual_records.append({
            "forecast_date": refit_date,
            "aic": fit_diag.get("aic", np.nan),
            "bic": fit_diag.get("bic", np.nan),
            "ljung_box_pvalue": fit_diag.get("ljung_box_pvalue", np.nan),
            "order": fit_diag.get("order"),
        })

        coef_dict = fit_diag.get("coefficients", {})
        for param_name, value in coef_dict.items():
            coefficient_records.append({
                "forecast_date": refit_date,
                "parameter": param_name,
                "value": value,
            })

    residual_diagnostics = (
        pd.DataFrame(residual_records).set_index("forecast_date")
        if residual_records else pd.DataFrame()
    )

    coefficients = (
        pd.DataFrame(coefficient_records).set_index("forecast_date")
        if coefficient_records else pd.DataFrame()
    )

    return {
        "coefficients": coefficients,
        "residual_diagnostics": residual_diagnostics,
    }

def _make_arimax_fit_fn(
    order,
    seasonal=False,
    seasonal_order=None,
    maxiter=50,
    method="lbfgs",
):
    def _fit(X_train, y_train, _current_date):
        model_kwargs = {
            "order": order,
            "method": method,
            "maxiter": maxiter,
            "suppress_warnings": True,
        }

        if seasonal:
            if seasonal_order is None:
                raise ValueError(
                    "Pri seasonal=True zadaj seasonal_order, napr. (1, 0, 1, 12)."
                )
            model_kwargs["seasonal_order"] = seasonal_order

        model = pmdarima.arima.ARIMA(**model_kwargs)
        model.fit(y_train, X=X_train)

        arima_res = model.arima_res_

        aic = float(arima_res.aic)
        bic = float(arima_res.bic)

        resid = arima_res.resid
        lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
        ljung_box_pvalue = float(lb_test["lb_pvalue"].iloc[0])

        params = arima_res.params
        coef_dict = params.to_dict()

        fit_diagnostics = {
            "order": order,
            "seasonal": seasonal,
            "seasonal_order": seasonal_order if seasonal else None,
            "train_size": len(y_train),
            "aic": aic,
            "bic": bic,
            "ljung_box_pvalue": ljung_box_pvalue,
            "coefficients": coef_dict,
        }

        return model, fit_diagnostics

    return _fit

def _make_arimax_predict_fn(alpha=0.05):
    def _predict(model, X_row, current_date):
        forecast, conf_int = model.predict(n_periods=1, X=X_row, return_conf_int=True, alpha=alpha)
        pred = float(np.asarray(forecast).ravel()[0])
        ci = np.asarray(conf_int).reshape(-1, 2)[0]
        return pred, {"ci_lower": float(ci[0]), "ci_upper": float(ci[1])}
    return _predict

def rolling_forecast_arimax(X, y_diff, level_series, initial_train_size,
                             order=(1, 0, 1), refit_every=1, seasonal=False,
                             seasonal_order=None, alpha=0.05, maxiter=50,
                             method="lbfgs", verbose=True):
    fit_fn = _make_arimax_fit_fn(order, seasonal, seasonal_order, maxiter=maxiter, method=method)
    predict_fn = _make_arimax_predict_fn(alpha)

    out = fg._rolling_forecast_core(X, y_diff, level_series, initial_train_size,
                                  fit_fn, predict_fn, refit_every, verbose=verbose)
    out["diagnostics"] = _extract_arimax_diagnostics(out["refit_diagnostics"], out["step_diagnostics"])
    return out


def forward_selection_old(dataset, notinc):
    BASE_ORDER = (1,0,1)
    SEAS_ORDER = (0,0,0,12)

    # Exogenne kandadata — vsetky okrem cielovej premennej
    exog_pool = [c for c in dataset.columns if c != notinc] 

    selected_cols = []
    best_bic      = np.inf

    print("BIC forward selection:")
    print("-" * 50)

    for iteration in range(len(exog_pool)):
        remaining = [c for c in exog_pool if c not in selected_cols]
        if not remaining:
            break

        candidate_bics = {}
        for col in remaining:
            trial_cols = selected_cols + [col]
            try:
                m = SARIMAX(
                    dataset[notinc], 
                    exog=dataset [trial_cols],
                    order=BASE_ORDER,
                    seasonal_order=SEAS_ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                candidate_bics[col] = m.bic # type: ignore[reportAttributeAccessIssue]
            except Exception:
                candidate_bics[col] = np.inf

        best_candidate = min(candidate_bics, key=candidate_bics.get) # type: ignore[reportAttributeAccessIssue]
        best_candidate_bic = candidate_bics[best_candidate]

        if best_candidate_bic < best_bic:
            best_bic = best_candidate_bic
            selected_cols.append(best_candidate)
            print(f"  + '{best_candidate}'  BIC = {best_bic:.2f}")
        else:
            print(f"  Zastavenie — pridanie ziadnej premennej nezlepsuje BIC.")
            break

    print("-" * 50)
    print(f"Vybrane premenne: {selected_cols}")
    print(f"Finalny BIC    : {best_bic:.2f}")

    return selected_cols


def forward_selection(dataset_train, target_col, infc = "bic"):
    BASE_ORDER = (1, 0, 1)
    SEAS_ORDER = (0, 0, 0, 12)

    # Odstránenie cieľovej premennej a prípadných konštantných stĺpcov
    exog_pool = [
        c for c in dataset_train.columns 
        if c != target_col and dataset_train[c].nunique() > 1
    ] 

    selected_cols = []
    best_infc = np.inf

    print(f"Forward selection ({infc}):")
    print("-" * 50)

    while True:
        remaining = [c for c in exog_pool if c not in selected_cols]
        if not remaining:
            break

        candidate_infc = {}
        for col in remaining:
            trial_cols = selected_cols + [col]
            try:
                m = SARIMAX(
                    dataset_train[target_col], 
                    exog=dataset_train[trial_cols],
                    order=BASE_ORDER,
                    seasonal_order=SEAS_ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                if infc == "bic":
                    candidate_infc[col] = m.bic
                elif infc == "aic":
                    candidate_infc[col] = m.aic
                else:
                    print(f"Neplatný názov informačného kritéria {infc}, použije sa BIC.")
                    candidate_infc[col] = m.bic
                        
            except Exception:
                candidate_infc[col] = np.inf

        best_candidate = min(candidate_infc, key=lambda k: candidate_infc[k])
        best_candidate_infc = candidate_infc[best_candidate]

        if best_candidate_infc < best_infc:
            best_infc = best_candidate_infc
            selected_cols.append(best_candidate)
            print(f"  + '{best_candidate}'  {infc} = {best_infc:.2f}")
        else:
            print(f"  Zastavenie — pridanie ďalšej premennej nezlepšuje {infc}.")
            break

    print("-" * 50)
    print(f"Vybrané premenné: {selected_cols}")
    print(f"Finálny {infc}     : {best_infc:.2f}")

    return selected_cols

