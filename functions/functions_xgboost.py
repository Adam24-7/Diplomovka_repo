# functions.py
from __future__ import annotations
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from typing import cast
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit

import functions_general as fg



def _extract_xgboost_diagnostics(refit_diagnostics, step_diagnostics):
    importance_rows, hyperparam_rows = [], []
    for rec in refit_diagnostics:
        d = rec["forecast_date"]
        for itype in ("gain", "weight", "cover"):
            for feat, val in rec.get(f"importance_{itype}", {}).items():
                importance_rows.append({"forecast_date": d, "feature": feat,
                                         "importance_type": itype, "value": float(val)})
        hyperparam_rows.append({"forecast_date": d, **rec.get("params", {})})
    return {
        "feature_importance": pd.DataFrame(importance_rows),
        "hyperparams": pd.DataFrame(hyperparam_rows).set_index("forecast_date")
                        if hyperparam_rows else pd.DataFrame(),
    }

def _make_xgboost_fit_fn(xgb_params, param_grid=None, cv_splits=5):

    def _fit(X_train, y_train, current_date):
        if param_grid:
            n_splits = min(cv_splits, max(len(X_train) // 10, 2))
            gs = GridSearchCV(XGBRegressor(**xgb_params), param_grid,
                               cv=TimeSeriesSplit(n_splits=n_splits),
                               scoring="neg_mean_squared_error", n_jobs=-1)
            gs.fit(X_train, y_train)
            model, used_params = gs.best_estimator_, {**xgb_params, **gs.best_params_}
        else:
            model = XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)
            used_params = dict(xgb_params)

        b = model.get_booster()
        diag = {"importance_gain": b.get_score(importance_type="gain"),
                "importance_weight": b.get_score(importance_type="weight"),
                "importance_cover": b.get_score(importance_type="cover"),
                "params": used_params}
        return model, diag
    return _fit

def _xgboost_predict_fn(model, X_row, current_date):
    return float(np.asarray(model.predict(X_row)).ravel()[0]), {}

def rolling_forecast_xgboost(X, y_diff, level_series, initial_train_size,
                              refit_every=1, xgb_params=None, param_grid=None,
                              cv_splits=5, objective="reg:squarederror", verbose=True):
    default_params = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
                       "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42}
    xgb_params = {**default_params, **(xgb_params or {}), "objective": objective}
    fit_fn = _make_xgboost_fit_fn(xgb_params, param_grid, cv_splits)

    out = fg._rolling_forecast_core(X, y_diff, level_series, initial_train_size,
                                  fit_fn, _xgboost_predict_fn, refit_every, verbose=verbose)
    out["diagnostics"] = _extract_xgboost_diagnostics(out["refit_diagnostics"], out["step_diagnostics"])
    return out

def make_features(data, target, exog_cols, target_lags=(1, 2, 3, 6, 12), exog_lags=(1,)):
    feat = pd.DataFrame(index=data.index)

    for lag in target_lags:
        feat[f"{target}_lag{lag}"] = data[target].shift(lag)

    for col in exog_cols:
        for lag in exog_lags:
            feat[f"{col}_lag{lag}"] = data[col].shift(lag)

    feat["month"] = data.index.month
    feat["quarter"] = data.index.quarter
    feat["trend"] = np.arange(len(data))

    feat[target] = data[target]

    return feat.dropna()

def hyperparameters_tuning(X_train, y_train, random_state = 24, n_splits = 5):

        #tuning hyperparametrov
    tscv = TimeSeriesSplit(n_splits=n_splits)

    #random_search
    random_param_space = {
        "n_estimators": [100, 200, 300, 400, 600],
        "max_depth": [1, 2, 3, 4, 5, 6],
        "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [0.5, 1, 1.5, 2],
        "gamma": [0, 0.1, 0.5, 1, 5],
    }

    random_search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=random_state),
        param_distributions=random_param_space,
        n_iter=60,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    random_search.fit(X_train, y_train)
    print("Najlepsie parametre z Random Search:", random_search.best_params_)

    best = random_search.best_params_

    #grid_search
    grid_param_space = {
        "n_estimators": sorted(set([max(50, best["n_estimators"] - 100), best["n_estimators"], best["n_estimators"] + 100])),
        "max_depth": sorted(set([max(1, best["max_depth"] - 1), best["max_depth"], best["max_depth"] + 1])),
        "learning_rate": sorted(set([round(best["learning_rate"] * 0.5, 4), best["learning_rate"], round(best["learning_rate"] * 1.5, 4)])),
        "gamma": sorted(set([max(0, best["gamma"] - 1), best["gamma"], best["gamma"] + 1])),
        "subsample": [best["subsample"]],
        "colsample_bytree": [best["colsample_bytree"]],
        "min_child_weight": [best["min_child_weight"]],
        "reg_alpha": [best["reg_alpha"]],
        "reg_lambda": [best["reg_lambda"]],
    }

    grid_search = GridSearchCV(
        estimator=XGBRegressor(random_state=random_state),
        param_grid=grid_param_space,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Finalne parametre po Grid Search:", best_params)

    return grid_search, best_params


def rolling_forecast_xgb(
    X: pd.DataFrame,
    y_diff: pd.Series,
    level_series: pd.Series | pd.DataFrame,
    initial_train_size: int,
    last_train_value: float,
    horizon: int = 1,
    refit_every: int = 1,
    objective: str = "reg:squarederror",
    xgb_params: dict | None = None,
) -> tuple[pd.DataFrame, dict]:

    if horizon != 1:
        raise ValueError("Tato funkcia podporuje iba horizon=1.")

    if xgb_params is None:
        xgb_params = {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
    elif not isinstance(xgb_params, dict):
        raise TypeError("xgb_params musi byt dict alebo None.")
    else:
        xgb_params = xgb_params.copy()

    xgb_params["objective"] = objective

    
    if not isinstance(y_diff, pd.Series):
        raise TypeError("y_diff musi byt pandas Series.")

    if not y_diff.index.is_unique:
        raise ValueError(
            "y_diff obsahuje duplicitne datumy v indexe. "
            "Kazdy datum musi mat prave jednu hodnotu."
        )

    if isinstance(level_series, pd.DataFrame):
        if level_series.shape[1] != 1:
            raise ValueError(
                "level_series musi byt pd.Series alebo DataFrame "
                "s presne jednym stlpcom. "
                f"Aktualne stlpce: {level_series.columns.tolist()}"
            )
        level_series = level_series.iloc[:, 0]

    if not isinstance(level_series, pd.Series):
        raise TypeError("level_series musi byt pandas Series.")

    if not level_series.index.is_unique:
        raise ValueError(
            "level_series obsahuje duplicitne datumy v indexe. "
            "Kazdy datum musi mat prave jednu hodnotu."
        )

    if len(X) != len(y_diff):
        raise ValueError("X a y_diff musia mat rovnaky pocet riadkov.")

    if not X.index.equals(y_diff.index):
        raise ValueError("Index X a y_diff nie je zarovnany.")

    if not level_series.index.is_monotonic_increasing:
        level_series = level_series.sort_index()

    if initial_train_size <= 0 or initial_train_size >= len(y_diff):
        raise ValueError("initial_train_size musi byt medzi 1 a len(y_diff)-1.")

    if refit_every < 1:
        raise ValueError("refit_every musi byt aspon 1.")

    missing_dates = y_diff.index.difference(level_series.index)
    if len(missing_dates) > 0:
        raise ValueError(
            "level_series neobsahuje vsetky datumy z y_diff. "
            f"Chybajuce datumy: {missing_dates.tolist()}"
        )

    
    records = []
    models_by_step: dict = {}
    model = None
    previous_predicted_level = float(last_train_value)
    refit_date = None
    misaligned_dates: list = []  

    y_diff_values = y_diff.to_numpy(dtype=float)
    level_values = level_series.to_numpy(dtype=float)

    for t in range(initial_train_size, len(y_diff)):
        current_date = y_diff.index[t]

        if model is None or (t - initial_train_size) % refit_every == 0:
            model = XGBRegressor(**xgb_params)
            model.fit(X.iloc[:t], y_diff.iloc[:t])

            models_by_step[current_date] = model
            refit_date = current_date

        pred_diff = float(
            np.asarray(model.predict(X.iloc[[t]])).ravel()[0]
        )

        actual_diff = float(y_diff_values[t])

        current_position = cast(int, level_series.index.get_loc(current_date))

        actual_level = float(level_values[current_position])

        if current_position == 0:
            raise ValueError(
                f"Pre datum {current_date} neexistuje predchadzajuca "
                "levelova hodnota potrebna pre one-step rekonstrukciu."
            )

        previous_level = float(level_values[current_position - 1])
        predicted_level = float(previous_level + pred_diff)


        implied_diff = actual_level - previous_level
        if not np.isclose(implied_diff, actual_diff, rtol=1e-4, atol=1e-8):
            misaligned_dates.append((current_date, implied_diff, actual_diff))

        error_diff = float(actual_diff - pred_diff)
        error_level = float(actual_level - predicted_level)

        records.append({
            "date": current_date,
            "actual_diff": actual_diff,
            "predicted_diff": pred_diff,
            "actual_level": actual_level,
            "predicted_level": predicted_level,
            "abs_error_diff": abs(error_diff),
            "sq_error_diff": error_diff ** 2,
            "abs_error_level": abs(error_level),
            "sq_error_level": error_level ** 2,
            "pct_error_level": (
                abs(error_level / actual_level) * 100
                if actual_level != 0.0
                else np.nan
            ),
            "error_diff": error_diff,
            "error_level": error_level,
            "refit": refit_date == current_date,
            "model_refit_date": refit_date,
        })

    results = pd.DataFrame(records).set_index("date")

    # ------------------------------------------------------------------
    # Priebezne (rolling) a okno-based metriky
    # ------------------------------------------------------------------
    results["rolling_rmse_diff"] = np.sqrt(
        results["sq_error_diff"].expanding().mean()
    )
    results["rolling_rmse_level"] = np.sqrt(
        results["sq_error_level"].expanding().mean()
    )

    results["rolling_mape_level"] = (
        results["abs_error_level"].expanding().sum()
        / results["actual_level"].abs().expanding().sum()
    ) * 100
    results["rolling_me_level"] = (
        results["error_level"].expanding().mean()
    )

    results["window6_rmse_level"] = np.sqrt(
        results["sq_error_level"].rolling(window=6, min_periods=1).mean()
    )
    results["window6_mape_level"] = (
        results["abs_error_level"].rolling(window=6, min_periods=1).sum()
        / results["actual_level"].abs().rolling(window=6, min_periods=1).sum()
    ) * 100
    results["window6_me_level"] = (
        results["error_level"].rolling(window=6, min_periods=1).mean()
    )

    # ------------------------------------------------------------------
    # Agregovane metriky
    # ------------------------------------------------------------------
    rmse_diff = float(np.sqrt(results["sq_error_diff"].mean()))
    rmse_level = float(np.sqrt(results["sq_error_level"].mean()))

    mape_level = float(
        results["abs_error_level"].sum() / results["actual_level"].abs().sum() * 100
    )
    me_level = float(results["error_level"].mean())

    print(f"RMSE (diferencia): {rmse_diff:.4f}")
    print(f"RMSE (level, onestep rekonstrukcia): {rmse_level:.4f}")
    print(f"MAPE (level, WMAPE): {mape_level:.2f}%")
    print(f"ME (level): {me_level:.4f}")

    if misaligned_dates:
        n_bad = len(misaligned_dates)
        first_example = misaligned_dates[0]
        print(
            f"\n[UPOZORNENIE] Pre {n_bad} z {len(results)} bodov neplati "
            "identita actual_level - previous_level == actual_diff "
            "(mala by platit vzdy pri 'onestep' rekonstrukcii). "
            "To znamena, ze level_series pravdepodobne NIE JE ta ista "
            "premenna/frekvencia/zarovnanie ako y_diff. Priklad "
            f"({first_example[0]}): implied_diff={first_example[1]:.4f} "
            f"vs actual_diff={first_example[2]:.4f}. Skontroluj, ci "
            "level_series = povodny (nediferencovany) rad, z ktoreho "
            "presne vznikol y_diff cez .diff()."
        )

    return results, models_by_step

def extract_tree_structure(model):
    """
    Vrati DataFrame s kompletnou strukturou VSETKYCH stromov v ansambli.
    Kazdy riadok = jeden uzol (vnutorny alebo listovy) v jednom strome.
    """
    trees_df = model.get_booster().trees_to_dataframe()
    return trees_df

def trace_path_for_observation(trees_df, tree_index):
    """Vypise vsetky uzly a podmienky pre dany strom (0 = koren)."""
    path = trees_df[(trees_df["Tree"] == tree_index)]
    return path[["Node", "Feature", "Split", "Yes", "No", "Gain", "Cover"]]