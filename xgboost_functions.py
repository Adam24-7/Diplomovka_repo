# functions.py
from __future__ import annotations
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from typing import cast

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
    reconstruction: str = "onestep",
) -> tuple[pd.DataFrame, dict]:
    """
    Rolling one-step XGBoost forecast s rekonstrukciou na povodnu uroven.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matica zarovnana s y_diff.
    y_diff : pd.Series
        Cielova premenna, ktoru XGBoost predikuje - typicky prva diferencia.
    level_series : pd.Series
        Povodna nediferencovana uroven, napr. df_pp["pp_sa"].
    initial_train_size : int
        Pocet pozorovani dostupnych pred prvym testovacim bodom.
    last_train_value : float
        Posledna znama uroven pred prvym testovacim bodom.
    horizon : int
        Aktualna implementacia je urcena pre horizon=1.
    refit_every : int
        Refit pri kazdom n-tom testovacom kroku.
    objective : str
        XGBoost objective funkcia.
    xgb_params : dict
        Hyperparametre XGBRegressor.
    reconstruction : str
        "onestep" pouzije skutocny predchadzajuci level.
        "cumulative" pouzije predikovany predchadzajuci level.

    Returns
    -------
    results : pd.DataFrame
        Vysledky po jednotlivych testovacich mesiacoch.
    models_by_step : dict
        Model pouzity v jednotlivych refit krokoch.
    """

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
        # fix: jasna chyba namiesto neskoreho a mätúceho AttributeError pri .copy()
        raise TypeError("xgb_params musi byt dict alebo None.")
    else:
        xgb_params = xgb_params.copy()

    xgb_params["objective"] = objective

    # ------------------------------------------------------------------
    # Kontroly vstupov
    # ------------------------------------------------------------------
    if not isinstance(y_diff, pd.Series):
        raise TypeError("y_diff musi byt pandas Series.")

    # fix: bez unikatneho indexu na y_diff/X by mohlo dojst k tichemu
    # prepisaniu zaznamov v models_by_step (kluc = current_date).
    if not y_diff.index.is_unique:
        raise ValueError(
            "y_diff obsahuje duplicitne datumy v indexe. "
            "Kazdy datum musi mat prave jednu hodnotu."
        )

    # Dovoli aj DataFrame s presne jednym stlpcom, interne ho premeni na Series.
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

    # Dolezita kontrola: levelovy rad musi obsahovat vsetky datumy modelovaneho radu.
    missing_dates = y_diff.index.difference(level_series.index)
    if len(missing_dates) > 0:
        raise ValueError(
            "level_series neobsahuje vsetky datumy z y_diff. "
            f"Chybajuce datumy: {missing_dates.tolist()}"
        )

    # ------------------------------------------------------------------
    # Rolling one-step cyklus
    # ------------------------------------------------------------------
    records = []
    models_by_step: dict = {}
    model = None
    previous_predicted_level = float(last_train_value)
    refit_date = None
    misaligned_dates: list = []  # fix: diagnostika nesuladu y_diff <-> level_series


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

        if reconstruction == "onestep":
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

        elif reconstruction == "cumulative":
            predicted_level = float(previous_predicted_level + pred_diff)
            previous_predicted_level = predicted_level

        else:
            raise ValueError(
                "reconstruction musi byt 'onestep' alebo 'cumulative'."
            )

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
    print(f"RMSE (level, {reconstruction} rekonstrukcia): {rmse_level:.4f}")
    print(f"MAPE (level, WMAPE = sum|chyba| / sum|skutocnost|): {mape_level:.2f}%")
    print(f"ME (level): {me_level:.4f}")

    if reconstruction == "onestep" and misaligned_dates:
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