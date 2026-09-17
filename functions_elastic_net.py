import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def fit_elastic_net(train_y, train_exog, cv_splits = 5, alphas = None, l1_ratios = None, random_state = 24):

    if alphas is None:
        alphas = np.logspace(-3, 1, 20)
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_exog)

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    param_grid = {
        "alpha": alphas,
        "l1_ratio": l1_ratios,
    }

    base_model = ElasticNet(
        max_iter= 20000,
        random_state= random_state
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv = tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1
    )

    grid_search.fit(X_scaled, train_y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    cv_results = pd.DataFrame(grid_search.cv_results_)[["param_alpha", "param_l1_ratio","mean_test_score", "std_test_score"]].sort_values("mean_test_score", ascending=False)

    return best_model, scaler, best_params, cv_results


def predict_elastic_net(model, scaler, X_new):
    X_scaled = scaler.transform(X_new)
    return model.predict(X_scaled)


def get_coef(model, feature_names):
    coefs = pd.DataFrame({
        "feature": feature_names,
        "coef": model.coef_,
        "abs_coef": np.abs(model.coef_),
    }).sort_values("abs_coef", ascending=False)

    n_selected = (coefs["coef"] != 0).sum()
    coefs["selected"] = coefs["coef"] != 0

    return coefs, n_selected


def rolling_elastic_net_old(df_final, y_col, exog_cols, test_size = 24, cv_splits = 5, refit_every = 1, alphas = None, l1_ratios = None, verbose = True, random_state = 24):

    data = df_final[[y_col] + exog_cols].dropna().copy()

    n_total = len(data)
    train_size = n_total - test_size

    if train_size <= cv_splits + 5:
        raise ValueError(
            f"Prilis malo trenovacich pozorovnani ({train_size}) pre {cv_splits}-fold TimeSeriesSplit."
        )

    predictions = []
    actuals = []
    dates = []
    models = {}
    coef_rows = []

    current_model = None
    current_scaler = None
    current_params = None

    for t in range(test_size):
        train_end = train_size + t
        train_y = data[y_col].iloc[:train_end]
        train_exog = data[exog_cols].iloc[:train_end]

        test_date = data.index[train_end]
        test_exog_row = data[exog_cols].iloc[[train_end]]
        actual_value = data[y_col].iloc[train_end]

        need_refit = (current_model is None) or (t % refit_every == 0)

        if need_refit:
            current_model, current_scaler, current_params, _ = fit_elastic_net(train_y=train_y,
                                                                               train_exog= train_exog, 
                                                                               cv_splits=cv_splits,
                                                                               alphas=alphas,
                                                                               l1_ratios=l1_ratios,
                                                                               random_state=random_state,
                                                                               )
            if verbose:
                print(f"  [{test_date.date()}] refit: alpha={current_params['alpha']:.4f}, l1_ratio={current_params['l1_ratio']:.2f}")

        pred_value = predict_elastic_net(current_model, current_scaler, test_exog_row)[0]

        predictions.append(pred_value)
        actuals.append(actual_value)
        dates.append(test_date)
        models[test_date] = (current_model, current_scaler)

        coefs, n_selected = get_coef(current_model, exog_cols)
        coefs["forecast_date"] = test_date
        coefs["n_selected"] = n_selected
        coef_rows.append(coefs)


    results_df = pd.DataFrame({
        "date": dates,
        "actual_diff": actuals,
        "predicted_diff": predictions,
    }).set_index("date")

    coef_history = pd.concat(coef_rows, ignore_index=True)

    return results_df, models, coef_history

from typing import cast

def rolling_forecast_elastic_net(
    X: pd.DataFrame,
    y_diff: pd.Series,
    level_series: pd.Series | pd.DataFrame,
    initial_train_size: int,
    last_train_value: float,
    horizon: int = 1,
    refit_every: int = 1,
    elnet_params: dict | None = None,
    reconstruction: str = "onestep",
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Rolling one-step ElasticNet forecast s rekonštrukciou na pôvodnú úroveň,
    identickou štruktúrou výstupov ako XGBoost a uchovávaním coef_history.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matica zarovnaná s y_diff.
    y_diff : pd.Series
        Cieľová premenná (diferencia).
    level_series : pd.Series | pd.DataFrame
        Pôvodná nediferencovaná úroveň.
    initial_train_size : int
        Počet pozorovaní v počiatočnom tréningovom okne.
    last_train_value : float
        Posledná známa úroveň pred prvým testovacím bodom.
    horizon : int
        Podporovaný iba horizon=1.
    refit_every : int
        Refit modelu po každých n krokoch.
    elnet_params : dict
        Hyperparametre pre ElasticNet.
    reconstruction : str
        "onestep" alebo "cumulative".

    Returns
    -------
    results : pd.DataFrame
        Tabuľka s predikciami diferencie aj levelu a rolling metrikami.
    models_by_step : dict
        Uložené modely pre jednotlivé refit dátumy.
    coef_history : pd.DataFrame
        História absolútnych koeficientov a interceptu pri každom refite.
    """
    if horizon != 1:
        raise ValueError("Tato funkcia podporuje iba horizon=1.")

    if elnet_params is None:
        elnet_params = {
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "random_state": 42,
            "max_iter": 2000,
        }
    else:
        elnet_params = elnet_params.copy()

    # ------------------------------------------------------------------
    # Kontroly vstupov
    # ------------------------------------------------------------------
    if not isinstance(y_diff, pd.Series):
        raise TypeError("y_diff musi byt pandas Series.")

    if not y_diff.index.is_unique:
        raise ValueError("y_diff obsahuje duplicitne datumy v indexe.")

    if isinstance(level_series, pd.DataFrame):
        if level_series.shape[1] != 1:
            raise ValueError("level_series musi mat presne jeden stlpec.")
        level_series = level_series.iloc[:, 0]

    if not isinstance(level_series, pd.Series):
        raise TypeError("level_series musi byt pandas Series.")

    if not level_series.index.is_unique:
        raise ValueError("level_series obsahuje duplicitne datumy v indexe.")

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
        raise ValueError(f"Chybajuce datumy v level_series: {missing_dates.tolist()}")

    # ------------------------------------------------------------------
    # Rolling one-step cyklus
    # ------------------------------------------------------------------
    records = []
    coef_records = []
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
            model = ElasticNet(**elnet_params)
            model.fit(X.iloc[:t], y_diff.iloc[:t])

            models_by_step[current_date] = model
            refit_date = current_date

            # Ukladanie koeficientov pri kazdom refite
            coef_dict = {
                "date": current_date,
                "intercept": float(model.intercept_),
            }
            coef_dict.update(zip(X.columns, model.coef_))
            coef_records.append(coef_dict)

        pred_diff = float(np.asarray(model.predict(X.iloc[[t]])).ravel()[0])
        actual_diff = float(y_diff_values[t])

        current_position = cast(int, level_series.index.get_loc(current_date))
        actual_level = float(level_values[current_position])

        if reconstruction == "onestep":
            if current_position == 0:
                raise ValueError(
                    f"Pre datum {current_date} neexistuje predchadzajuca levelova hodnota."
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
            raise ValueError("reconstruction musi byt 'onestep' alebo 'cumulative'.")

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
    coef_history = (
        pd.DataFrame(coef_records).set_index("date")
        if coef_records
        else pd.DataFrame()
    )

    # ------------------------------------------------------------------
    # Priebežné (rolling) a window-based metriky
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
    results["rolling_me_level"] = results["error_level"].expanding().mean()

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
    # Agregované metriky
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
            "identita actual_level - previous_level == actual_diff. "
            f"Priklad ({first_example[0]}): implied_diff={first_example[1]:.4f} "
            f"vs actual_diff={first_example[2]:.4f}."
        )

    return results, models_by_step, coef_history


def evaluate_forecast(results: pd.DataFrame) -> pd.Series:
    """
    Vypočíta súhrnné evaluačné metriky pre diferencie aj pre úroveň (level).
    
    Parameters
    ----------
    results : pd.DataFrame
        Výstupná tabuľka z vyhodnocovacích funkcií obsahujúca
        sq_error_diff, sq_error_level, abs_error_level, actual_level, error_level.
        
    Returns
    -------
    pd.Series
        Séria s metrikami RMSE_diff, RMSE_level, WMAPE_level, MAE_level a ME_level.
    """
    rmse_diff = np.sqrt(results["sq_error_diff"].mean())
    rmse_level = np.sqrt(results["sq_error_level"].mean())
    wmape_level = (results["abs_error_level"].sum() / results["actual_level"].abs().sum()) * 100
    mae_level = results["abs_error_level"].mean()
    me_level = results["error_level"].mean()

    return pd.Series({
        "RMSE_diff": rmse_diff,
        "RMSE_level": rmse_level,
        "WMAPE_level (%)": wmape_level,
        "MAE_level": mae_level,
        "ME_level": me_level
    })