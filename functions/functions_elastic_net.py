import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import cast
import warnings
warnings.filterwarnings("ignore")

import functions_general as fg

def _extract_elasticnet_diagnostics(refit_diagnostics, step_diagnostics):
    coef_rows, sparsity_rows, hyper_rows = [], [], []
    for rec in refit_diagnostics:
        d = rec["forecast_date"]
        coefs = rec.get("coefficients", {})
        n_sel = rec.get("n_selected", np.nan)
        for feat, val in coefs.items():
            coef_rows.append({"forecast_date": d, "feature": feat, "coef": float(val),
                               "abs_coef": abs(float(val)), "selected": val != 0.0})
        sparsity_rows.append({"forecast_date": d, "n_selected": n_sel, "n_total": len(coefs),
                               "sparsity_ratio": 1 - n_sel / len(coefs) if coefs else np.nan})
        hyper_rows.append({"forecast_date": d, "alpha": rec.get("alpha", np.nan),
                            "l1_ratio": rec.get("l1_ratio", np.nan),
                            "intercept": rec.get("intercept", np.nan),
                            "l1_penalty_weight": rec.get("alpha", np.nan) * rec.get("l1_ratio", np.nan),
                            "l2_penalty_weight": rec.get("alpha", np.nan) * (1 - rec.get("l1_ratio", np.nan)),
                        })
    return {
        "coefficients": pd.DataFrame(coef_rows),
        "sparsity": pd.DataFrame(sparsity_rows).set_index("forecast_date") if sparsity_rows else pd.DataFrame(),
        "hyperparams": pd.DataFrame(hyper_rows).set_index("forecast_date") if hyper_rows else pd.DataFrame(),
    }

def _make_elasticnet_fit_fn(cv_splits=5, alphas=None, l1_ratios=None, random_state=24):
    alphas = alphas if alphas is not None else np.logspace(-3, 1, 20)
    l1_ratios = l1_ratios if l1_ratios is not None else [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

    def _fit(X_train, y_train, current_date):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        n_splits = min(cv_splits, max(len(X_train) // 10, 2))
        gs = GridSearchCV(ElasticNet(max_iter=20_000, random_state=random_state),
                           {"alpha": alphas, "l1_ratio": l1_ratios},
                           cv=TimeSeriesSplit(n_splits=n_splits),
                           scoring="neg_mean_squared_error", n_jobs=-1)
        gs.fit(X_scaled, y_train)
        best = gs.best_estimator_

        diag = {"coefficients": dict(zip(X_train.columns, best.coef_)),
                "intercept": float(best.intercept_),
                "n_selected": int(np.sum(best.coef_ != 0)),
                "alpha": float(gs.best_params_["alpha"]),
                "l1_ratio": float(gs.best_params_["l1_ratio"])}
        return (best, scaler), diag
    return _fit


def _elasticnet_predict_fn(model, X_row, current_date):
    best, scaler = model
    return float(np.asarray(best.predict(scaler.transform(X_row))).ravel()[0]), {}

def rolling_forecast_elastic_net(X, y_diff, level_series, initial_train_size,
                                  refit_every=1, cv_splits=5, alphas=None,
                                  l1_ratios=None, random_state=24, verbose=True):
    fit_fn = _make_elasticnet_fit_fn(cv_splits, alphas, l1_ratios, random_state)

    out = fg._rolling_forecast_core(X, y_diff, level_series, initial_train_size,
                                  fit_fn, _elasticnet_predict_fn, refit_every, verbose=verbose)
    out["diagnostics"] = _extract_elasticnet_diagnostics(out["refit_diagnostics"], out["step_diagnostics"])
    return out



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



def rolling_forecast_elastic_net_old(
    X: pd.DataFrame,
    y_diff: pd.Series,
    level_series: pd.Series | pd.DataFrame,
    initial_train_size: int,
    last_train_value: float,
    horizon: int = 1,
    refit_every: int = 1,
    cv_splits: int = 5,
    reconstruction: str = "onestep",
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:

    if horizon != 1:
        raise ValueError("Tato funkcia podporuje iba horizon=1.")

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

    records = []
    coef_records = []
    models_by_step: dict = {}
    model = None
    scaler = None
    previous_predicted_level = float(last_train_value)
    refit_date = None
    misaligned_dates: list = []
    best_params_history: list = []

    y_diff_values = y_diff.to_numpy(dtype=float)
    level_values = level_series.to_numpy(dtype=float)

    for t in range(initial_train_size, len(y_diff)):
        current_date = y_diff.index[t]

        if model is None or (t - initial_train_size) % refit_every == 0:
            model, scaler, best_params, _ = fit_elastic_net(
                train_y=y_diff.iloc[:t],
                train_exog=X.iloc[:t],
                cv_splits=cv_splits
            )
            models_by_step[current_date] = (model, scaler)
            refit_date = current_date

            best_params_history.append({
                "forecast_date": current_date,
                "alpha": best_params["alpha"],
                "l1_ratio": best_params["l1_ratio"],
            })
            
            coefs, n_selected = get_coef(model, X.columns)
            coefs["forecast_date"] = current_date
            coefs["intercept"] = float(model.intercept_)
            coefs["n_selected"] = n_selected
            coef_records.append(coefs)

        X_test_scaled = scaler.transform(X.iloc[[t]]) # type: ignore
        pred_diff = float(np.asarray(model.predict(X_test_scaled)).ravel()[0])
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
                if actual_level != 0.0 else np.nan
            ),
            "error_diff": error_diff,
            "error_level": error_level,
            "refit": refit_date == current_date,
            "model_refit_date": refit_date,
        })

    results = pd.DataFrame(records).set_index("date")
    coef_history = (
        pd.concat(coef_records, ignore_index=True)
        if coef_records else pd.DataFrame()
    )

    results["rolling_rmse_diff"] = np.sqrt(results["sq_error_diff"].expanding().mean())
    results["rolling_rmse_level"] = np.sqrt(results["sq_error_level"].expanding().mean())
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
    results["window6_me_level"] = results["error_level"].rolling(window=6, min_periods=1).mean()

    rmse_diff = float(np.sqrt(results["sq_error_diff"].mean()))
    rmse_level = float(np.sqrt(results["sq_error_level"].mean()))
    mape_level = float(results["abs_error_level"].sum() / results["actual_level"].abs().sum() * 100)
    me_level = float(results["error_level"].mean())

    print(f"RMSE (diferencia): {rmse_diff:.4f}")
    print(f"RMSE (level, {reconstruction} rekonstrukcia): {rmse_level:.4f}")
    print(f"MAPE (level, WMAPE): {mape_level:.2f}%")
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
    
    best_params_df = (
        pd.DataFrame(best_params_history).set_index("forecast_date")
        if best_params_history else pd.DataFrame()
    )


    return results, models_by_step, coef_history, best_params_df


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