# functions.py
from __future__ import annotations
import pandas as pd
import numpy as np
import pmdarima
from statsmodels.tsa.statespace.sarimax import SARIMAX



def evaluate_forecast_rolling(data_diff, exog, initial_train_size, order,
                               last_train_value, level_series,
                               seasonal=False, m=12,
                               enforce_stationarity=True,
                               maxiter=50, method="lbfgs"):
    """
    Rolling (walk-forward) vyhodnotenie ARIMAX modelu - metodicky konzistentne
    s rolling_forecast_xgb() pre XGBoost.

    Namiesto jednorazoveho natrenovania a 24-krokovej predikcie naraz sa model
    v kazdom mesiaci natrénuje znovu na vsetkych dostupnych datach do vtedy
    a predikuje len 1 mesiac dopredu. Chyba sa neakumuluje, lebo kazdy krok
    vyuziva SKUTOCNE predchadzajuce hodnoty (nie predikovane).

    Parametre
    ----------
    data_diff : pd.Series
        Cielova premenna na diferencii (napr. pp_sa.diff()).
    exog : pd.DataFrame
        Exogenne premenné zarovnané s data_diff.
    initial_train_size : int
        Pocet pozorovani v pociatocnej trenovacej mnozine (pred zaciatkom testu).
    order : tuple
        (p,d,q) alebo (p,d,q)(P,D,Q,m) pre sezonny ARIMAX.
    last_train_value : float
        Posledna znama urovnova hodnota tesne PRED prvym testovacim bodom.
    level_series : pd.Series
        Povodny (nediferencovany) rad pp_sa, so stejnym datumovym indexom
        ako data_diff (dlhsi o 1 pozorovanie na zaciatku).
    seasonal : bool
        Ci ide o sezonny model SARIMAX (True) alebo obycajny ARIMAX (False).
    m : int
        Sezonny period (12 pre mesacné´´ data).
    enforce_stationarity : bool
        Obmedzenie na stacionaritu (odporucane pre mensie datasety).
    maxiter : int
        Maximalny pocet iteracii optimalizacie.
    method : str
        Optimalizacná´´ metoda (napr. "lbfgs", "bfgs").

    Vracia
    -------
    results : pd.DataFrame
        Indexované´´ datumom, obsahuje skutocné´´ a predikované´´ hodnoty
        na diferenciach aj urovniach, plus chybové´´ metriky.
    """
    records = []
    models_by_step = {}

    for t in range(initial_train_size, len(data_diff)):
        train_end = t
        model = pmdarima.arima.ARIMA(
            order=order,
            seasonal=seasonal,
            maxiter=maxiter,
            method=method,
            enforce_stationarity=enforce_stationarity,
            suppress_warnings=True
        )
        model.fit(data_diff.iloc[:train_end], X=exog.iloc[:train_end])
        models_by_step[data_diff.index[t]] = model

        # Predikcia 1 mesiac dopredu (jednokrokova)
        forecast_step = model.predict(
            n_periods=1,
            X=exog.iloc[[t]]
        )

        pred_diff = float(np.asarray(forecast_step).ravel()[0])
        true_diff = float(data_diff.iloc[t])
        current_date = data_diff.index[t]

        # Rekonstrukcia na uroven (onestep)
        true_level = level_series.loc[current_date]
        true_prev_level = level_series.shift(1).loc[current_date]
        pred_level = pred_diff + true_prev_level

        records.append({
            "date": current_date,
            "actual_diff": true_diff,
            "predicted_diff": pred_diff,
            "actual_level": true_level,
            "predicted_level": pred_level,
            "abs_error_diff": abs(true_diff - pred_diff),
            "sq_error_diff": (true_diff - pred_diff) ** 2,
            "abs_error_level": abs(true_level - pred_level),
            "sq_error_level": (true_level - pred_level) ** 2,
            "pct_error_level": abs((true_level - pred_level) / true_level) * 100
                                if true_level != 0 else np.nan,
        })

    results = pd.DataFrame(records).set_index("date")

    # Agregovane metriky
    rmse_diff = np.sqrt(results["sq_error_diff"].mean())
    rmse_level = np.sqrt(results["sq_error_level"].mean())
    mape_level = results["pct_error_level"].mean()
    results["actual_level"] = pd.to_numeric(results["actual_level"], errors="coerce").astype(float)
    results["predicted_level"] = pd.to_numeric(results["predicted_level"], errors="coerce").astype(float)

    me_level = (results["actual_level"] - results["predicted_level"]).mean()
    me_level = me_level.mean()

    print(f"RMSE (diferencia): {rmse_diff:.4f}")
    print(f"RMSE (level, onestep rekonstrukcia): {rmse_level:.4f}")
    print(f"MAPE (level): {mape_level:.2f}%")
    print(f"ME (level): {me_level:.4f}")

    return results, models_by_step

def forward_selection(dataset, notinc):
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



