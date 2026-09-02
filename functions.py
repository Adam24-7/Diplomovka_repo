# functions.py
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import pmdarima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from xgboost import XGBRegressor

def adf_test(df, alpha=0.05):
    """
    executes Augmented Dickey–Fuller test (stacionarity test)
    df: dataframe
    """
    not_st = []
    for col in df.select_dtypes(include='number').columns:
        y = df[col]
        res = adfuller(y.dropna())
        print(f"\n--- {col} ---")
        print(f"ADF stat: {res[0]:.4f}")
        print(f"p-val: {res[1]:.4f}")
        if res[1] < alpha:
            print("Zam. H0: stacionarny")
        else:
            print("Nezam. H0: nestacionarny")
            not_st.append(col)
    return not_st


def train_test_split_exog(df, y_col, exog_cols=None, train_ratio=0.8):
    """
    y_col     : single target column (str)
    exog_cols : list of exogenous columns (list of str), if None uses all except y_col
    """
    train_size = int(len(df) * train_ratio)

    train = df[y_col].iloc[:train_size]
    test  = df[y_col].iloc[train_size:]

    if exog_cols is None:
        exog_cols = df.columns.difference([y_col])

    train_exog = df[exog_cols].iloc[:train_size]
    test_exog  = df[exog_cols].iloc[train_size:]

    return train, test, train_exog, test_exog


def evaluate_forecast(model, test, test_exog, last_train_value, original=True):
    """
    Vyhodnotenie predikcie ARIMA/ARIMAX modelu.
    
    Parametre
    ----------
    model            : fitovany model (statsmodels alebo pmdarima)
    test             : pd.Series — skutocne hodnoty (diferencie alebo urovne)
    test_exog        : pd.DataFrame — exogénne premenné pre testovacie obdobie
    last_train_value : float — posledna znama UROVNOVA hodnota PRED testovacim obdobim
    original         : bool — ak True, rekonstruuje urovne z diferencii cez last_train_value
    """
    import pmdarima

    # Detekcia typu modelu
    is_pmdarima = isinstance(model, pmdarima.arima.ARIMA)

    if is_pmdarima:
        forecast_vals = model.predict(n_periods=len(test), X=test_exog.values)
        forecast = pd.Series(forecast_vals, index=test.index)
    else:
        forecast = model.forecast(steps=len(test), exog=test_exog)
        forecast.index = test.index

    if not original:
        # Vyhodnotenie priamo na diferenciach
        rmse = root_mean_squared_error(test, forecast)
        mape = mean_absolute_percentage_error(test, forecast) * 100
        me = np.mean(test.values - forecast.values)
    else:
        # Rekonstrukcia urovni z diferencii
        y_pred_original = last_train_value + np.cumsum(forecast)
        y_test_original = last_train_value + np.cumsum(test)

        rmse = root_mean_squared_error(y_test_original, y_pred_original)
        mape = mean_absolute_percentage_error(y_test_original, y_pred_original) * 100
        me = np.mean(y_test_original.values - y_pred_original.values)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"ME:   {me:.4f}")
    
    return forecast

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
        # Tré´´ning na vsetkych dostupnych datach do t-1 (vratane)
        train_end = t  # data_diff.iloc[:t] = indexy 0..t-1
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


def plot_forecast(train, test, forecast, title="ARIMA Forecast", zoom_from=None):
    """
    Plot's the forecast plot
    """
    plt.figure(figsize=(12, 5))
    plt.plot(train.index, train, label="Train", color="steelblue")
    plt.plot(test.index, test, label="Test", color="orange")
    plt.plot(forecast.index, forecast, label="Predikcia", color="green", alpha=0.7, linestyle="--")

    if zoom_from is not None:
        plt.xlim(left=pd.Timestamp(zoom_from))

    plt.title(title)
    plt.xlabel("Dátum")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()





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



def make_features(data, target, exog_cols, n_lags=1):
    feat = pd.DataFrame(index=data.index)
    for lag in range(1, n_lags + 1):
        feat[f"{target}_lag{lag}"] = data[target].shift(lag)
    for col in exog_cols:
        feat[f"{col}_lag1"] = data[col].shift(1)
        feat[col] = data[col]  
    feat["month"] = data.index.month
    feat["quarter"] = data.index.quarter
    feat["trend"] = np.arange(len(data))
    feat[target] = data[target]
    return feat.dropna()

def rolling_forecast_xgb(X, y, initial_train_size, horizon=1, refit_every=1, objective = "reg:squarederror",
                          xgb_params=None):
    """
    Walk-forward validacia: model sa periodicky prerefituje (refit_every mesiacov)
    a generuje jednokrokove (h=1) predikcie, rovnako ako pri rolling ARIMAX evaluacii.
    """
    if xgb_params is None:
        xgb_params = dict(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

    preds, actuals, dates = [], [], []
    n = len(y)
    model = None

    for t in range(initial_train_size, n):
        if model is None or (t - initial_train_size) % refit_every == 0:
            model = XGBRegressor(**xgb_params)
            model.fit(X.iloc[:t], y.iloc[:t])

        X_step = X.iloc[[t]]
        y_hat = model.predict(X_step)[0]

        preds.append(y_hat)
        actuals.append(y.iloc[t])
        dates.append(y.index[t])

    result = pd.DataFrame({"actual": actuals, "predicted": preds}, index=dates)
    return result, model

def extract_tree_structure(model, output_csv="xgb_trees_structure.csv"):
    """
    Vrati DataFrame s kompletnou strukturou VSETKYCH stromov v ansambli.
    Kazdy riadok = jeden uzol (vnutorny alebo listovy) v jednom strome.
    """
    trees_df = model.get_booster().trees_to_dataframe()
    trees_df.to_csv(output_csv, index=False)
    return trees_df

def trace_path_for_observation(trees_df, tree_index):
    """Vypise vsetky uzly a podmienky pre dany strom (0 = koren)."""
    path = trees_df[(trees_df["Tree"] == tree_index)]
    return path[["Node", "Feature", "Split", "Yes", "No", "Gain", "Cover"]]