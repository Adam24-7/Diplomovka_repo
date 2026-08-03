# functions.py
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import pmdarima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from dateutil.easter import easter

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


def sk_fixed_holidays(year): #check what in 2025 that some holidays were removed
    fixed = [
        (1, 1), (1, 6), (5, 1), (5, 8), (7, 5),
        (8, 29), (9, 1), (9, 15), (11, 1), (11, 17),
        (12, 24), (12, 25), (12, 26),
    ]
    return {pd.Timestamp(year=year, month=m, day=d).date() for m, d in fixed}



'''
def calendar_adjust(
    df: pd.DataFrame,
    output_csv: str = "ip_calendar_adjusted.csv",
    value_col: str = "value",
    date_col: str = "date",
    add_trend: bool = True,
    log_transform: bool = True,
    print_summary: bool = True,
) -> tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Loads monthly time series, constructs calendar regressors for Slovakia,
    estimates the calendar component using OLS, and returns the 
    calendar-adjusted series.

    Parameters
    ----------
    df            : input DataFrame
    output_csv    : path to the output CSV file (None = do not save)
    value_col     : name of the column containing the values
    date_col      : name of the column containing the dates
    add_trend     : whether to include a linear trend in the regression
    log_transform : True for level series (positive values), 
                    False for percentage changes or series with negative values
    print_summary : whether to print the OLS regression summary

    Returns
    -------
    (df_out, model) – adjusted DataFrame and the fitted OLS model
    """
    # 1) Príprava
    df = df.copy()
    if df.index.name == date_col:
        df = df.reset_index()
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period("M")
    df = df.sort_values(date_col).reset_index(drop=True)

    # 2) Kalendárne regresory
    rows = []
    for p in df[date_col]:
        start = p.to_timestamp(how="start")
        end = p.to_timestamp(how="end").normalize()
        days = pd.date_range(start, end, freq="D")

        fixed_h = sk_fixed_holidays(start.year)

        working_days = sum(
            (d.weekday() < 5) and (d.date() not in fixed_h)
            for d in days
        )

        e = pd.Timestamp(easter(start.year))
        good_friday = (e - pd.Timedelta(days=2)).date()
        easter_monday = (e + pd.Timedelta(days=1)).date()

        easter_nonworkdays = sum([
            int(start.month == good_friday.month),
            int(start.month == easter_monday.month),
        ])

        leap_feb = int((start.month == 2) and start.is_leap_year)

        rows.append({
            date_col: p,
            "working_days": working_days,
            "easter_nonworkdays": easter_nonworkdays,
            "leap_feb": leap_feb,
        })

    X = pd.DataFrame(rows)

    # 3) Centrovanie
    cal_cols = ["working_days", "easter_nonworkdays", "leap_feb"]
    for c in cal_cols:
        X[c] = X[c] - X[c].mean()

    if add_trend:
        X["trend"] = np.arange(len(X))

    X = sm.add_constant(X)

    # 4) OLS — s logom alebo bez
    y = np.log(df[value_col].astype(float)) if log_transform else df[value_col].astype(float)
    model = sm.OLS(y, X.drop(columns=[date_col])).fit() # type: ignore[reportAttributeAccessIssue]

    if print_summary:
        print(model.summary())

    # 5) Kalendárna korekcia
    calendar_part = (
        model.params["working_days"] * X["working_days"].values # type: ignore[reportAttributeAccessIssue]
        + model.params["easter_nonworkdays"] * X["easter_nonworkdays"].values # type: ignore[reportAttributeAccessIssue]
        + model.params["leap_feb"] * X["leap_feb"].values # type: ignore[reportAttributeAccessIssue]
    )

    if log_transform:
        df["value_ca"] = np.exp(y.values - calendar_part) # type: ignore[reportAttributeAccessIssue]
    else:
        df["value_ca"] = y.values - calendar_part # type: ignore[reportAttributeAccessIssue]

    df["mom_ca_pct"] = 100 * (df["value_ca"] / df["value_ca"].shift(1) - 1)
    df["yoy_ca_pct"] = 100 * (df["value_ca"] / df["value_ca"].shift(12) - 1)

    # 6) Výstup
    for c in cal_cols:
        df[c] = X[c].values # type: ignore[reportAttributeAccessIssue]

    out = df.copy()

    if output_csv:
        out[date_col] = out[date_col].astype(str)
        out.to_csv(output_csv, index=False)
        out[date_col] = pd.to_datetime(out[date_col])

    out = out.set_index(date_col)

    return out, model

    '''