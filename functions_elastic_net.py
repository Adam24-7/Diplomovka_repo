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


def rolling_elastic_net(df_final, y_col, exog_cols, test_size = 24, cv_splits = 5, refit_every = 1, alphas = None, l1_ratios = None, verbose = True, random_state = 24):

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
    models = []
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
        "actual": actuals,
        "elastic_net_pred": predictions,
    }).set_index("date")

    coef_history = pd.concat(coef_rows, ignore_index=True)

    return results_df, models, coef_history


def evaluate_forecast(results_df, actual_col = "actual", pred_col = "elastic_net_pred"):
    y_true = results_df[actual_col]
    y_pred = results_df[pred_col]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    me = np.mean(y_true.values - y_pred.values)

    nonzero_mask = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100

    return pd.DataFrame({
        "Model": ["Elastic Net"],
        "RMSE": [rmse],
        "ME": [me],
        "MAPE": [mape]
    })