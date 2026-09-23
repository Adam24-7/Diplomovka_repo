import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scienceplots
from itertools import cycle
import numpy as np
import pandas as pd


def _rolling_forecast_core(X, y_diff, level_series, initial_train_size,
                            fit_fn, predict_fn, refit_every=1, window=6, verbose=True):
    if isinstance(level_series, pd.DataFrame):
        level_series = level_series.iloc[:, 0]

    if not X.index.equals(y_diff.index):
        raise ValueError("Index X a y_diff nie je zarovnany.")
    if y_diff.index.difference(level_series.index).size > 0:
        raise ValueError("level_series neobsahuje vsetky datumy z y_diff.")
    if X.isna().any().any() or y_diff.isna().any():
        raise ValueError("X alebo y_diff obsahuje chybajuce hodnoty (NaN).")
    if initial_train_size <= 0 or initial_train_size >= len(y_diff):
        raise ValueError("initial_train_size musi byt medzi 1 a len(y_diff)-1.")

    records, refit_diagnostics, step_diagnostics = [], [], []
    models_by_step, misaligned_dates = {}, []
    model, refit_date = None, None

    y_vals = y_diff.to_numpy(dtype=float)
    lvl_vals = level_series.to_numpy(dtype=float)

    for t in range(initial_train_size, len(y_diff)):
        current_date = y_diff.index[t]

        if model is None or (t - initial_train_size) % refit_every == 0:
            try:
                model, refit_diag = fit_fn(X.iloc[:t], y_diff.iloc[:t], current_date)
            except Exception as exc:
                raise RuntimeError(f"Trenovanie zlyhalo pre {current_date}: {exc}") from exc
            models_by_step[current_date] = model
            refit_date = current_date
            refit_diag = dict(refit_diag or {})
            refit_diag["forecast_date"] = current_date
            refit_diagnostics.append(refit_diag)

        try:
            pred_diff, step_diag = predict_fn(model, X.iloc[[t]], current_date)
        except Exception as exc:
            raise RuntimeError(f"Predikcia zlyhala pre {current_date}: {exc}") from exc
        step_diag = dict(step_diag or {})
        step_diag["forecast_date"] = current_date
        step_diagnostics.append(step_diag)

        pred_diff = float(pred_diff)
        actual_diff = float(y_vals[t])
        pos = level_series.index.get_loc(current_date)
        actual_level = float(lvl_vals[pos])

        if pos == 0:
            raise ValueError(f"Pre {current_date} chyba predchadzajuca uroven.")
        previous_level = float(lvl_vals[pos - 1]) # type: ignore
        predicted_level = previous_level + pred_diff

        implied_diff = actual_level - previous_level
        if not np.isclose(implied_diff, actual_diff, rtol=1e-4, atol=1e-8):
            misaligned_dates.append((current_date, implied_diff, actual_diff))

        error_diff = actual_diff - pred_diff
        error_level = actual_level - predicted_level

        records.append({
            "date": current_date, "actual_diff": actual_diff, "predicted_diff": pred_diff,
            "actual_level": actual_level, "predicted_level": predicted_level,
            "abs_error_diff": abs(error_diff), "sq_error_diff": error_diff ** 2,
            "abs_error_level": abs(error_level), "sq_error_level": error_level ** 2,
            "pct_error_level": abs(error_level / actual_level) * 100 if actual_level != 0 else np.nan,
            "error_diff": error_diff, "error_level": error_level,
            "refit": refit_date == current_date, "model_refit_date": refit_date,
        })

    results = pd.DataFrame(records).set_index("date")
    results["rolling_rmse_level"] = np.sqrt(results["sq_error_level"].expanding().mean())
    results["rolling_mape_level"] = (results["abs_error_level"].expanding().sum()
                                      / results["actual_level"].abs().expanding().sum()) * 100
    results[f"window{window}_rmse_level"] = np.sqrt(
        results["sq_error_level"].rolling(window=window, min_periods=1).mean())

    summary = {
        "rmse_diff": float(np.sqrt(results["sq_error_diff"].mean())),
        "rmse_level": float(np.sqrt(results["sq_error_level"].mean())),
        "wmape_level": float(results["abs_error_level"].sum() / results["actual_level"].abs().sum() * 100),
        "me_level": float(results["error_level"].mean()),
        "n_obs": int(len(results)), "n_refits": int(results["refit"].sum()),
    }

    if verbose:
        print(f"RMSE (diferencia):      {summary['rmse_diff']:.4f}")
        print(f"RMSE (level, onestep):  {summary['rmse_level']:.4f}")
        print(f"WMAPE (level):          {summary['wmape_level']:.2f}%")
        print(f"ME (level):             {summary['me_level']:.4f}")
        print(f"Refity:                 {summary['n_refits']} / {summary['n_obs']}")
        if misaligned_dates:
            d, imp, act = misaligned_dates[0]
            print(f"\n[UPOZORNENIE] {len(misaligned_dates)} bodov nesedi s level_series "
                  f"(napr. {d}: implied={imp:.4f} vs actual={act:.4f}).")

    return {
        "results": results, "models_by_step": models_by_step,
        "refit_diagnostics": refit_diagnostics, "step_diagnostics": step_diagnostics,
        "misaligned_dates": misaligned_dates, "summary": summary,
    }


def forecast_plot(
    
    X_axis,
    Y_axis,
    titles=None,
    labels=None,
    colors=None,
    linestyles=None,
    linewidths=None,
    alphas=None,
    fig_size=[13, 4],
    num_rows=1,
    num_cols=1,
    curves_per_plot=2,
):

    plt.style.use("default")

    n = len(X_axis)
    n_subplots = num_rows * num_cols

    if isinstance(curves_per_plot, int):
        curves_per_plot_list = [curves_per_plot] * n_subplots
    else:
        curves_per_plot_list = list(curves_per_plot)
        if len(curves_per_plot_list) < n_subplots:
            curves_per_plot_list += [curves_per_plot_list[-1]] * (
                n_subplots - len(curves_per_plot_list)
            )

    default_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                       "#ff7f0e", "#8c564b", "#17becf", "#e377c2"]
    default_linestyles = ["-", "--", "-.", ":"]

    colors = list(colors) if colors else []
    linestyles = list(linestyles) if linestyles else []
    linewidths = list(linewidths) if linewidths else []
    alphas = list(alphas) if alphas else []
    labels = list(labels) if labels else []

    color_cycle = cycle(colors if colors else default_colors)
    linestyle_cycle = cycle(linestyles if linestyles else default_linestyles)
    linewidth_cycle = cycle(linewidths if linewidths else [2.0])
    alpha_cycle = cycle(alphas if alphas else [1.0])

    colors_full = [next(color_cycle) for _ in range(n)]
    linestyles_full = [next(linestyle_cycle) for _ in range(n)]
    linewidths_full = [next(linewidth_cycle) for _ in range(n)]
    alphas_full = [next(alpha_cycle) for _ in range(n)]
    labels_full = labels + [f"Séria {k+1}" for k in range(len(labels), n)]

    if not titles:
        titles = []

    fig, ax = plt.subplots(
        num_rows,
        num_cols,
        squeeze=False,
        figsize=(fig_size[0], fig_size[1]),
        dpi=120,
    )

    ix = 0
    for i in range(num_rows):
        for j in range(num_cols):
            current_ax = ax[i, j]
            subplot_idx = i * num_cols + j
            n_curves_here = curves_per_plot_list[subplot_idx]

            if ix < n:
                any_plotted = False
                for k in range(n_curves_here):
                    if ix + k >= n:
                        break
                    current_ax.plot(
                        X_axis[ix + k], Y_axis[ix + k],
                        color=colors_full[ix + k],
                        linewidth=linewidths_full[ix + k],
                        linestyle=linestyles_full[ix + k],
                        alpha=alphas_full[ix + k],
                        label=labels_full[ix + k],
                    )
                    any_plotted = True

                if any_plotted:
                    current_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%y"))
                    current_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    plt.setp(current_ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

                    current_ax.grid(True, linestyle=":", alpha=0.6)
                    current_ax.spines["top"].set_visible(False)
                    current_ax.spines["right"].set_visible(False)
                    current_ax.legend(frameon=True, facecolor="white", framealpha=0.9,
                                       edgecolor="#d3d3d3", fontsize=9)

                    if subplot_idx < len(titles):
                        current_ax.set_title(titles[subplot_idx], fontsize=11, fontweight="bold")
                else:
                    current_ax.axis("off")
            else:
                current_ax.axis("off")

            ix += n_curves_here

    plt.tight_layout()
    plt.show()
    return fig