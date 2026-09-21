import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scienceplots
from itertools import cycle


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