import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scienceplots


def forecast_plot(
    X_axis, 
    Y_axis, 
    titles=None, 
    labels=None, 
    colors=None, 
    linestyles=None, 
    linewidths=None, 
    alphas=None,
    fig_size = [13, 4],
    num_rows=1, 
    num_cols=1
):

    plt.style.use("seaborn-v0_8-notebook")
    



    n = len(X_axis)

    # Ak je zoznam None alebo prázdny [], naplní sa predvolenými hodnotami
    if not colors: colors = ["#1f77b4", "#d62728"] * n
    if not linestyles: linestyles = ["-", "--"] * n
    if not linewidths: linewidths = [2.0] * n
    if not alphas: alphas = [1.0] * n
    if not labels: labels = [f"Séria {k+1}" for k in range(n)]
    if not titles: titles = []

    fig, ax = plt.subplots(
        num_rows, 
        num_cols, 
        squeeze=False,
        figsize=(fig_size[0],fig_size[1]),
        dpi=120
    )
    
    ix = 0
    for i in range(num_rows):
        for j in range(num_cols):
            current_ax = ax[i, j]

            if ix + 1 < n:
                # Priame indexovanie bez modulo %
                current_ax.plot(
                    X_axis[ix], Y_axis[ix], 
                    color=colors[ix], 
                    linewidth=linewidths[ix], 
                    linestyle=linestyles[ix], 
                    alpha=alphas[ix], 
                    label=labels[ix]
                )
                current_ax.plot(
                    X_axis[ix + 1], Y_axis[ix + 1], 
                    color=colors[ix + 1], 
                    linewidth=linewidths[ix + 1], 
                    linestyle=linestyles[ix + 1], 
                    alpha=alphas[ix + 1], 
                    label=labels[ix + 1]
                )

                # Formátovanie osi X (dátumy)
                current_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%y"))
                current_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(current_ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

                # Mriežka a vzhľad
                current_ax.grid(True, linestyle=":", alpha=0.6)
                current_ax.spines["top"].set_visible(False)
                current_ax.spines["right"].set_visible(False)
                current_ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="#d3d3d3", fontsize=9)

                title_idx = i * num_cols + j
                if title_idx < len(titles):
                    current_ax.set_title(titles[title_idx], fontsize=11, fontweight="bold")
            else:
                current_ax.axis("off")

            ix += 2

    plt.tight_layout()
    plt.show()