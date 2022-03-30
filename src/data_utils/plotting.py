from matplotlib import pyplot as plt
import os
import numpy as np
import seaborn as sns


def plot_1_delta(
    pd_boxplot, save_dir, eta, fsize=16, fig_xsize=10, fig_ysize=10, save=True
):
    N = int(max(pd_boxplot.N))
    # fsize = 16
    # fig_xsize = 10
    # fig_ysize = 10
    xlims = [
        0,
        N,
    ]  # further will be refined accoridng to the last method reached \hat{\delta} = 1.0
    N_reached_1 = []
    plt.figure(figsize=(fig_xsize, fig_ysize))
    names = pd_boxplot["Method"].unique()
    # fsize = 16
    if eta != 0.05:
        figure_path_1_beta = os.path.join(
            save_dir,
            "figures",
            "1_beta_N_" + str(N) + "_eta_" + str(np.round(eta, 2)) + ".png",
        )
    else:
        figure_path_1_beta = os.path.join(
            save_dir,
            "figures",
            "1_beta_N_" + str(N) + ".png",
        )
    for i in range(len(names)):
        pdSeries_tmp = pd_boxplot.loc[
            (pd_boxplot["Method"] == names[i]) & (pd_boxplot["N"] > 2)
        ]
        pdSeries_tmp.loc[:, r"$(\hat{\mathbb{P}}_N)_l$"] = (
            pdSeries_tmp[r"$(\hat{\mathbb{P}}_N)_l$"] > 1 - eta
        ).values
        pdSeries_tmp = pdSeries_tmp.groupby("N").mean()
        x_plot = pdSeries_tmp.index
        y_plot = pdSeries_tmp[r"$(\hat{\mathbb{P}}_N)_l$"].values
        # find N that yielded 1.0 delta
        idxs_reached_1 = np.where(y_plot >= 1 - 1e-9)[0]
        if len(idxs_reached_1) == 0:
            N_reached_1.append(N)
        else:
            N_reached_1.append(x_plot.values[idxs_reached_1[0]])
        plt.plot(x_plot, y_plot, label=names[i], alpha=0.5, linewidth=2.5)
        # plt.plot(np.array(ks)[1:] * N0, scenario_prob_esimate[i, 1:], label=names[i])
    plt.xlabel("N", fontsize=fsize)
    plt.ylabel(r"$1 - \hat{\delta}$", fontsize=fsize)
    plt.grid()
    xlims[-1] = np.max(N_reached_1) + 20
    plt.xlim(xlims)
    plt.legend(prop={"size": fsize}, loc="lower right")
    if save:
        try:
            plt.savefig(figure_path_1_beta)
        except FileNotFoundError:
            os.makedirs(os.path.join(save_dir, "figures"))
            plt.savefig(figure_path_1_beta)
        print("Saved to ", figure_path_1_beta)
    return xlims


def plot_boxplot(
    pd_boxplot, save_dir, eta, fsize=16, fig_xsize=10, fig_ysize=10, save=True
):
    N = int(max(pd_boxplot.N))
    # fsize = 16
    # fig_xsize = 10
    # fig_ysize = 5
    if eta != 0.05:
        figure_path_box = os.path.join(
            save_dir,
            "figures",
            "boxplot_J_N_" + str(N) + "_eta_" + str(np.round(eta, 2)) + ".png",
        )
    else:
        figure_path_box = os.path.join(
            save_dir,
            "figures",
            "boxplot_J_N_" + str(N) + ".png",
        )
    plt.figure(figsize=(fig_xsize, fig_ysize))
    ax = sns.boxplot(
        x="N",
        y=r"$(\hat{\mathbb{P}}_N)_l$",
        hue="Method",
        data=pd_boxplot[pd_boxplot["N"] > 6],
        palette="Set3",
    )
    ax.axhline(
        1 - eta,
        0,
        1,
        label=r"$1 - \eta$",
        color="black",
        linewidth=2,
        alpha=0.7,
        linestyle="dotted",
    )
    plt.ylim((1 - 2 * eta, 1.0))
    plt.legend(prop={"size": fsize}, loc="lower right")
    if save:
        try:
            plt.savefig(figure_path_box)
        except FileNotFoundError:
            os.makedirs(os.path.join(save_dir, "figures"))
            plt.savefig(figure_path_box)
        print("Saved to ", figure_path_box)


def plot_grids(pds, save_dir, eta, include_O=True, truncate_names=True):
    for grid_name in pds.keys():
        pd_boxplot = pds[grid_name]
        if truncate_names:
            method_names = pd_boxplot.Method.unique()
            pd_boxplot.Method = pd_boxplot.Method.map(
                {mn: mn.split("-")[0] for mn in method_names}
            )
        fig_xsize = 10
        fig_ysize = 5
        fsize = 20
        if include_O:
            no_SAO = pd_boxplot
            no_SAO["N"] = no_SAO["N"].astype(int)
        else:
            if truncate_names:
                no_SAO = pd_boxplot.drop(
                    pd_boxplot[pd_boxplot["Method"] == "SAO"].index
                )
            else:
                no_SAO = pd_boxplot.drop(
                    pd_boxplot[pd_boxplot["Method"] == "SAO-ScenarioApproxWithO"].index
                )
            no_SAO["N"] = no_SAO["N"].astype(int)
        save_dir_grid30 = os.path.join(save_dir, grid_name)
        xlims = plot_1_delta(
            no_SAO,
            save_dir_grid30,
            eta,
            save=True,
            fig_xsize=fig_xsize,
            fig_ysize=fig_ysize,
            fsize=fsize,
        )

        pd_boxplot = pds[grid_name]
        fig_xsize = 10
        fig_ysize = 5
        fsize = 15
        no_SAO_lim = no_SAO.drop(no_SAO[no_SAO["N"] >= xlims[-1]].index)
        save_dir_grid30 = os.path.join(save_dir, grid_name)
        plot_boxplot(
            no_SAO_lim,
            save_dir_grid30,
            eta,
            save=True,
            fig_xsize=fig_xsize,
            fig_ysize=fig_ysize,
            fsize=fsize,
        )
