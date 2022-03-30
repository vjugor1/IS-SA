import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as optim
from scipy import stats
import cvxpy as cp
from tqdm import tqdm
import json
import os
import sys
import seaborn as sns
import pandas as pd
import multiprocessing


from src.samplers.importance_sampler import *
from src.data_utils import grid_data
from src.samplers.utils import check_feasibility_out_of_sample
from src.samplers import preprocessing as pre
from src.data_utils import synthetic as synth
from src.solvers import scenario_approx as SA
from src.solvers import utils as SU
from src.solvers import analytical_approx as AA


def map_names(
    results,
    new_names=[
        "SAO-ScenarioApproxWithO",
        "SA-ScenarioApprox",
        "SAIS-ScenarioApproxImportanceSampling",
    ],
):
    for r in results.keys():
        if r not in ["Sigma", "mu"]:
            for l in results[r]:
                keys = list(l.keys())
                for i in range(len(keys)):
                    l[new_names[i]] = l.pop(keys[i])
    return results


# Gamma, Beta = synth.regular_polyhedron(10, 6)
if __name__ == "__main__":
    grid_name = "grid30"

    (
        Gamma,
        Beta,
        gens,
        cost_coeffs,
        cost_correction_term,
        cost_dc_opf,
    ) = grid_data.get_linear_constraints(grid_name, check_pp_vs_new_form=True)
    x0 = gens[1:]
    print(Gamma, Beta)
    mu = np.zeros(len(gens) - 1)
    Sigma = np.eye(len(gens) - 1) * 0.01
    # making matrix psd
    Sigma = Sigma.dot(Sigma.T)
    # A = Gamma
    Gamma, Beta, A = pre.standartize(Gamma, Beta, mu, Sigma)
    print(len(Gamma), len(Beta))
    J = Gamma.shape[0]
    print(len(A))
    c = cost_coeffs

    eta = 0.05

    # Standartization in prior must be conducted
    # res_boole = AA.inner_polyhedral(eta, Gamma, Beta, x0=x0, c=c)
    # boole_prob = check_feasibility_out_of_sample(res_boole.x, Gamma, Beta, A, 1000000)

    # Solve scenario approximations
    # Store sigma and mu, next, the solutions for approximation will be pushed
    results = {
        "Sigma": [[float(v) for v in row] for row in Sigma],
        "mu": [float(v) for v in mu],
    }
    N0 = 10  # for 118 eta = 0.01 > 1500 at 2000 live starts for SAIMIN
    ks = list(range(1, 30))[::5]
    L = 100

    # parallel and discard useless planes and samples
    x0_dict = {"SAIMIN": x0, "SCSA": x0, "SCSA_no_O": x0}
    # import time

    for k in tqdm(ks):
        N = N0 * k
        # t1 = time.time()
        with multiprocessing.Pool() as pool:
            res_L = pool.starmap(
                SU.solve_approximations,
                [(Gamma, Beta, A, N, c, eta, x0_dict, True)] * L,
            )

        for res in res_L:
            for k in x0_dict.keys():
                x0_dict[k] = res[k][0]
            try:
                results[N].append(res)
            except KeyError:
                results[N] = []
                results[N].append(res)
        print("Finished N = ", N)
    # save the results
    save_dir = os.path.join("saves", grid_name)
    json_file = os.path.join(
        # "N_" + str(N0 * ks[-1]) + "_eta_" + str(np.round(eta, 2)) + ".json"
        "N_"
        + str(N0 * ks[-1])
        + ".json"
    )

    results = map_names(
        results,
        new_names=[
            "SAO-ScenarioApproxWithO",
            "SA-ScenarioApprox",
            "SAIS-ScenarioApproxImportanceSampling",
        ],
    )

    try:
        with open(os.path.join(save_dir, json_file), "w") as fp:
            json.dump(results, fp, indent=4)
    except FileNotFoundError:
        os.makedirs(save_dir)
        with open(os.path.join(save_dir, json_file), "w") as fp:
            json.dump(results, fp, indent=4)

    # processing results for plotting average behaviour on L different computations
    def unpack_results(results, c, k, cost_correction_term):
        try:
            names = list(results[N0 * ks[0]][k].keys())
        except KeyError:
            names = list(results[str(N0 * ks[0])][k].keys())
        fns = []
        xs = []
        for r in results.keys():
            if r not in ["Sigma", "mu"]:
                for v in results[r][k].values():
                    try:
                        xs.append(v[0])
                        fns.append(np.dot(v[0], c) + cost_correction_term)
                    except ValueError:
                        fns.append(np.nan)
        fns = np.array(fns).reshape(-1, len(names))
        xs = np.array(xs).reshape(-1, len(names), A.shape[1])
        return fns, xs, names

    fns, xs, names = unpack_results(
        results=results, c=c, k=0, cost_correction_term=cost_correction_term
    )
    scenario_prob_estimate = np.zeros((len(names), len(ks)))
    scenario_probs_several_starts = []
    for k in tqdm(range(L)):
        _, xs, names = unpack_results(
            results=results,
            c=cost_coeffs,
            cost_correction_term=cost_correction_term,
            k=k,
        )
        # boole_prob = check_feasibility_out_of_sample(
        #     res_boole.x, Gamma, Beta, A, 100000
        # )
        scenarios_probs = np.array(
            [
                np.apply_along_axis(
                    arr=xs[:, i, :],
                    func1d=lambda x: check_feasibility_out_of_sample(
                        x, Gamma, Beta, A, 100000
                    ),
                    axis=1,
                )
                for i in range(3)
            ]
        )
        scenario_prob_estimate += scenarios_probs - eta >= 0.0
        scenario_probs_several_starts.append(scenarios_probs)
    scenario_prob_esimate = scenario_prob_estimate / L
    scenario_probs_several_starts = np.array(np.stack(scenario_probs_several_starts))

    # shaping into pandas

    pd_boxplot = pd.DataFrame({"N": [], "Method": [], r"$(\hat{\mathbb{P}}_N)_l$": []})
    for method_idx in range(scenario_probs_several_starts.shape[1]):
        data = scenario_probs_several_starts[:, method_idx, :]
        pd_boxplot_tmp = pd.DataFrame(
            {"N": [], "Method": [], r"$(\hat{\mathbb{P}}_N)_l$": []}
        )
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                pd_boxplot_tmp = pd.concat(
                    [
                        pd_boxplot_tmp,
                        pd.DataFrame(
                            {
                                "N": [ks[j] * N0],
                                "Method": [names[method_idx]],
                                r"$(\hat{\mathbb{P}}_N)_l$": [data[i, j]],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
        pd_boxplot = pd.concat([pd_boxplot, pd_boxplot_tmp])
    # save to csv
    # pandas_name = "multistarts" + "_eta_" + str(np.round(eta, 2)) + ".csv"
    pandas_name = "multistarts.csv"
    pd_boxplot.to_csv(os.path.join(save_dir, pandas_name))
    # 1 - beta plot
    plt.figure(figsize=(10, 10))
    fsize = 16
    figure_path_1_beta = os.path.join(
        save_dir,
        "figures",
        "1_beta_N_" + str(N0 * ks[-1]) + "_eta_" + str(np.round(eta, 2)) + ".png",
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
        plt.plot(x_plot, y_plot, label=names[i], alpha=0.5)
        # plt.plot(np.array(ks)[1:] * N0, scenario_prob_esimate[i, 1:], label=names[i])
    plt.xlabel("N", fontsize=fsize)
    plt.ylabel(r"$1 - \hat{\delta}$", fontsize=fsize)
    plt.grid()
    plt.legend(prop={"size": fsize})
    try:
        plt.savefig(figure_path_1_beta)
    except FileNotFoundError:
        os.makedirs(os.path.join(save_dir, "figures"))
        plt.savefig(figure_path_1_beta)

    # box plots

    figure_path_box = os.path.join(
        save_dir,
        "figures",
        "boxplot_J_N_" + str(N0 * ks[-1]) + "_eta_" + str(np.round(eta, 2)) + ".png",
    )
    plt.figure(figsize=(20, 5))
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
    # plt.grid()
    try:
        plt.savefig(figure_path_box)
    except FileNotFoundError:
        os.makedirs(os.path.join(save_dir, "figures"))
        plt.savefig(figure_path_box)
