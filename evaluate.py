import numpy as np
import argparse
import os
import pickle
import pandas as pd
from scipy import stats

from crc import eval_controller

setup_to_table_name = {
    "airfoil": "Airfoil",
    "load_pos": "Load Positioning",
    "pendulum": "Furuta Pendulum",
    "battery": "DC Microgrids",
}

label_to_table_name = {
    "random-critical": "Random Critical",
    "random-olmss_weak": "Random OL MSS (Weak)",
    "random-olmsus": "Random OL MSUS",
    "rowcol-critical": "Row-Col Critical",
    "rowcol-olmss_weak": "Row-Col OL MSS (Weak)",
    "rowcol-olmsus": "Row-Col OL MSUS",
    "crc": "CPC",
    "mult_alg1": "Shared Lyapunov",
    "mult_alg2": "Auxiliary Stabilizer",
    "hinf_0.5": r"$\mathcal{H}_{\infty}(0.5)$",
    "hinf_1.0": r"$\mathcal{H}_{\infty}(1.0)$",
}

def populate_results(cfg, controller_trials, raw_setup, optimal_values_df, percent_filter_df):
    controller_evals = []
    for trial_idx in list(range(len(controller_trials))):
        controllers = controller_trials[trial_idx]
        evals = {}
        for controller in controllers:
            if controllers[controller] is None:
                evals[controller] = np.inf
            else:
                C = cfg["test_C"][trial_idx]
                if isinstance(controllers[controller], tuple):
                    evals[controller] = eval_controller(C, controllers[controller][0])
                else:
                    evals[controller] = eval_controller(C, controllers[controller])
        controller_evals.append(evals)
    
    controller_algs = sorted(list(controller_evals[0].keys()))

    results_per_alg = {}
    for controller_alg in controller_algs:
        results_per_alg[controller_alg] = []
        for controller_eval in controller_evals:
            results_per_alg[controller_alg].append(controller_eval[controller_alg])
        results_per_alg[controller_alg] = np.array(results_per_alg[controller_alg])

    setup = setup_to_table_name[raw_setup]
    optimal_values_df[setup] = ""
    percent_filter_df[setup] = ""

    for controller_alg in controller_algs:
        if controller_alg not in label_to_table_name: # only interested in reporting robust baselines here
            continue

        filtered_nominal = results_per_alg["optimal"][np.isfinite(results_per_alg[controller_alg])]
        filtered_result  = results_per_alg[controller_alg][np.isfinite(results_per_alg[controller_alg])]
        percent_inf      = 1 - len(filtered_result) / len(results_per_alg[controller_alg])

        subopt_gaps      = (filtered_result - filtered_nominal) / filtered_nominal
        subopt_gaps = subopt_gaps[np.where(subopt_gaps > 0)]
        alg_table_name   = label_to_table_name[controller_alg]
        optimal_values_df[setup][alg_table_name] = f"{np.around(np.median(subopt_gaps), 3)} ({np.around(stats.median_abs_deviation(subopt_gaps), 4)})"
        percent_filter_df[setup][alg_table_name] = f"{np.around(percent_inf, 3)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setups")
    args = parser.parse_args()
    setups = args.setups.split(",")

    optimal_values_df, percent_filter_df = None, None
    for setup in setups:
        with open(os.path.join(f"experiments", f"{setup}.pkl"), "rb") as f:
            cfg = pickle.load(f)

        with open(os.path.join(f"results", f"{setup}.pkl"), "rb") as f:
            controller_trials = pickle.load(f)

        controller_names = list(label_to_table_name.values())
        if optimal_values_df is None:
            optimal_values_df, percent_filter_df = pd.DataFrame(index=controller_names), pd.DataFrame(index=controller_names)
        populate_results(cfg, controller_trials, setup, optimal_values_df, percent_filter_df)

    print(optimal_values_df.to_latex())
    print(percent_filter_df.to_latex())