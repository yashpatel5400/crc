import numpy as np
import argparse
import os
import pickle
from scipy import stats

from crc import eval_controller

def print_results(cfg, controller_trials):
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
    
    controller_algs = list(controller_evals[0].keys())

    results_per_alg = {}
    for controller_alg in controller_algs:
        results_per_alg[controller_alg] = []
        for controller_eval in controller_evals:
            results_per_alg[controller_alg].append(controller_eval[controller_alg])
        results_per_alg[controller_alg] = np.array(results_per_alg[controller_alg])

    optimal_values = ""
    percent_filter = ""
    for controller_alg in controller_algs[2:]:
        filtered_nominal = results_per_alg["optimal"][np.isfinite(results_per_alg[controller_alg])]
        filtered_result  = results_per_alg[controller_alg][np.isfinite(results_per_alg[controller_alg])]
        percent_inf      = 1 - len(filtered_result) / len(results_per_alg[controller_alg])

        subopt_gaps      = (filtered_result - filtered_nominal) / filtered_nominal
        optimal_values  += f" & {np.around(np.median(subopt_gaps), 3)} ({np.around(stats.median_abs_deviation(subopt_gaps), 4)})"
        percent_filter  += f" & {np.around(percent_inf, 3)}"

    print(" | ".join(controller_algs[2:]))
    print(optimal_values)
    print(percent_filter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup")
    args = parser.parse_args()
    setup = args.setup

    with open(os.path.join(f"experiments", f"{setup}.pkl"), "rb") as f:
        cfg = pickle.load(f)

    with open(os.path.join(f"results", f"{setup}.pkl"), "rb") as f:
        controller_trials = pickle.load(f)

    print_results(cfg, controller_trials)