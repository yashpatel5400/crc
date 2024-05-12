import numpy as np
import argparse
import pickle

from crc import eval_controller

def print_results(setup):
    with open(f"experiments/{setup}_controllers.pkl", "rb") as f:
        ctrls = pickle.load(f)

    with open(f"experiments/{setup}.pkl", "rb") as f:
        cfg = pickle.load(f)

    controller_evals = []
    for ctrl_idx in list(range(len(ctrls))):
        controllers = ctrls[ctrl_idx]
        evals = {}
        for controller in controllers:
            if controllers[controller] is None:
                evals[controller] = np.inf
            else:
                C = cfg["test_C"][ctrl_idx]
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
        optimal_values  += f" & {np.around(np.nanmean(subopt_gaps), 3)} ({np.around(np.std(subopt_gaps), 3)})"
        percent_filter  += f" & {np.around(percent_inf, 3)}"

    print(optimal_values)
    print(percent_filter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup")
    args = parser.parse_args()
    setup = args.setup
    print_results(setup)