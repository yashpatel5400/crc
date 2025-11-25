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
    "fusion": "Fusion Plant",
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
    "hinf": r"$\mathcal{H}_{\infty}$",
}

def populate_results(cfg, controller_trials, raw_setup, optimal_values_df, percent_filter_df):
    controller_evals = []
    for trial_idx in range(len(controller_trials)):
        controllers = controller_trials[trial_idx][0] # output is controllers, times (just want controllers now)
        evals = {}
        for controller in controllers:
            if controllers[controller] is None:
                evals[controller] = np.inf
            else:
                C = cfg["test_C"][trial_idx]
                K = controllers[controller][0] if isinstance(controllers[controller], tuple) else controllers[controller]
                evals[controller] = eval_controller(C, K)
        controller_evals.append(evals)

    controller_algs = sorted(controller_evals[0].keys())
    results_per_alg = {alg: np.array([ce[alg] for ce in controller_evals]) for alg in controller_algs}

    setup = setup_to_table_name[raw_setup]
    optimal_values_df[setup] = ""
    percent_filter_df[setup] = ""

    for controller_alg in controller_algs:
        if controller_alg not in label_to_table_name:
            continue

        filtered_nominal = results_per_alg["optimal"][np.isfinite(results_per_alg[controller_alg])]
        filtered_result = results_per_alg[controller_alg][np.isfinite(results_per_alg[controller_alg])]
        percent_inf = 1 - len(filtered_result) / len(results_per_alg[controller_alg])

        subopt_gaps = (filtered_result - filtered_nominal) / filtered_nominal
        subopt_gaps = subopt_gaps[subopt_gaps > 0]
        alg_table_name = label_to_table_name[controller_alg]

        if len(subopt_gaps) == 0 or not np.isfinite(np.median(subopt_gaps)):
            val = "---"
        else:
            median = np.median(subopt_gaps)
            mad = stats.median_abs_deviation(subopt_gaps)
            val = f"{median:.3f} ({mad:.3f})"

        optimal_values_df.at[alg_table_name, setup] = val
        percent_filter_df.at[alg_table_name, setup] = f"{percent_inf:.3f}"

def mask_and_bold(opt_df, pct_df, threshold=0.80):
    masked_df = opt_df.copy()

    for col in masked_df.columns:
        for idx in masked_df.index:
            val = str(masked_df.at[idx, col])
            try:
                if float(pct_df.at[idx, col]) > threshold or "nan" in val.lower():
                    masked_df.at[idx, col] = "---"
            except ValueError:
                masked_df.at[idx, col] = "---"

    for col in masked_df.columns:
        best_val = np.inf
        best_idx = None
        for idx in masked_df.index:
            val = masked_df.at[idx, col]
            if val == "---":
                continue
            try:
                numeric_val = float(val.split()[0])
                if numeric_val < best_val:
                    best_val = numeric_val
                    best_idx = idx
            except ValueError:
                continue
        if best_idx is not None:
            masked_df.at[best_idx, col] = f"\\textbf{{{masked_df.at[best_idx, col]}}}"

    return masked_df

def mask_pval_table(ttest_df, pct_df, threshold=0.80, alpha=0.05):
    masked = ttest_df.copy()
    for col in masked.columns:
        for idx in masked.index:
            try:
                # Mask if the invalid sample percent is above threshold
                if float(pct_df.at[idx, col]) > threshold:
                    masked.at[idx, col] = "---"
                    continue
            except Exception:
                masked.at[idx, col] = "---"
                continue

            val = masked.at[idx, col]
            if val == "---" or val is None:
                masked.at[idx, col] = "---"
                continue
            try:
                p = float(val.replace("\\textbf{", "").replace("}", ""))
                if p < alpha:
                    masked.at[idx, col] = f"\\textbf{{{p:.4f}}}"
                else:
                    masked.at[idx, col] = f"{p:.4f}"
            except:
                masked.at[idx, col] = "---"
    return masked

def paired_ttest_crc_vs(method_name, controller_trials, cfg):
    crc_gaps = []
    method_gaps = []

    for trial_idx in range(len(controller_trials)):
        controllers = controller_trials[trial_idx][0]

        if "optimal" not in controllers or "crc" not in controllers or method_name not in controllers:
            continue
        if any(controllers[k] is None for k in ["optimal", "crc", method_name]):
            continue

        C = cfg["test_C"][trial_idx]
        K_opt = controllers["optimal"]
        K_crc = controllers["crc"]
        K_other = controllers[method_name]

        if isinstance(K_opt, tuple): K_opt = K_opt[0]
        if isinstance(K_crc, tuple): K_crc = K_crc[0]
        if isinstance(K_other, tuple): K_other = K_other[0]

        try:
            opt_val = eval_controller(C, K_opt)
            crc_val = eval_controller(C, K_crc)
            other_val = eval_controller(C, K_other)
        except Exception:
            continue

        if not np.isfinite(opt_val) or not np.isfinite(crc_val) or not np.isfinite(other_val):
            continue

        crc_gap = (crc_val - opt_val) / opt_val
        other_gap = (other_val - opt_val) / opt_val

        if np.isfinite(crc_gap) and np.isfinite(other_gap):
            crc_gaps.append(crc_gap)
            method_gaps.append(other_gap)

    crc_gaps = np.array(crc_gaps)
    method_gaps = np.array(method_gaps)

    if len(crc_gaps) < 2:
        return None

    t_stat, p_val = stats.ttest_rel(crc_gaps, method_gaps, alternative="less")
    return t_stat, p_val

def format_pval(p_val, alpha=0.05):
    if p_val is None:
        return "---"
    if p_val < alpha:
        return f"\\textbf{{{p_val:.4f}}}"
    return f"{p_val:.4f}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setups")
    args = parser.parse_args()
    setups = args.setups.split(",")

    optimal_values_df, percent_filter_df, ttest_df = None, None, None

    for setup in setups:
        with open(os.path.join("experiments", f"{setup}.pkl"), "rb") as f:
            cfg = pickle.load(f)

        with open(os.path.join("results", f"{setup}.pkl"), "rb") as f:
            controller_trials = pickle.load(f)

        controller_names = list(label_to_table_name.values())
        if optimal_values_df is None:
            optimal_values_df = pd.DataFrame(index=controller_names)
            percent_filter_df = pd.DataFrame(index=controller_names)

            comparison_methods = [m for m in label_to_table_name if m != "crc"]
            ttest_df = pd.DataFrame(index=[label_to_table_name[m] for m in comparison_methods])

        populate_results(cfg, controller_trials, setup, optimal_values_df, percent_filter_df)

        setup_label = setup_to_table_name[setup]
        for method in comparison_methods:
            result = paired_ttest_crc_vs(method, controller_trials, cfg)
            p_val = result[1] if result is not None else None
            ttest_df.at[label_to_table_name[method], setup_label] = format_pval(p_val)

    masked_optimal_df = mask_and_bold(optimal_values_df, percent_filter_df)
    masked_ttest_df = mask_pval_table(ttest_df, percent_filter_df)
    num_cols = len(masked_optimal_df.columns)
    print(masked_optimal_df.to_latex(escape=False, column_format='l' + 'c' * num_cols))
    print(percent_filter_df.to_latex(column_format='l' + 'c' * num_cols))
    print(masked_ttest_df.to_latex(escape=False, column_format='l' + 'c' * num_cols))