import argparse
import cvxpy as cp
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_dynamics import generate_data
from policygradient import PolicyGradientOptions, run_policy_gradient, run_dynamics_gradient, Regularizer
from ltimult import LQRSysMult

approach_config = {
    "box": {
        "K": 1,
        "norm": 1,
        "prediction_type": "mean",
    }, 
    "ptc_b": {
        "K": 1,
        "norm": 1,
        "prediction_type": "gen",
    }, 
    "ellipsoid":  {
        "K": 1,
        "norm": 2,
        "prediction_type":  "mean",
    }, 
    "ptc_e":  {
        "K": 1,
        "norm": 2,
        "prediction_type": "gen",
    }, 
    "crc":  {
        "K": 10,
        "norm": 2,
        "prediction_type": "gen",
    }
}

def mtx_mean(Cs, norm):
    C_mean = cp.Variable(Cs[0].shape)
    obj = sum([cp.atoms.norm(C - C_mean, norm) for C in Cs])
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()
    return C_mean.value

def generate_full_dataset():
    # C in R^2x3
    cal_sims = 1_000
    cal_C, cal_x = generate_data(cal_sims)
    cal_C = cal_C.reshape((cal_sims, 2, 3))
    
    test_sims = 1
    test_C, test_x = generate_data(test_sims)
    test_C = test_C.detach().cpu().numpy().reshape((2, 3))
    return (cal_x, cal_C), (test_x, test_C)

def calc_scores(Cs, C_hats, norm):
    K = C_hats.shape[1]
    cal_tiled = np.transpose(np.tile(Cs.detach().cpu().numpy(), (K, 1, 1, 1)), (1, 0, 2, 3))
    cal_diff = C_hats - cal_tiled
    cal_norms = np.linalg.norm(cal_diff, ord=norm, axis=(2,3))
    return np.min(cal_norms, axis=-1)

def construct_prediction_region(cal_x, cal_C, test_x, K, norm, prediction_type, alpha=0.05):
    device = "cpu"
    cached_fn = os.path.join("dynamics_trained", "dynamics.nf")
    with open(cached_fn, "rb") as f:
        encoder = pickle.load(f)
    encoder.to(device)

    if prediction_type == "mean":
        mean_cal = mtx_mean(cal_C, norm)
        predict  = lambda x : np.tile(mean_cal, (x.shape[0],1,1,1))
    else:
        predict = lambda x : encoder.sample(K, x).detach().cpu().numpy().reshape((x.shape[0], K, 2, 3))

    cal_C_hats = predict(cal_x)
    cal_scores = calc_scores(cal_C, cal_C_hats, norm)
    q_hat = np.quantile(cal_scores, q = 1-alpha)

    test_C_hats = predict(test_x)[0]
    return test_C_hats, q_hat

def init_system(C, K_0):
    A, B = C[:,:2], C[:,2:]
    
    # System problem data
    Q = np.eye(A.shape[-1])
    R = np.eye(B.shape[-1])
    S0 = np.eye(A.shape[-1])

    # useless for us, but just keep to use library code
    Aa = np.zeros(A.shape)
    Aa = Aa[:, :, np.newaxis]
    Bb = np.zeros(B.shape)
    Bb = Bb[:, :, np.newaxis]
    a = np.zeros((1, 1))
    b = np.zeros((1, 1))

    SS = LQRSysMult(A, B, a, Aa, b, Bb, Q, R, S0)
    SS.setK(K_0)
    return SS

def get_optimizer(K_size, K_steps):
    return PolicyGradientOptions(epsilon=(1e-2) * K_size,
                                    eta=1e-3,
                                    max_iters=K_steps,
                                    max_C_iters=500,
                                    disp_stride=1,
                                    keep_hist=True,
                                    opt_method='proximal',
                                    keep_opt='last',
                                    step_direction='gradient',
                                    stepsize_method='constant',
                                    exact=True,
                                    regularizer=Regularizer('vec1'),
                                    regweight=0.0,
                                    stop_crit='gradient',
                                    fbest_repeat_max=0,
                                    display_output=True,
                                    display_inplace=True,
                                    slow=False)

def main(C, C_hats, q_hat):   
    # Get nominal system solution (should be bounded above by robust optimal value unless there's a bug)
    # To find K^*, we can use PG but can get exact soln w/ Riccati Equations
    A, B = C[:,:2], C[:,2:]
    K_shape = ([B.shape[1], A.shape[1]])
    nominal_system = init_system(C, np.zeros(K_shape))
    nominal_system.setK(nominal_system.Kare)
    nominal_cost = nominal_system.c

    # Solve robust system using policy gradient (based on Danskin's Theorem)
    K_star      = np.zeros(K_shape)
    robust_pgo  = get_optimizer(np.prod(K_shape), 500) # optimization for K is done in single steps to give correct gradients
    opt_steps   = 1
    for opt_step in range(opt_steps):
        print(f"Step: {opt_step}")
        
        # Need to find argmax_(C_k) l(C_k) for current K^*
        C_k_stars, l_k_stars = [], []
        for k in range(len(C_hats)):
            robust_system_k = init_system(C_hats[k], K_star) # (i.e. using predictions from generative model (A, B))
            C_k_star, l_k_star = run_dynamics_gradient(robust_system_k, robust_pgo, norm, q_hat) # find the C^* = [A^*, B^*] for Danskin's Theorem (with fixed controller K^(t))
            C_k_stars.append(C_k_star)
            l_k_stars.append(l_k_star)
        C_star = C_k_stars[np.argmax(l_k_stars)]

        print("---------------------------------------")
        robust_system_star = init_system(C_star, K_star) # fix C^* from above and take a step to update K
        K_star, robust_cost = run_policy_gradient(robust_system_star, robust_pgo)
    
        # Print the regularized optimal gains (from proximal gradient optimization)
        # and the unregularized optimal gains (from solving a Riccati equation)
        print(f"Nominal Cost : {nominal_cost} | Robust Cost : {robust_cost}")
    return C_star, K_star, robust_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=["box", "ptc_b", "ellipsoid", "ptc_e", "crc"])
    parser.add_argument("--generate_data", action='store_true')
    args = parser.parse_args()

    cached_data_fn = "data.pkl"
    if args.generate_data:
        # by default, we just terminate the program post-generation to avoid bugs in running
        (cal_x, cal_C), (test_x, test_C) = generate_full_dataset()
        with open(cached_data_fn, "wb") as f:
            pickle.dump(((cal_x, cal_C), (test_x, test_C)), f)
    else:
        with open(cached_data_fn, "rb") as f:
            (cal_x, cal_C), (test_x, test_C) = pickle.load(f)
        
        cfg = approach_config[args.approach]
        K, norm, prediction_type = cfg["K"], cfg["norm"], cfg["prediction_type"]
        test_C_hats, q_hat = construct_prediction_region(cal_x, cal_C, test_x, K, norm, prediction_type)
        C_star, K_star, robust_cost = main(test_C, test_C_hats, q_hat)
        with open(os.path.join("results", f"{args.approach}.pkl"), "wb") as f:
            pickle.dump((C_star, K_star, robust_cost), f)