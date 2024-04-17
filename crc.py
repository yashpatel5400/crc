import argparse
import cvxpy as cp
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

from policygradient import PolicyGradientOptions, run_policy_gradient, run_dynamics_gradient, Regularizer
from ltimult import LQRSysMult

def init_system(C, K_0):
    A, B = C[:,:4], C[:,4:]
    
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
                                    C_eta=5e-3,
                                    max_iters=K_steps,
                                    max_C_iters=5_000,
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

def main(C, C_hat, q_hat):   
    # Get nominal system solution (should be bounded above by robust optimal value unless there's a bug)
    # To find K^*, we can use PG but can get exact soln w/ Riccati Equations
    A, B = C[:,:4], C[:,4:]
    K_shape = ([B.shape[1], A.shape[1]])
    nominal_system = init_system(C, np.zeros(K_shape))
    nominal_system.setK(nominal_system.Kare)
    nominal_cost = nominal_system.c

    # Solve robust system using policy gradient (based on Danskin's Theorem)
    robust_system_init = init_system(C_hat, np.zeros(K_shape))
    K_star     = robust_system_init.Kare # np.zeros(K_shape) # get 
    robust_pgo = get_optimizer(np.prod(K_shape), 1) # optimization for K is done in single steps to give correct gradients
    opt_steps  = 200
    for opt_step in range(opt_steps):
        print(f"Step: {opt_step}")
        
        robust_system = init_system(C_hat, K_star) # (i.e. using predictions from generative model (A, B))
        C_star = run_dynamics_gradient(robust_system, robust_pgo, norm=2, q_hat=q_hat) # find the C^* = [A^*, B^*] for Danskin's Theorem (with fixed controller K^(t))
        
        print("---------------------------------------")
        robust_system_star  = init_system(C_star, K_star) # fix C^* from above and take a step to update K
        K_star, robust_cost = run_policy_gradient(robust_system_star, robust_pgo)
    
        # Print the regularized optimal gains (from proximal gradient optimization)
        # and the unregularized optimal gains (from solving a Riccati equation)
        print(f"Nominal Cost : {nominal_cost} | Robust Cost : {robust_cost}")
    return C_star, K_star, robust_cost

if __name__ == "__main__":
    with open("experiments/airfoil.pkl", "rb") as f:
        cfg = pickle.load(f)
    C_star, K_star, robust_cost = main(cfg["test_C"][0], cfg["test_C_hat"][0], cfg["q_hat"])