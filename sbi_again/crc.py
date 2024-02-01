import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_dynamics import generate_data
from policygradient import PolicyGradientOptions, run_policy_gradient, run_dynamics_gradient, Regularizer
from ltimult import LQRSysMult

def calc_scores(encoder, xs, Cs, k=10):
    cal_samples = encoder.sample(k, xs).detach().cpu().numpy()
    cal_samples = cal_samples.reshape((cal_samples.shape[0], k, 2, 3))
    cal_tiled = np.transpose(np.tile(Cs.detach().cpu().numpy(), (k, 1, 1, 1)), (1, 0, 2, 3))
    cal_diff = cal_samples - cal_tiled
    cal_norms = np.linalg.norm(cal_diff, ord=2, axis=(2,3))
    return np.min(cal_norms, axis=-1)

def get_real_robust_dynamics(alpha=0.05):
    device = "cpu"
    cached_fn = os.path.join("dynamics_trained", "dynamics.nf")
    with open(cached_fn, "rb") as f:
        encoder = pickle.load(f)
    encoder.to(device)

    # C in R^2x3
    cal_sims = 1_000
    cal_C_hats, cal_x = generate_data(cal_sims)
    cal_C_hats = cal_C_hats.reshape((cal_sims, 2, 3))
    cal_scores = calc_scores(encoder, cal_x, cal_C_hats)
    q_hat = np.quantile(cal_scores, q = 1-alpha)

    k = 10
    test_sims = 1
    test_C, test_x = generate_data(test_sims)
    test_C_hats = encoder.sample(k, test_x).detach().cpu().numpy()
    
    test_C = test_C.detach().cpu().numpy().reshape((2, 3))
    test_C_hats = test_C_hats.reshape((k, 2, 3))
    return test_C, (test_C_hats, q_hat)

def init_system(A, B, K_0):
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

def main():
    # Get system dynamics from side information
    C, (C_hats, q_hat) = get_real_robust_dynamics(alpha=0.05)
    A, B    = C[:,:2], C[:,2:]
    K_shape = ([B.shape[1], A.shape[1]])
    
    # Get nominal system solution (should be bounded above by robust optimal value unless there's a bug)
    # To find K^*, we can use PG but can get exact soln w/ Riccati Equations
    nominal_system = init_system(A, B, np.zeros(K_shape)) # (i.e. using true (A, B))
    nominal_system.setK(nominal_system.Kare)
    nominal_cost = nominal_system.c

    # Solve robust system using policy gradient (based on Danskin's Theorem)
    K_star      = np.zeros(K_shape)
    robust_pgo  = get_optimizer(np.prod(K_shape), 500) # optimization for K is done in single steps to give correct gradients
    opt_steps   = 2_000
    for opt_step in range(opt_steps):
        print(f"Step: {opt_step}")
        
        # Need to find argmax_(C_k) l(C_k) for current K^*
        C_k_stars, l_k_stars = [], []
        for k in range(len(C_hats)):
            A_k_hat, B_k_hat = C_hats[k][:,:2], C_hats[k][:,2:]
            robust_system_k = init_system(A_k_hat, B_k_hat, K_star)       # (i.e. using predictions from generative model (A, B))
            C_k_star, l_k_star = run_dynamics_gradient(robust_system_k, robust_pgo, q_hat) # find the C^* = [A^*, B^*] for Danskin's Theorem (with fixed controller K^(t))
            C_k_stars.append(C_k_star)
            l_k_stars.append(l_k_star)
        C_star = C_k_stars[np.argmax(l_k_stars)]

        print("---------------------------------------")
        A_star, B_star = C_star[:,:2], C_star[:,2:]
        robust_system_star = init_system(A_star, B_star, K_star) # fix C^* from above and take a step to update K
        K_star, robust_cost = run_policy_gradient(robust_system_star, robust_pgo)
    
        # Print the regularized optimal gains (from proximal gradient optimization)
        # and the unregularized optimal gains (from solving a Riccati equation)
        print(f"Nomninal Cost : {nominal_cost} | Robust Cost : {robust_cost}")

if __name__ == "__main__":
    main()