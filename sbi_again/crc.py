import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_dynamics import generate_data
from policygradient import PolicyGradientOptions, run_policy_gradient, run_dynamics_gradient, Regularizer
from ltimult import LQRSysMult

def calc_scores(xs, Cs, k=10):
    cal_samples = encoder.sample(k, xs).detach().cpu().numpy()
    cal_samples = cal_samples.reshape((cal_samples.shape[0], k, 2, 3))
    cal_tiled = np.transpose(np.tile(Cs.detach().cpu().numpy(), (k, 1, 1, 1)), (1, 0, 2, 3))
    cal_diff = cal_samples - cal_tiled
    cal_norms = np.linalg.norm(cal_diff, ord=2, axis=(2,3))
    return np.min(cal_norms, axis=-1)

device = "cpu"
cached_fn = os.path.join("dynamics_trained", "dynamics.nf")
with open(cached_fn, "rb") as f:
    encoder = pickle.load(f)
encoder.to(device)

# C in R^2x3
cal_sims = 1_000
cal_C_hats, cal_x = generate_data(cal_sims)
cal_C_hats = cal_C_hats.reshape((cal_sims, 2, 3))
cal_scores = calc_scores(cal_x, cal_C_hats)
q_hat = np.quantile(cal_scores, q = 0.95)

k = 10
test_sims = 1
_, test_x = generate_data(test_sims)
test_C_hats = encoder.sample(k, test_x).detach().cpu().numpy()
test_C_hats = test_C_hats.reshape((test_C_hats.shape[0], k, 2, 3))

# Get system dynamics from side information
C_hat = test_C_hats[0][0]
A = C_hat[:,:2]
B = C_hat[:,2:]

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

# Start with an initially stabilizing (feasible) controller;
# for this example the system is open-loop mean-square stable
A_star = np.copy(A)
B_star = np.copy(B)
K_star = np.zeros([B.shape[1], A.shape[1]])

PGO = PolicyGradientOptions(epsilon=(1e-2) * K_star.size,
                                eta=1e-3,
                                max_iters=1,
                                max_C_iters=1_000,
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

opt_steps = 1
for opt_step in range(opt_steps):
    SS = LQRSysMult(A_star, B_star, a, Aa, b, Bb, Q, R, S0)
    SS.setK(K_star)

    # Run (regularized) policy gradient
    run_dynamics_gradient(SS, PGO, C_hat, q_hat) # find the C^* = [A^*, B^*] for Danskin's Theorem (with fixed controller K^(t))
    print("---------------------------------------")
    run_policy_gradient(SS, PGO)   # fix C^* from above and take a step to update K

    A_star = SS.A
    B_star = SS.B
    K_star = SS.K

    # Print the regularized optimal gains (from proximal gradient optimization)
    # and the unregularized optimal gains (from solving a Riccati equation)
    print('Optimized sparse gains')
    print(SS.K)
    print('Riccati gains')
    print(SS.Kare)