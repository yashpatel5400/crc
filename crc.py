import argparse
import cvxpy as cp
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are
import multiprocessing
import torch

from polgrad.policygradient import PolicyGradientOptions, run_policy_gradient, run_dynamics_gradient, Regularizer
from polgrad.ltimult import LQRSysMult, dare_mult, dlyap_mult
from polgrad.robust import algo1, algo2

def get_noise_matrices(mult_noise_method, n, m, p):
    if mult_noise_method == 'random':
        noise = np.random.standard_normal([n, m, p])
    elif mult_noise_method == 'rowcol':
        # Pick a random row and column
        noise = np.zeros([n, m, p])
        noise[np.random.randint(n), :, 0] = np.ones(m)
        noise[:, np.random.randint(m), 1] = np.ones(n)
    elif mult_noise_method == 'random_plus_rowcol':
        noise = 0.3 * np.random.standard_normal([n, m, p])
        noise[np.random.randint(n), :, 0] = np.ones(m)
        noise[:, np.random.randint(m), 1] = np.ones(n)
    return noise

def get_noise(mult_noise_method, noise, nom_system_params):
    A, B, Q, R, S0 = nom_system_params
    n, m = B.shape

    # Multiplicative noise data
    p = 5  # Number of multiplicative noises on A
    q = 5  # Number of multiplicative noises on B
    Aa = get_noise_matrices(mult_noise_method, n, n, p)
    Bb = get_noise_matrices(mult_noise_method, n, m, q)

    incval = 1.05
    decval = 1.00 * (1 / incval)
    weakval = 0.90

    # a = np.random.standard_normal([p,1])
    # b = np.random.standard_normal([q,1])
    a = np.ones([p, 1])
    b = np.ones([q, 1])
    a = a * (float(1) / (p * n**2))  # scale as rough heuristic
    b = b * (float(1) / (q * m**2))  # scale as rough heuristic

    #    noise = 'weak'
    if noise == 'weak' or noise == 'critical':
        # Ensure near-critically mean square stabilizable
        # increase noise if not
        P, Kare = dare_mult(A, B, a, Aa, b, Bb, Q, R, show_warn=False)
        mss = True
        while mss:
            if Kare is None:
                mss = False
            else:
                a = incval * a
                b = incval * b
                P, Kare = dare_mult(A, B, a, Aa, b, Bb, Q, R, show_warn=False)
        # Extra mean square stabilizability margin
        a = decval * a
        b = decval * b
        if noise == 'weak':
            #            print('Multiplicative noise set weak')
            a = weakval * a
            b = weakval * b
    elif noise == 'olmss_weak' or noise == 'olmss_critical':
        # Ensure near-critically open-loop mean-square stable
        # increase noise if not
        K0 = np.zeros([m, n])
        P = dlyap_mult(A, B, K0, a, Aa, b, Bb, Q, R, S0, matrixtype='P')
        mss = True
        while mss:
            if P is None:
                mss = False
            else:
                a = incval * a
                b = incval * b
                P = dlyap_mult(A, B, K0, a, Aa, b, Bb, Q, R, S0, matrixtype='P')
        # Extra mean square stabilizability margin
        a = decval * a
        b = decval * b
        if noise == 'olmss_weak':
            #            print('Multiplicative noise set to open-loop mean-square stable')
            a = weakval * a
            b = weakval * b
    elif noise == 'olmsus':
        # Ensure near-critically open-loop mean-square unstable
        # increase noise if not
        K0 = np.zeros([m, n])
        P = dlyap_mult(A, B, K0, a, Aa, b, Bb, Q, R, S0, matrixtype='P')
        mss = True
        while mss:
            if P is None:
                mss = False
            else:
                a = incval * a
                b = incval * b
                P = dlyap_mult(A, B, K0, a, Aa, b, Bb, Q, R, S0, matrixtype='P')
    #        # Extra mean square stabilizability margin
    #        a = decval*a
    #        b = decval*b
    #        print('Multiplicative noise set to open-loop mean-square unstable')
    elif noise == 'none':
        print('MULTIPLICATIVE NOISE SET TO ZERO!!!')
        a = np.zeros([p, 1])  # For testing only - no noise
        b = np.zeros([q, 1])  # For testing only - no noise
    else:
        raise Exception('Invalid noise setting chosen')

    return a, Aa, b, Bb

def init_system(mult_noise_method, noise, C, K_0):
    m, n = K_0.shape
    A, B = C[:,:n], C[:,n:]
    
    # System problem data
    Q = np.eye(A.shape[-1])
    R = np.eye(B.shape[-1])
    S0 = np.eye(A.shape[-1])

    # useless for us, but just keep to use library code
    if noise == "none":
        Aa = np.zeros(A.shape)
        Aa = Aa[:, :, np.newaxis]
        Bb = np.zeros(B.shape)
        Bb = Bb[:, :, np.newaxis]
        a = np.zeros((1, 1))
        b = np.zeros((1, 1))
    else:
        a, Aa, b, Bb = get_noise(mult_noise_method, noise, (A, B, Q, R, S0))
    SS = LQRSysMult(A, B, a, Aa, b, Bb, Q, R, S0)
    SS.setK(K_0)
    return SS

def get_optimizer(K_size, K_steps):
    return PolicyGradientOptions(epsilon=(1e-2) * K_size,
                                    eta=1e-3,
                                    C_eta=5e-4,
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

def eval_controller(C, K):
    nominal_system = init_system("random", "none", C, K)
    return nominal_system.c

def hinf(C_hat, gamma=10.0):
    n, _ = C_hat.shape
    A_hat, B_hat = C_hat[:,:n], C_hat[:,n:]
    n, m = B_hat.shape
    Q = np.eye(A_hat.shape[-1])
    
    R_rob = np.linalg.inv(np.eye(m) - 1 / gamma**2 * np.linalg.inv(B_hat.T @ B_hat))
    M = solve_discrete_are(A_hat, B_hat, Q, R_rob)
    Lambda = np.eye(n) + (B_hat @ B_hat.T - 1 / gamma**2 * np.eye(n)) @ M
    K_inf = -B_hat.T @ M @ np.linalg.inv(Lambda) @ A_hat
    return K_inf

def crc(C_hat, q_hat):
    n, _ = C_hat.shape
    A_hat, B_hat = C_hat[:,:n], C_hat[:,n:]
    K_shape = ([B_hat.shape[1], A_hat.shape[1]])

    try:
        # Solve robust system using policy gradient (based on Danskin's Theorem)
        robust_system_init = init_system("random", "none", C_hat, np.zeros(K_shape))
        K_star     = robust_system_init.Kare # initial with nominal ARE solution 
        robust_pgo = get_optimizer(np.prod(K_shape), 1) # optimization for K is done in single steps to give correct gradients
        opt_steps  = 200
        for opt_step in range(opt_steps):
            print(f"Step: {opt_step}")
            
            robust_system = init_system("random", "none", C_hat, K_star) # (i.e. using predictions from generative model (A, B))
            C_star, _ = run_dynamics_gradient(robust_system, robust_pgo, norm=2, q_hat=q_hat) # find the C^* = [A^*, B^*] for Danskin's Theorem (with fixed controller K^(t))
            
            print("---------------------------------------")
            robust_system_star  = init_system("random", "none", C_star, K_star) # fix C^* from above and take a step to update K
            K_star, _ = run_policy_gradient(robust_system_star, robust_pgo)
        return K_star
    except:
        return None # if prediction region is too large, GDA falls outside stabilizing region
    
def get_ctrls(args):
    C, C_hat, q_hat, setup, C_idx = args
    n, _ = C_hat.shape
    A, B = C[:,:n], C[:,n:]
    A_hat, B_hat = C_hat[:,:n], C_hat[:,n:]
    K_shape = ([B.shape[1], A.shape[1]])
    PGO = get_optimizer(np.prod(K_shape), K_steps=1_000)

    ctrls = {}
    ctrls["optimal"] = init_system("random", "none", C, np.zeros(K_shape)).Kare
    ctrls["nominal"] = init_system("random", "none", C_hat, np.zeros(K_shape)).Kare
    
    # baseline from: "Robust control design for linear systems via multiplicative noise"
    p = 5
    Ai = np.transpose(get_noise_matrices("random", n, n, p), (2, 0, 1))
    Q = np.eye(A.shape[-1])
    R = np.eye(B.shape[-1])
    eta_bar = np.ones(p) * .1
    
    try:
        K1, _ = algo1(A_hat, B_hat, Ai, Q, R, eta_bar)
        ctrls["mult_alg1"] = K1
    except:
        ctrls["mult_alg1"] = None
    
    try:
        K2, _ = algo2(A_hat, B_hat, Ai, Q, R, eta_bar)
        ctrls["mult_alg2"] = K2
    except:
        ctrls["mult_alg2"] = None
        
    # baseline from: standard h-infinity control
    for gamma in [0.5,1.0]:
        ctrl_name = f"hinf_{gamma}"
        try:
            ctrls[ctrl_name] = hinf(C_hat, gamma=gamma)
        except:
            ctrls[ctrl_name] = None

    # baselines from: "Learning optimal controllers for linear systems with multiplicative noise via policy gradient"
    for mult_noise_method in ["random", "rowcol"]:
        for noise in ["critical", "olmss_weak", "olmsus"]:
            ctrl_name = f"{mult_noise_method}-{noise}"
            if ctrls["nominal"] is None:
                ctrls[ctrl_name] = None
                continue
            
            SS = init_system(mult_noise_method, noise, C_hat, ctrls["nominal"])
            try:
                ctrls[ctrl_name] = run_policy_gradient(SS, PGO)
            except:
                ctrls[ctrl_name] = None

    # proposed conformal control method
    ctrls["crc"] = crc(C_hat, q_hat)
    return ctrls

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup")
    args = parser.parse_args()
    setup = args.setup

    with open(os.path.join(f"experiments", f"{setup}.pkl"), "rb") as f:
        cfg = pickle.load(f)

    os.makedirs(os.path.join(f"results", setup), exist_ok=True)
    workers = 50
    pool = multiprocessing.Pool(workers)
    controllers = list(pool.map(
       get_ctrls,
       [(cfg["test_C"][C_idx], cfg["test_C_hat"][C_idx], cfg["q_hat"], setup, C_idx) for C_idx in range(len(cfg["test_C"]))]
    ))

    os.makedirs("results", exist_ok=True)
    with open(os.path.join(f"results", f"{setup}.pkl"), "wb") as f:
        pickle.dump(controllers, f)