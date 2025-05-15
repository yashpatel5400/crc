import argparse
import cvxpy as cp
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import einops
import seaborn as sns
from pydmd import DMDc

device = "cuda"

np.random.seed(0)

def load_pos_generate_dynamics_matrices(num_samples):
    # Sample from the reparameterized variable ranges
    inv_m_L = np.random.uniform(0.3333, 1.0, size=(num_samples, 1))
    inv_m_B = np.random.uniform(0.04, 0.0667, size=(num_samples, 1))
    kB_over_mB = np.random.uniform(0.4, 1.3333, size=(num_samples, 1))
    dB_over_mB = np.random.uniform(0.004, 0.0667, size=(num_samples, 1))

    # Fix dL as specified in the paper
    dL = 10.0

    # Recover original parameters
    m_L = 1.0 / inv_m_L
    m_B = 1.0 / inv_m_B
    k_B = kB_over_mB * m_B
    d_B = dB_over_mB * m_B

    thetas = np.hstack([m_B, m_L, np.full_like(m_L, dL), k_B, d_B])

    # Build dynamics matrices
    As = np.zeros((num_samples, 4, 4))
    As[:, 0, 1] = 1
    As[:, 1, 1] = -dL / m_L[:, 0] - dL / m_B[:, 0]
    As[:, 1, 2] = k_B[:, 0] / m_B[:, 0]
    As[:, 1, 3] = d_B[:, 0] / m_B[:, 0]
    As[:, 2, 3] = 1
    As[:, 3, 1] = dL / m_B[:, 0]
    As[:, 3, 2] = -k_B[:, 0] / m_B[:, 0]
    As[:, 3, 3] = -d_B[:, 0] / m_B[:, 0]

    Bs = np.zeros((num_samples, 4, 1))
    Bs[:, 1, 0] = 1.0 / m_L[:, 0] + 1.0 / m_B[:, 0]
    Bs[:, 3, 0] = -1.0 / m_B[:, 0]

    return thetas, (As, Bs)

# distribution parameters need to be fixed for the simulation
def get_rand_mean_sigma(d):
    mu        = np.random.rand(d)
    sigma_tmp = np.random.rand(d, d)
    sigma     = np.dot(sigma_tmp, sigma_tmp.transpose())
    return mu, sigma

mu_G, sigma_G = get_rand_mean_sigma(5)
mu_L, sigma_L = get_rand_mean_sigma(5)
mu_N, sigma_N = get_rand_mean_sigma(5)

g       = 1
U_0     = 1
theta_0 = 0

def airfoil_generate_dynamics_matrices(num_samples):
    G = np.random.multivariate_normal(mu_G, sigma_G, num_samples)
    L = np.random.multivariate_normal(mu_L, sigma_L, num_samples)
    N = np.random.multivariate_normal(mu_N, sigma_N, num_samples)
    
    final_row = np.ones((G.shape[0], 4))
    final_row[:,0] = final_row[:,3] = 0
    final_row[:,2] = np.tan(theta_0)

    thetas = np.hstack([G, L, N])
    As = np.transpose(np.array([
        np.hstack([G[:,:3], np.ones((G.shape[0],1)) * (g * np.cos(theta_0))]) / U_0,
        np.hstack([L[:,:3], np.zeros((L.shape[0],1))]),
        np.hstack([N[:,:3], np.zeros((N.shape[0],1))]),
        final_row,
    ]), (1,0,2))

    Bs = np.transpose(np.array([
        G[:,3:] / U_0,
        L[:,3:],
        N[:,3:],
        np.zeros((G.shape[0], 2)),
    ]), (1,0,2))
    
    return thetas, (As, Bs)

pendulum_means = {
    # Motor parameters
    "R_m": 8.4,
    "K_t": 0.042,
    "K_m": 0.042,

    # Rotor parameters
    "m_p": 0.095,
    "L_r": 0.085,
    "D_r": 0.0015,

    # Pendulum parameters
    "M_p": 0.024,
    "L_p": 0.129,
    "D_p": 0.0005,
}

pendulum_means["J_r"] = (pendulum_means["m_p"] * pendulum_means["L_r"]**2) / 12
pendulum_means["J_p"] = (pendulum_means["M_p"] * pendulum_means["L_p"]**2) / 12
pendulum_means["J_T"] = pendulum_means["J_p"] * pendulum_means["m_p"] * pendulum_means["L_r"]**2 + pendulum_means["J_r"] * pendulum_means["J_p"] + (1/4) * pendulum_means["J_r"] * pendulum_means["m_p"] * pendulum_means["L_p"]**2

pendulum_std_devs = {key: np.random.random() for key, value in pendulum_means.items()}

def pendulum_generate_dynamics_matrices(num_samples):
    scale = 1
    M_p = np.expand_dims(np.abs(np.random.normal(pendulum_means["M_p"], pendulum_std_devs["M_p"], num_samples)), axis=0) * scale
    m_p = np.expand_dims(np.abs(np.random.normal(pendulum_means["m_p"], pendulum_std_devs["m_p"], num_samples)), axis=0) * scale
    L_p = np.expand_dims(np.abs(np.random.normal(pendulum_means["L_p"], pendulum_std_devs["L_p"], num_samples)), axis=0) * scale
    L_r = np.expand_dims(np.abs(np.random.normal(pendulum_means["L_r"], pendulum_std_devs["L_r"], num_samples)), axis=0) * scale
    J_T = np.expand_dims(np.abs(np.random.normal(pendulum_means["J_T"], pendulum_std_devs["J_T"], num_samples)), axis=0) * scale
    J_p = np.expand_dims(np.abs(np.random.normal(pendulum_means["J_p"], pendulum_std_devs["J_p"], num_samples)), axis=0) * scale
    J_r = np.expand_dims(np.abs(np.random.normal(pendulum_means["J_r"], pendulum_std_devs["J_r"], num_samples)), axis=0) * scale
    D_p = np.expand_dims(np.abs(np.random.normal(pendulum_means["D_p"], pendulum_std_devs["D_p"], num_samples)), axis=0) * scale
    D_r = np.expand_dims(np.abs(np.random.normal(pendulum_means["D_r"], pendulum_std_devs["D_r"], num_samples)), axis=0) * scale

    thetas = np.vstack([M_p, L_p, L_r, J_T, J_p, J_r, m_p, D_p, D_r]).T
    div_J_T = np.expand_dims(np.expand_dims(J_T.flatten(), axis=-1), axis=-1)
    
    As = np.zeros((num_samples, 4, 4))
    As[:,0,2] = J_T
    As[:,1,3] = J_T

    As[:,2,1] = 1/4 * M_p ** 2 * L_p ** 2 * L_r ** 2 * g
    As[:,2,2] = -(J_p + 1/4 * m_p * L_p ** 2) * D_r
    As[:,2,3] = 1/2 * m_p * L_p * L_r * D_p

    As[:,3,1] = -1/2 * m_p * L_p * g * (J_r + m_p * L_r ** 2)
    As[:,3,2] = 1/2 * m_p * L_p * L_r * D_r
    As[:,3,3] = -(J_r + m_p * L_r ** 2) * D_p
    As /= div_J_T

    Bs = np.zeros((num_samples, 4, 1))
    Bs[:,2,0] = J_p + 1/4 * m_p * L_p ** 2
    Bs[:,3,0] = -1/2 * m_p * L_p * L_r
    Bs /= div_J_T

    return thetas, (As, Bs)

# Define the constants
T = 298.15
F = 96485
R = 8.314
E = 1.4

battery_means = {
    "N": 37,
    "K2": 8.768e-10,
    "K3": 3.222e-10,
    "K4": 6.825e-10,
    "K5": 5.897e-10,
    "Vs": 40,
    "Vt": 500,
    "S": 24,
    "d": 1.27e-3,
    "le": 6,
    "we": 0.04,
    "he": 4,
    "r": 0.03,
    "Cc2": 1.0,
    "Cc3": 1.0,
    "Cc4": 1.0,
    "Cc5": 1.0,
    "Ct2": 1.0,
    "Ct3": 1.0,
    "Ct4": 1.0,
    "Ct5": 1.0
}

battery_std_scale = 1.5
battery_std_devs = {key: value / battery_std_scale for key, value in battery_means.items()}

def battery_generate_dynamics_matrices(num_samples):
    Vs  = np.expand_dims(np.random.normal(battery_means["Vs"], battery_std_devs["Vs"], num_samples), axis=0)
    Vt  = np.expand_dims(np.random.normal(battery_means["Vt"], battery_std_devs["Vt"], num_samples), axis=0)
    
    S   = np.expand_dims(np.random.normal(battery_means["S"], battery_std_devs["S"], num_samples), axis=0)
    d   = np.expand_dims(np.random.normal(battery_means["d"], battery_std_devs["d"], num_samples), axis=0)
    N   = np.expand_dims(np.random.normal(battery_means["N"], battery_std_devs["N"], num_samples), axis=0)
    
    K2  = np.expand_dims(np.random.normal(battery_means["K2"], battery_std_devs["K2"], num_samples), axis=0)
    K3  = np.expand_dims(np.random.normal(battery_means["K3"], battery_std_devs["K3"], num_samples), axis=0)
    K4  = np.expand_dims(np.random.normal(battery_means["K4"], battery_std_devs["K4"], num_samples), axis=0)
    K5  = np.expand_dims(np.random.normal(battery_means["K5"], battery_std_devs["K5"], num_samples), axis=0)
    
    Cc2 = np.expand_dims(np.random.normal(battery_means["Cc2"], battery_std_devs["Cc2"], num_samples), axis=0)
    Cc3 = np.expand_dims(np.random.normal(battery_means["Cc3"], battery_std_devs["Cc3"], num_samples), axis=0)
    Cc4 = np.expand_dims(np.random.normal(battery_means["Cc4"], battery_std_devs["Cc4"], num_samples), axis=0)
    Cc5 = np.expand_dims(np.random.normal(battery_means["Cc5"], battery_std_devs["Cc5"], num_samples), axis=0)
    
    Ct2 = np.expand_dims(np.random.normal(battery_means["Ct2"], battery_std_devs["Ct2"], num_samples), axis=0)
    Ct3 = np.expand_dims(np.random.normal(battery_means["Ct3"], battery_std_devs["Ct3"], num_samples), axis=0)
    Ct4 = np.expand_dims(np.random.normal(battery_means["Ct4"], battery_std_devs["Ct4"], num_samples), axis=0)
    Ct5 = np.expand_dims(np.random.normal(battery_means["Ct5"], battery_std_devs["Ct5"], num_samples), axis=0)

    thetas = np.vstack([Vs, Vt, S, d, N, K2, K3, K4, K5, Cc2, Cc3, Cc4, Cc5, Ct2, Ct3, Ct4, Ct5]).T

    As = np.zeros((num_samples, 9, 9))
    Bs = np.zeros((num_samples, 9, 1))
    
    As[:,0,0] = 2*(-E*d - N*K2*S)/(Vs*d)
    As[:,0,2] = -2*N*K4*S/(Vs*d)
    As[:,0,3] = -4*N*K5*S/(Vs*d)
    As[:,0,4] = 2*E/Vs

    As[:,1,1] = 2*(-E*d - N*K3*S)/(Vs*d)
    As[:,1,2] = 4*N*K4*S/(Vs*d)
    As[:,1,3] = 6*N*K5*S/(Vs*d)
    As[:,1,5] = 2*E/Vs

    As[:,2,0] = 6*N*K2*S/(Vs*d)
    As[:,2,1] = 4*N*K3*S/(Vs*d)
    As[:,2,2] = 2*(-E*d - N*K4*S)/(Vs*d)
    As[:,2,6] = 2*E/Vs

    As[:,3,0] = -4*N*K2*S/(Vs*d)
    As[:,3,1] = -2*N*K3*S/(Vs*d)
    As[:,3,3] = 2*(-E*d - N*K5*S)/(Vs*d)
    As[:,3,6] = 2*E/Vs

    As[:,4,0] = E/Vt
    As[:,4,4] = -E/Vt

    As[:,5,1] = E/Vt
    As[:,5,5] = -E/Vt

    As[:,6,2] = E/Vt
    As[:,6,6] = -E/Vt

    As[:,7,3] = E/Vt
    As[:,7,7] = -E/Vt

    As[:,8,0] = N*R*T/(F*Cc2)
    As[:,8,1] = -N*R*T/(F*Cc3)
    As[:,8,2] = -N*R*T/(F*Cc4)
    As[:,8,3] = -N*R*T/(F*Cc5)

    Bs[:,0,0] = (Ct2 - Cc2)/(Vs/2)
    Bs[:,1,0] = (Ct3 - Cc3)/(Vs/2)
    Bs[:,2,0] = (Ct4 - Cc4)/(Vs/2)
    Bs[:,3,0] = (Ct5 - Cc5)/(Vs/2)
    Bs[:,4,0] = (Cc2 - Ct2)/Vt
    Bs[:,5,0] = (Cc3 - Ct3)/Vt
    Bs[:,6,0] = (Cc4 - Ct4)/Vt
    Bs[:,7,0] = (Cc5 - Ct5)/Vt

    return thetas, (As, Bs)

# Fusion rod control
# Note that some of these values were not reported in the original paper and were instead pulled from
# "Static output feedback H∞ based integral sliding mode control law design for nuclear reactor power-level."
# Additionally, some of the units of the values were changed for numerical stability from their reported values of the paper
fusion_means = {
    "alpha_c": -2.0,
    "alpha_f": -14.0,

    "beta": 0.0065,    # Total delayed neutron fraction
    "beta1": 0.00021,  # Fraction of 1st group neutron precursor
    "beta2": 0.00225,  # Fraction of 2nd group neutron precursor
    "beta3": 0.00404,  # Fraction of 3rd group neutron precursor
    
    "Lambda":   2.1,  # us^-1 (prompt neutron lifetime)
    "lambda_I": 10.0, # us^-1 (decay constant of iodine)
    "lambda_X": 2.9,  # us^-1 (decay constant of xenon)

    "lambda1": 0.0124,  # s^-1 (decay constant of 1st group neutron precursor)
    "lambda2": 0.0369,  # s^-1 (decay constant of 2nd group neutron precursor)
    "lambda3": 0.632,   # s^-1 (decay constant of 3rd group neutron precursor)
    
    "mu_c": .001,  # GJ/K (heat capacity of the coolant): https://www.sciencedirect.com/science/article/am/pii/S0306454917301391
    "mu_f": .0263, # GJ/K (heat capacity of the fuel)
    
    "gamma_X": 0.003,  # Fission yield of xenon
    "gamma_I": 0.059,  # Fission yield of iodine
    
    "sigma_X": 3.5e-12,  # m^2 (microscopic absorption cross-section of xenon)
    "Sigma_f": 0.3358,   # s^-1 (effective microscopic fission cross-section)
    
    "epsilon_f": 0.92,  # Fraction of reactor power deposited in the fuel
    
    "P0":    3,  # GW

    # the following parameters were not specified in the paper and are therefore assumed to be normalized
    "theta": 1, 
    "mu_c":  1,
    "nu":    1,
    "Omega": 1,
    "M":     1,
    "phi0":  1,
    "X0":    1,
}

fusion_std_devs = {}
for key, value in fusion_means.items():
    if value < 1e2:
        fusion_std_devs[key] = np.abs(value)
    else:
        fusion_std_devs[key] = np.sqrt(np.abs(value))

fusion_std_scale = 1.0
fusion_std_devs = {key: value / fusion_std_scale for key, value in fusion_std_devs.items()}

def fusion_generate_dynamics_matrices(num_samples):
    # Define constants
    alpha_c = np.expand_dims(np.random.normal(fusion_means["alpha_c"], fusion_std_devs["alpha_c"], num_samples), axis=0)    # αc
    alpha_f = np.expand_dims(np.random.normal(fusion_means["alpha_f"], fusion_std_devs["alpha_f"], num_samples), axis=0)    # αf
    
    beta    = np.expand_dims(np.random.normal(fusion_means["beta"], fusion_std_devs["beta"], num_samples), axis=0)             # β
    beta1   = np.expand_dims(np.random.normal(fusion_means["beta1"], fusion_std_devs["beta1"], num_samples), axis=0)          # β1
    beta2   = np.expand_dims(np.random.normal(fusion_means["beta2"], fusion_std_devs["beta2"], num_samples), axis=0)          # β2
    beta3   = np.expand_dims(np.random.normal(fusion_means["beta3"], fusion_std_devs["beta3"], num_samples), axis=0)          # β3
    
    Lambda   = np.expand_dims(np.random.normal(fusion_means["Lambda"], fusion_std_devs["Lambda"], num_samples), axis=0)       # Λ
    lambda_I = np.expand_dims(np.random.normal(fusion_means["lambda_I"], fusion_std_devs["lambda_I"], num_samples), axis=0) # λI
    lambda_X = np.expand_dims(np.random.normal(fusion_means["lambda_X"], fusion_std_devs["lambda_X"], num_samples), axis=0) # λx
    lambda1  = np.expand_dims(np.random.normal(fusion_means["lambda1"], fusion_std_devs["lambda1"], num_samples), axis=0)    # λ1
    lambda2  = np.expand_dims(np.random.normal(fusion_means["lambda2"], fusion_std_devs["lambda2"], num_samples), axis=0)    # λ2
    lambda3  = np.expand_dims(np.random.normal(fusion_means["lambda3"], fusion_std_devs["lambda3"], num_samples), axis=0)    # λ3
    
    mu_f     = np.expand_dims(np.random.normal(fusion_means["mu_f"], fusion_std_devs["mu_f"], num_samples), axis=0)             # μf
    mu_c     = np.expand_dims(np.random.normal(fusion_means["mu_c"], fusion_std_devs["mu_c"], num_samples), axis=0)             # μc
    
    gamma_X  = np.expand_dims(np.random.normal(fusion_means["gamma_X"], fusion_std_devs["gamma_X"], num_samples), axis=0)    # γx
    gamma_I  = np.expand_dims(np.random.normal(fusion_means["gamma_I"], fusion_std_devs["gamma_I"], num_samples), axis=0)    # γI

    sigma_X  = np.expand_dims(np.random.normal(fusion_means["sigma_X"], fusion_std_devs["sigma_X"], num_samples), axis=0)    # σx
    Sigma_f  = np.expand_dims(np.random.normal(fusion_means["Sigma_f"], fusion_std_devs["Sigma_f"], num_samples), axis=0)    # Σf
    
    nu        = np.expand_dims(np.random.normal(fusion_means["nu"], fusion_std_devs["nu"], num_samples), axis=0)                       # ν
    epsilon_f = np.expand_dims(np.random.normal(fusion_means["epsilon_f"], fusion_std_devs["epsilon_f"], num_samples), axis=0)  # εf
    Omega     = np.expand_dims(np.random.normal(fusion_means["Omega"], fusion_std_devs["Omega"], num_samples), axis=0)              # Ω
    M         = np.expand_dims(np.random.normal(fusion_means["M"], fusion_std_devs["M"], num_samples), axis=0)                          # M
    theta     = np.expand_dims(np.random.normal(fusion_means["theta"], fusion_std_devs["theta"], num_samples), axis=0)              # θ

    P0        = np.expand_dims(np.random.normal(fusion_means["P0"], fusion_std_devs["P0"], num_samples), axis=0)                       # P0
    phi0      = np.expand_dims(np.random.normal(fusion_means["phi0"], fusion_std_devs["phi0"], num_samples), axis=0)                 # φ0
    X0        = np.expand_dims(np.random.normal(fusion_means["X0"], fusion_std_devs["X0"], num_samples), axis=0)                       # X0
    
    thetas = np.vstack([alpha_c, alpha_f, beta, beta1, beta2, beta3, Lambda, lambda_I, lambda_X, lambda1, lambda2, lambda3, mu_f, mu_c, gamma_X, gamma_I, sigma_X, Sigma_f, nu, epsilon_f, Omega, M, theta, P0, phi0, X0]).T

    As = np.zeros((num_samples, 8, 8))
    Bs = np.zeros((num_samples, 8, 1))

    As[:,0,0] = -beta / Lambda
    As[:,0,1] = beta1 / Lambda
    As[:,0,2] = beta2 / Lambda
    As[:,0,3] = beta3 / Lambda
    As[:,0,4] = alpha_f * theta / Lambda
    As[:,0,5] = alpha_c * theta / (2 * Lambda)
    As[:,0,6] = -sigma_X * theta / (nu * Sigma_f * Lambda)

    As[:,1,0] = lambda1
    As[:,1,1] = -lambda1

    As[:,2,0] = lambda2
    As[:,2,2] = -lambda2

    As[:,3,0] = lambda3
    As[:,3,3] = -lambda3

    As[:,4,0] = epsilon_f * P0 / mu_f
    As[:,4,4] = -Omega / mu_f
    As[:,4,5] = Omega / mu_f

    As[:,5,0] = (1 - epsilon_f) * P0 / mu_c
    As[:,5,4] = Omega / mu_f
    As[:,5,5] = -(2 * M + Omega) / (2 * mu_c)

    As[:,6,0] = (gamma_X * Sigma_f - sigma_X * X0) * phi0 * P0
    As[:,6,6] = -(lambda_X + phi0 * P0 * theta)
    As[:,6,7] = lambda_I

    As[:,7,0] = gamma_I * Sigma_f * phi0 * P0
    As[:,7,7] = -lambda_I

    Bs[:,0,0] = theta / Lambda

    # HACK: normalize dynamics to avoid numerical instability
    maxes = einops.reduce(As, "n h w -> n", "max")
    As /= np.expand_dims(np.expand_dims(maxes, axis=-1), axis=-1)
    return thetas, (As, Bs)

# m: trajectory length
def generate_system_trajectories(As, Bs, m = 25):
    n = As.shape[-1]
    l = Bs.shape[-1]

    x0 = np.random.random((n, 1))
    u = np.random.rand(l, m - 1) - .5

    x0 = np.tile(x0, reps=(As.shape[0],1,1)).astype(np.float32)
    u  = np.tile(u,  reps=(Bs.shape[0],1,1)).astype(np.float32)

    snapshots = [x0]

    for i in range(m - 1):
        snapshots.append(As @ snapshots[i] + Bs @ u[:, :, i:i+1])
    snapshots = np.array(snapshots).T
    return {'snapshots': snapshots, 'u': u, 'B': Bs, 'A': As}

def estimate_dynamics_matrices(system):
    mb_size = system["A"].shape[0]
    A_hats, B_hats = [], []
    for i in range(mb_size):
        x_data = system['snapshots'][0,:,i,:]
        u_data = system['u'][i]

        X = x_data[:, :-1]        # shape (n, T)
        X_next = x_data[:, 1:]    # shape (n, T)
        U = u_data                # shape (m, T)

        Z = np.vstack([X, U])     # shape (n + m, T)

        # Solve: X_next ≈ [A B] @ [X; U]
        AB = X_next @ np.linalg.pinv(Z)  # shape (n, n + m)

        n = x_data.shape[0]
        A_hat = AB[:, :n]
        B_hat = AB[:, n:]

        A_hats.append(A_hat)
        B_hats.append(B_hat)

    A_hats = np.array(A_hats).reshape(mb_size, -1)
    B_hats = np.array(B_hats).reshape(mb_size, -1)
    return A_hats, B_hats

def generate_data(generate_dynamics_matrices, n_pts):
    thetas, (As, Bs) = generate_dynamics_matrices(n_pts)

    system = generate_system_trajectories(As, Bs, m = 25)
    A_hats, B_hats = estimate_dynamics_matrices(system)
    A_hats, B_hats = A_hats.reshape(As.shape), B_hats.reshape(Bs.shape)

    # HACK: sometimes DMD seems to produce egregiously terrible results: not sure why this happens? ignore those cases for now
    thresh    = 0.1
    valid_ind = np.where(np.logical_and(
        np.linalg.norm(A_hats - As, ord="fro", axis=(1,2)) < thresh,
        np.linalg.norm(B_hats - Bs, ord="fro", axis=(1,2)) < thresh,
    ))
    thetas, As, Bs, A_hats, B_hats = thetas[valid_ind], As[valid_ind], Bs[valid_ind], A_hats[valid_ind], B_hats[valid_ind]

    thetas         = torch.from_numpy(thetas).to(torch.float32).to(device)
    As, Bs         = torch.from_numpy(As).to(torch.float32).to(device), torch.from_numpy(Bs).to(torch.float32).to(device)
    A_hats, B_hats = torch.from_numpy(A_hats).to(torch.float32).to(device), torch.from_numpy(B_hats).to(torch.float32).to(device)

    return thetas, (As, Bs), (A_hats, B_hats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup")
    args = parser.parse_args()
    setup = args.setup

    setup_to_generate_func = {
        "airfoil": airfoil_generate_dynamics_matrices,
        "load_pos": load_pos_generate_dynamics_matrices,
        "pendulum": pendulum_generate_dynamics_matrices,
        "battery": battery_generate_dynamics_matrices,
        "fusion": fusion_generate_dynamics_matrices,
    }

    # HACK: not sure why, but load pos setup results in poorer A_hat/B_hat estimation?
    gen_train, gen_test = (2_000, 1_000) if setup != "load_pos" else (20_000, 25_000)

    thetas, _, (A_hats, B_hats) = generate_data(setup_to_generate_func[setup], gen_train)
    N_train, N_cal = (int(len(thetas) * 0.80), int(len(thetas) * 0.20))
    thetas_test, (As_test, Bs_test), _ = generate_data(setup_to_generate_func[setup], gen_test)

    thetas_train, thetas_cal = thetas[:N_train], thetas[N_train:N_train+N_cal]
    A_hats_train, A_hats_cal = A_hats[:N_train], A_hats[N_train:N_train+N_cal]
    B_hats_train, B_hats_cal = B_hats[:N_train], B_hats[N_train:N_train+N_cal]

    os.makedirs(os.path.join("data", setup), exist_ok=True)
    
    # for training and calibration, we only have access to the DMD-estimated dynamics
    with open(os.path.join("data", setup, "train.pkl"), "wb") as f:
        pickle.dump((thetas_train, (A_hats_train, B_hats_train)), f)

    with open(os.path.join("data", setup, "cal.pkl"), "wb") as f:
        pickle.dump((thetas_cal, (A_hats_cal, B_hats_cal)), f)

    # for testing, we will only use theta *but* we test for coverage on the *true* dynamics
    with open(os.path.join("data", setup, "test.pkl"), "wb") as f:
        pickle.dump((thetas_test, (As_test, Bs_test)), f)