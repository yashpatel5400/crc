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
import seaborn as sns
from pydmd import DMDc

device = "cuda"

# distribution parameters need to be fixed for the simulation
mu_m_B, sigma_m_B = np.random.random(), np.abs(np.random.random())
mu_m_L, sigma_m_L = np.random.random(), np.abs(np.random.random())
mu_d_L, sigma_d_L = np.random.random(), np.abs(np.random.random())
mu_k_B, sigma_k_B = np.random.random(), np.abs(np.random.random())
mu_d_B, sigma_d_B = np.random.random(), np.abs(np.random.random())

def load_pos_generate_dynamics_matrices(num_samples):
    m_B = np.expand_dims(np.abs(np.random.normal(mu_m_B, sigma_m_B, num_samples)),axis=0)
    m_L = np.expand_dims(np.abs(np.random.normal(mu_m_L, sigma_m_L, num_samples)),axis=0)
    d_L = np.expand_dims(np.abs(np.random.normal(mu_d_L, sigma_d_L, num_samples)),axis=0)
    k_B = np.expand_dims(np.abs(np.random.normal(mu_k_B, sigma_k_B, num_samples)),axis=0)
    d_B = np.expand_dims(np.abs(np.random.normal(mu_d_B, sigma_d_B, num_samples)),axis=0)
    
    thetas = np.vstack([m_B, m_L, d_L, k_B, d_B]).T
    As = np.zeros((num_samples, 4, 4))
    As[:,0,1]  = 1

    As[:,1,1] = -d_L / m_L - d_L / m_B 
    As[:,1,2] = k_B / m_B
    As[:,1,3] = d_B / m_B 

    As[:,2,3] = 1

    As[:,3,1] = d_L / m_B
    As[:,3,2] = -k_B / m_B
    As[:,3,3] = -d_B / m_B

    Bs = np.zeros((num_samples, 4, 1))
    Bs[:,1,0] = 1 / m_L + 1 / m_B
    Bs[:,3,0] = -1 / m_B

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

means = {
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

std_devs = {key: value for key, value in means.items()}

def battery_generate_dynamics_matrices(num_samples):
    Vs  = np.expand_dims(np.random.normal(means["Vs"], std_devs["Vs"], num_samples), axis=0)
    Vt  = np.expand_dims(np.random.normal(means["Vt"], std_devs["Vt"], num_samples), axis=0)
    
    S   = np.expand_dims(np.random.normal(means["S"], std_devs["S"], num_samples), axis=0)
    d   = np.expand_dims(np.random.normal(means["d"], std_devs["d"], num_samples), axis=0)
    N   = np.expand_dims(np.random.normal(means["N"], std_devs["N"], num_samples), axis=0)
    
    K2  = np.expand_dims(np.random.normal(means["K2"], std_devs["K2"], num_samples), axis=0)
    K3  = np.expand_dims(np.random.normal(means["K3"], std_devs["K3"], num_samples), axis=0)
    K4  = np.expand_dims(np.random.normal(means["K4"], std_devs["K4"], num_samples), axis=0)
    K5  = np.expand_dims(np.random.normal(means["K5"], std_devs["K5"], num_samples), axis=0)
    
    Cc2 = np.expand_dims(np.random.normal(means["Cc2"], std_devs["Cc2"], num_samples), axis=0)
    Cc3 = np.expand_dims(np.random.normal(means["Cc3"], std_devs["Cc3"], num_samples), axis=0)
    Cc4 = np.expand_dims(np.random.normal(means["Cc4"], std_devs["Cc4"], num_samples), axis=0)
    Cc5 = np.expand_dims(np.random.normal(means["Cc5"], std_devs["Cc5"], num_samples), axis=0)
    
    Ct2 = np.expand_dims(np.random.normal(means["Ct2"], std_devs["Ct2"], num_samples), axis=0)
    Ct3 = np.expand_dims(np.random.normal(means["Ct3"], std_devs["Ct3"], num_samples), axis=0)
    Ct4 = np.expand_dims(np.random.normal(means["Ct4"], std_devs["Ct4"], num_samples), axis=0)
    Ct5 = np.expand_dims(np.random.normal(means["Ct5"], std_devs["Ct5"], num_samples), axis=0)

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
        dmdc = DMDc(svd_rank=-1, opt=True)
        dmdc.fit(system['snapshots'][:,:,i,:], system['u'][i])
        A_hat, B_hat, _ = dmdc.reconstructed_data() # NOTE: the PyDMD reconstructed_data() function was modified to return the dynamics -- this will *not* work by default
        A_hats.append(np.real(A_hat))
        B_hats.append(np.real(B_hat))
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
    }

    thetas, _, (A_hats, B_hats) = generate_data(setup_to_generate_func[setup], 2_000)
    N_train, N_cal = (int(len(thetas) * 0.80), int(len(thetas) * 0.20))
    thetas_test, (As_test, Bs_test), _ = generate_data(setup_to_generate_func[setup], 1_000)

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