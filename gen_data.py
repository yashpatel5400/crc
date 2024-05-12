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