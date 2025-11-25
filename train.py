import argparse
import cvxpy as cp
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from pydmd import DMDc

device = "cuda"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ContextualLQR(nn.Module):
    def __init__(self, theta_shape, A_shape, B_shape):
        super().__init__()
        self.A_shape = A_shape
        self.B_shape = B_shape
        
        self.fc1 = nn.Linear(np.prod(theta_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        
        self.fc_A = nn.Linear(64, np.prod(self.A_shape))
        self.fc_B = nn.Linear(64, np.prod(self.B_shape))

    def forward(self, theta):
        x = F.relu(self.fc1(theta))
        
        fc2_x = self.fc2(x)
        x     = F.relu(x + fc2_x)

        fc3_x = self.fc3(x)
        x     = F.relu(x + fc3_x)

        A = self.fc_A(x).reshape((-1,) + self.A_shape)
        B = self.fc_B(x).reshape((-1,) + self.B_shape)
        return A, B

def train_net(net, thetas_train, As_train, Bs_train, epochs):
    optimizer = optim.Adam(net.parameters())
    criterion = nn.MSELoss()

    batch_size  = 100
    num_batches = len(thetas_train) // batch_size
    
    losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        rolling_loss = 0
        for i in range(num_batches):
            optimizer.zero_grad()

            theta_batch, (A_batch, B_batch) = thetas_train[i*batch_size:(i+1)*batch_size], (As_train[i*batch_size:(i+1)*batch_size], Bs_train[i*batch_size:(i+1)*batch_size])
            A_hat_batch, B_hat_batch = net(theta_batch)
            
            loss = criterion(A_hat_batch, A_batch) + criterion(B_hat_batch, B_batch)
            
            loss.backward()
            optimizer.step()

            rolling_loss += loss.item()
        losses.append(rolling_loss)
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {rolling_loss}')
    print('Finished Training')

def generate_scores(net, thetas, As, Bs):
    A_hats, B_hats = net(thetas)
    Cs = torch.cat([As, Bs], axis=-1).cpu().detach().numpy()
    C_hats = torch.cat([A_hats, B_hats], axis=-1).cpu().detach().numpy()
    diff = Cs - C_hats
    return (Cs, C_hats), np.linalg.norm(diff, ord=2, axis=(1,2))

def plot_calibration(setup, cal_scores, test_scores):
    sns.set_theme()

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    alphas = np.arange(0, 1, 0.05)
    coverages = []
    for alpha in alphas:
        q = np.quantile(cal_scores, q = 1-alpha)
        coverages.append(np.sum(test_scores < q) / N_test)

    sns.lineplot(x=(1-alphas), y=(1-alphas), linestyle='--')
    sns.lineplot(x=(1-alphas), y=coverages)
    plt.xlabel(r"$\mathrm{Expected\ Coverage} (1-\alpha)$")
    plt.ylabel(r"$\mathrm{Empirical\ Coverage}$")
    plt.title(r"$\mathrm{" + setup + r"}$")

if __name__ == "__main__":
    seed_everything(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--setup")
    args = parser.parse_args()
    setup = args.setup

    # for training and calibration, we only have access to the DMD-estimated dynamics
    with open(os.path.join("data", setup, "train.pkl"), "rb") as f:
        (thetas_train, (As_train, Bs_train)) = pickle.load(f)

    with open(os.path.join("data", setup, "cal.pkl"), "rb") as f:
        (thetas_cal, (As_cal, Bs_cal)) = pickle.load(f)

    # for testing, we will only use theta *but* we test for coverage on the *true* dynamics
    with open(os.path.join("data", setup, "test.pkl"), "rb") as f:
        (thetas_test, (As_test, Bs_test)) = pickle.load(f)

    # difficulty of system ID tasks varies, so adapt epochs accordingly: we want to consider a setting where
    # there is some misspecification (to test robustness), so we do early stopping on the training
    epochs = {
        "airfoil":  25,
        "load_pos": 25,
        "pendulum": 25,
        "battery":  2_000,
        "fusion":   2_000,
    }

    net = ContextualLQR(thetas_train.shape[1:], tuple(As_train.shape[1:]), tuple(Bs_train.shape[1:])).to(device)
    train_net(net, thetas_train, As_train, Bs_train, epochs[setup])

    alpha = 0.05
    _, cal_scores  = generate_scores(net, thetas_cal, As_cal, Bs_cal)
    (test_Cs, test_C_hats), test_scores = generate_scores(net, thetas_test, As_test, Bs_test)
    N_cal, N_test = len(cal_scores), len(test_scores)
    q_hat = np.quantile(cal_scores, q = 1-alpha)

    alphas = np.arange(0.025, 1, 0.025)
    for alpha in alphas:
        print(f"{alpha} -> {np.quantile(cal_scores, q = 1-alpha)}") 

    os.makedirs("experiments", exist_ok=True)
    with open(os.path.join("experiments", f"{setup}.pkl"), "wb") as f:
        pickle.dump({
            "test_C": test_Cs, 
            "test_C_hat": test_C_hats,
            "q_hat": q_hat,
            "cal_scores": cal_scores,
            "test_scores": test_scores,
        }, f)

    plot_titles = {
        "airfoil": r"Airfoil\ Calibration",
        "load_pos": r"Load\ Postioning\ Calibration",
        "pendulum": r"Pendulum\ Calibration",
        "battery": r"Battery\ Calibration",
        "fusion": r"Fusion\ Plant\ Calibration",
    }    
    plot_calibration(plot_titles[setup], cal_scores, test_scores)
    plt.savefig(os.path.join("experiments", f"{setup}.png"))