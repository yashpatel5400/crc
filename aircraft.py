import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lsim
from pydmd import DMD

# Number of trajectories to generate
num_trajectories = 5

# Define the true system matrices
A_true = np.array([[0, 1], [-1, -1]])
B_true = np.array([[0], [1]])
C_true = np.array([[1, 0]])
D_true = np.array([[0]])

# Create a true state-space system
sys_true = (A_true, B_true, C_true, D_true)

for i in range(num_trajectories):
    # Generate a random gain matrix Ki
    K_i = np.random.rand(1, 2)

    # Simulate the closed-loop system with the current gain matrix
    closed_loop_sys_i = (A_true - B_true @ K_i, B_true, C_true, D_true)
    time, response, _ = lsim(closed_loop_sys_i, np.random.rand(100), range(100))
    print(response)

    # Use DMD for system identification
    print(response[:, np.newaxis].shape)
    dmd = DMD(svd_rank=2)  # Adjust svd_rank based on system complexity
    dmd.fit(response[:, np.newaxis])

    # Extract eigenvalues and eigenvectors
    eigenvalues = np.log(dmd.eigs) / dmd.dt
    modes = dmd.modes

    # Estimate A and B based on identified modes
    A_identified = np.diag(eigenvalues)
    B_identified = np.linalg.lstsq(modes[:-1], modes[1:], rcond=None)[0]

    # Display the identified state-space matrices
    print(f"\nTrajectory {i+1} - Identified A matrix:", A_identified)
    print(f"Trajectory {i+1} - Identified B matrix:", B_identified)

    # Simulate the identified system and plot the response
    identified_sys_closed_loop = (A_identified, B_identified, C_true, D_true)
    _, identified_response = lsim(identified_sys_closed_loop, np.random.rand(100), range(100))

    # Plot the true and identified responses for comparison
    plt.figure()
    plt.plot(time, response, label='True System')
    plt.plot(time[1:], identified_response, label='Identified System')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title(f'Trajectory {i+1} - True vs Identified System Response')
    plt.legend()
    plt.grid(True)
    plt.show()
