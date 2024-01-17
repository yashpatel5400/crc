import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

# Define the state-space system matrices
A = np.array([[0, 1], [-1, -1]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Define the weighting matrices for the cost function
Q = np.eye(2)  # State cost matrix
R = 1           # Control cost matrix

# Number of trajectories to generate
num_trajectories = 5

for i in range(num_trajectories):
    # Generate a random gain matrix K_i
    K_i = np.random.rand(1, 2)

    # Display the current gain matrix
    print(f"\nTrajectory {i+1} - Random Gain Matrix (K_{i+1}):", K_i)

    # Simulate the closed-loop system with the current gain matrix
    closed_loop_sys_i = ctrl.ss(A - B @ K_i, B, C, D)
    time, response = ctrl.step_response(closed_loop_sys_i)

    # Plot the response for the current trajectory
    plt.plot(time, response, label=f'Trajectory {i+1}')

# Plot settings
plt.xlabel('Time')
plt.ylabel('Response')
plt.title('Closed-Loop System Trajectories with Random Gain Matrices')
plt.legend()
plt.grid(True)
plt.savefig('traj.png')