import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

# Define the state-space system matrices
A = np.array([[0, 1], [-1, -1]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Create a state-space system
sys = ctrl.ss(A, B, C, D)

# Define the weighting matrices for the cost function
Q = np.eye(2)  # State cost matrix
R = 1           # Control cost matrix

# Solve the continuous-time LQR problem
K, S, E = ctrl.lqr(sys, Q, R)

# Display the optimal controller gains
print("Optimal Controller Gains (K):", K)

# Simulate the closed-loop system
closed_loop_sys = ctrl.ss(A - B @ K, B, C, D)
time, response = ctrl.step_response(closed_loop_sys)

# Plot the response
plt.plot(time, response)
plt.xlabel('Time')
plt.ylabel('Response')
plt.title('LQR Control System Response')
plt.grid(True)
plt.show()