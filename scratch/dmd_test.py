import numpy as np
from pydmd import DMD

# Generate synthetic data (replace this with your actual data)
t = np.linspace(0, 10, 100)
X = np.sin(2 * np.pi * t) + 0.2 * np.random.randn(100)  # Replace this with your state trajectory
U = np.random.randn(100)  # Replace this with your control input

# Combine state and control input into a matrix
data_matrix = np.vstack([X, U])
print(data_matrix.shape)

# Split the data matrix into X and Y (state and control)
X_data = data_matrix[:, :-1]
Y_data = data_matrix[:, 1:]

print(X_data.shape)

# Perform DMD
dmd = DMD(svd_rank=-1)
dmd.fit(X_data)

# Retrieve system matrix A
A = dmd.modes @ np.diag(dmd.eigs)

# Construct the control input matrix B
B = dmd.reconstructed_data[:,-1][:, np.newaxis]

# Display the results
print("Estimated Matrix A:")
print(A)
print("\nEstimated Matrix B:")
print(B)
