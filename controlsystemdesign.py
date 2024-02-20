import cvxpy as cp
import numpy as np

# Define system dynamics matrices
A = np.array([[0.9835, 2.782, 0],
              [-0.0006821, 0.978, 0],
              [-0.0009730, 2.804, 1]])
B = np.array([[0.01293],
              [0.00100],
              [0.001425]])
n = A.shape[1]  # Number of states
m = B.shape[1]  # Number of inputs

# MPC parameters
N = 10  # Prediction horizon
dt = 0.05  # Sampling time, not used explicitly as dynamics are discrete

# State and input constraints
alpha_max = np.deg2rad(11.5)  # Max angle of attack in radians
delta_min, delta_max = np.deg2rad(-24), np.deg2rad(27)  # Elevator deflection angle limits in radians
theta_max = np.deg2rad(35)  # Max pitch angle in radians
q_max = np.deg2rad(14)  # Max pitch rate in radians per second
gamma_max = np.deg2rad(23)  # Max slope in radians

# Variables (states and inputs over the prediction horizon)
X = cp.Variable((n, N+1))  # State trajectory (alpha, q, theta)
U = cp.Variable((m, N))   # Control input trajectory (delta)

# Objective function and constraints
objective = cp.Minimize(0)  # Objective function (might be updated according to specific requirements)
constraints = []

# Dynamics constraints
for t in range(N):
    constraints += [X[:, t+1] == A@X[:, t] + B@U[:, t]]

# State and input constraints
for t in range(N+1):
    constraints += [cp.abs(X[0, t]) <= alpha_max,
                    cp.abs(X[1, t]) <= q_max,
                    cp.abs(X[2, t]) <= theta_max,
                    cp.abs(X[2, t] - X[0, t]) <= gamma_max]  # Slope constraint

for t in range(N):
    constraints += [U[:, t] >= delta_min,
                    U[:, t] <= delta_max]

# Initial condition (assumed to be zero for simplicity, adjust as needed)
constraints += [X[:, 0] == np.zeros(n)]

# Solve MPC problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Extract optimal control sequence
optimal_delta = U.value[0]

print("Optimal control sequence (delta):", optimal_delta)
