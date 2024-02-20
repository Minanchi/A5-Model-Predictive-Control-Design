import numpy as np
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix, kron, eye, hstack, vstack, block_diag

# System dynamics
A = np.array([[0.9835, 2.782, 0], [-0.0006821, 0.978, 0], [-0.0009730, 2.804, 1]])
B = np.array([[0.01293], [0.00100], [0.001425]])

# MPC parameters
N = 10  # Prediction horizon
Q = np.eye(3)  # State weighting matrix
R = np.eye(1)  # Input weighting matrix
DeltaR = np.eye(1) * 0.1  # Input rate change weighting matrix

# Constraint parameters
alpha_max = np.deg2rad(11.5)  # Maximum angle of attack in radians
delta_min, delta_max = np.deg2rad(-24), np.deg2rad(27)  # Elevator deflection angle bounds in radians
theta_max = np.deg2rad(35)  # Maximum pitch angle in radians
q_max = np.deg2rad(14)  # Maximum pitch rate in radians per second

def construct_qp_matrices(A, B, Q, R, DeltaR, N):
    n = A.shape[0]  # Number of states
    m = B.shape[1]  # Number of inputs
    # Constructing block diagonal matrices for the cost function
    P = block_diag([kron(eye(N), Q), kron(eye(N), R)])
    q = np.zeros(N*(n+m))
    return csc_matrix(P), q

def construct_constraint_matrices(A, B, N, alpha_max, delta_min, delta_max, theta_max, q_max):
    n = A.shape[0]  # Number of states
    m = B.shape[1]  # Number of inputs
    total_vars = N * (n + m)  # Total number of variables (states and inputs) over the horizon

    # Assuming 4 state constraints per timestep and 2 input constraints per timestep
    num_constraints = N * 4 + N * 2  # Modify based on actual constraint counts
    G = np.zeros((num_constraints, total_vars))

    # State constraints for alpha, q, and theta
    for k in range(N):
        base_row = k * 6  # Adjust row index based on actual constraints
        G[base_row, k*(n+m)] = 1  # alpha <= alpha_max
        G[base_row + 1, k*(n+m)] = -1  # -alpha <= alpha_max
        G[base_row + 2, k*(n+m) + 2] = 1  # theta <= theta_max
        G[base_row + 3, k*(n+m) + 2] = -1  # -theta <= theta_max
        
        # Input constraints for delta
        delta_idx = N*n + k*m  # Index for delta variable
        G[base_row + 4, delta_idx] = 1  # delta <= delta_max
        G[base_row + 5, delta_idx] = -1  # -delta <= -delta_min

    h = np.zeros(num_constraints)
    h[::6] = alpha_max
    h[1::6] = alpha_max
    h[2::6] = theta_max
    h[3::6] = theta_max
    h[4::6] = delta_max
    h[5::6] = -delta_min

    # Convert G to sparse matrix for performance
    G = csc_matrix(G)
    return G, h

def extract_control_sequence(x_opt, N, n, m):
    return x_opt[-N*m:].reshape(N, m)

# Initial state
x0 = np.array([0.1, 0.1, 0.1])

# Construct QP matrices
P, q = construct_qp_matrices(A, B, Q, R, DeltaR, N)
G, h = construct_constraint_matrices(A, B, N, alpha_max, delta_min, delta_max, theta_max, q_max)

# Solve QP problem
x_opt = solve_qp(P, q, G, h, solver="osqp")

# Extract optimal control sequence
u_opt = extract_control_sequence(x_opt, N, A.shape[0], B.shape[1])

print("Optimal control sequence (delta):", u_opt)
