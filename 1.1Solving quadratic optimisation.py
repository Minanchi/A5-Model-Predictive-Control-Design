import numpy as np
from scipy.sparse import csc_matrix
from qpsolvers import solve_qp

# Define the problem parameters using sparse matrices
P = csc_matrix([[1, 0], [0, 1]])  # P as a 2x2 identity matrix
q = np.array([-1, -1])  #q vector remains a dense array
G = csc_matrix([[-1, 0], [0, -1]]) # G for x >= 0 constraints
h = np.array([0, 0]) #  h vector remains a dense array
A = csc_matrix([[1, 1]])  # A for a sum constraint
b = np.array([1])   #b vector remains a dense array
xmin = np.array([0, 0])  # Lower bounds remain dense
xmax = np.array([1, 1])  # Upper bounds remain dense

# Convert bound constraints to inequality constraints
G_bounds = csc_matrix(np.vstack([-np.eye(2), np.eye(2)]))  # Negative for lower bounds, positive for upper bounds
h_bounds = np.hstack([-xmin, xmax])  # Combine as before, h_bounds remains dense

# Combine G and h with bounds
# For combining sparse matrices, use scipy.sparse.vstack
from scipy.sparse import vstack
G_combined = vstack([G, G_bounds])
h_combined = np.hstack([h, h_bounds])  # h_combined remains a dense array

# Solve the quadratic program using a specific solver
x_opt = solve_qp(P, q, G_combined, h_combined, A, b, solver='osqp')

print("Optimal solution:", x_opt)
