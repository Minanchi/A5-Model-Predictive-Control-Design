import numpy as np

# Define system dynamics
A = np.array([[0.9835000, 2.782, 0],
              [-0.0006821, 0.978, 0],
              [-0.0009730, 2.804, 1]])

B = np.array([[0.01293],
              [0.00100],
              [0.001425]])

# Define constraint parameters
alpha_max = np.deg2rad(11.5)  # Maximum angle of attack in radians
delta_min, delta_max = np.deg2rad(-24), np.deg2rad(27)  # Elevator deflection angle bounds in radians
theta_max = np.deg2rad(35)  # Maximum pitch angle in radians
q_max = np.deg2rad(14)  # Maximum pitch rate in radians per second

def check_constraints(x_next):
    # Extract states
    alpha = x_next[0]
    theta = x_next[1]
    q = x_next[2]
    
    # Check if constraints are satisfied
    constraint_satisfied = np.abs(alpha) <= alpha_max and delta_min <= 0 <= delta_max \
                            and np.abs(theta) <= theta_max and np.abs(q) <= q_max
    
    return constraint_satisfied

def simulate_system(x0, u, A, B, num_steps):
    x_traj = [x0]
    for t in range(num_steps):
        x_next = np.dot(A, x_traj[-1]) + np.dot(B, u[t])
        if not check_constraints(x_next):
            print(f"Constraints violated at timestep {t}")
            break
        x_traj.append(x_next)
    return np.array(x_traj)

# Define extreme points of the set X_N
extreme_points = [
    np.array([-0.20071286, -0.2443461, -0.61086524]),
    np.array([-0.20071286, -0.2443461, 0.61086524]),
    np.array([-0.20071286, 0.2443461, -0.61086524]),
    np.array([-0.20071286, 0.2443461, 0.61086524]),
    np.array([0.20071286, -0.2443461, -0.61086524]),
    np.array([0.20071286, -0.2443461, 0.61086524]),
    np.array([0.20071286, 0.2443461, -0.61086524]),
    np.array([0.20071286, 0.2443461, 0.61086524])
]

# Simulate the system for each extreme point
for x0 in extreme_points:
    print("Initial state:", x0)
    u = np.zeros((10, 1))  # Use zero control inputs for simplicity
    x_traj = simulate_system(x0, u, A, B, num_steps=len(u))
    for t, state in enumerate(x_traj):
        print(f"Next state: {state}")
    print()
