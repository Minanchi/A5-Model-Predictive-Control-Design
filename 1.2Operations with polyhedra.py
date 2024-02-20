import numpy as np
import matplotlib.pyplot as plt
import polytope as pt

# Define the polytope E = {x ∈ R^n : Fx ≤ f}
F = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])  # Coefficients for the inequalities
f = np.array([1, 1, 0, 0])  # Right-hand side values
poly = pt.Polytope(F, f)

# Define another polytope
F2 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
f2 = np.array([0.5, 0.5, -0.2, -0.2])
poly2 = pt.Polytope(F2, f2)

# Check if poly2 is a subset of poly
is_subset = poly2 <= poly
print("Is poly2 a subset of poly?", is_subset)

# Plotting function for a polytope defined by Fx <= f
def plot_polytope(F, f, ax, color='b', linestyle='-'):
    # Generate points for each inequality line and plot
    x = np.linspace(-1, 2, 400)
    for i in range(F.shape[0]):
        a, b = F[i]
        c = f[i]
        if b != 0:
            y = (c - a * x) / b
            valid_mask = (y >= ax.get_ylim()[0]) & (y <= ax.get_ylim()[1])
            ax.plot(x[valid_mask], y[valid_mask], linestyle=linestyle, color=color)
        else:  # Vertical line
            x_const = c / a
            ax.axvline(x=x_const, linestyle=linestyle, color=color)

fig, ax = plt.subplots()
plot_polytope(F, f, ax, 'b', '-')  # Plot the first polytope in blue
plot_polytope(F2, f2, ax, 'r', '--')  # Plot the second polytope in red with dashed lines
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Polytopes Visualization')
plt.grid(True)
plt.show()

