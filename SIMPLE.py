import numpy as np
import matplotlib.pyplot as plt

# Domain and fluid properties
L = 1.0
Nx, Ny = 60, 60
dx, dy = L / (Nx - 1), L / (Ny - 1)
vis = 0.01  # Kinematic viscosity
Re = 200
conv_crit, max_iter = 1e-6, 300
alpha_v, alpha_p = 0.3, 0.3

# Coefficients for ADI
a_E = vis / dx**2
a_W = vis / dx**2
a_N = vis / dy**2
a_S = vis / dy**2
a_P = a_E + a_W + a_N + a_S
Aw, Ae, As, An = -1 / dx**2, -1 / dx**2, -1 / dy**2, -1 / dy**2
Ap = 2 / dx**2 + 2 / dy**2

# Initialize fields
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))
p = np.zeros((Nx, Ny))
b = np.zeros((Nx, Ny))

# Boundary conditions for lid-driven cavity
u[0, :] = v[0, :] = 0  # Bottom wall
u[Nx - 1, :] = v[Nx - 1, :] = 0  # Top wall
u[:, 0] = v[:, 0] = 0  # Left wall
u[:, Ny - 1] = 1  # Top lid velocity
v[:, Ny - 1] = 0  # No vertical velocity at the lid


# Define TDMA for ADI solving
def tdma(a, b, c, d):
    n = len(d)
    c_new = np.zeros(n)
    d_new = np.zeros(n)
    c_new[0] = c[0] / b[0]
    d_new[0] = d[0] / b[0]
    for i in range(1, n):
        factor = b[i] - a[i] * c_new[i - 1]
        c_new[i] = c[i] / factor
        d_new[i] = (d[i] - a[i] * d_new[i - 1]) / factor
    x = np.zeros(n)
    x[-1] = d_new[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_new[i] - c_new[i] * x[i + 1]
    return x


# Define ADI functions for u, v, and pressure corrections
def ADI_u(u, v, p, dx, dy, a_E, a_W, a_N, a_S, a_P):
    u_new = u.copy()
    for j in range(1, Ny - 1):
        a = np.ones(Nx - 2) * a_W
        b = np.ones(Nx - 2) * -a_P
        c = np.ones(Nx - 2) * a_E
        d = -(p[1:-1, j] - p[2:, j]) / (2 * dx) + a_S * u_new[1:-1, j - 1] + a_N * u_new[1:-1, j + 1]
        u_new[1:-1, j] = tdma(a, b, c, d)
    return u_new


def ADI_v(u, v, p, dx, dy, a_E, a_W, a_N, a_S, a_P):
    v_new = v.copy()
    for i in range(1, Nx - 1):
        a = np.ones(Ny - 2) * a_S
        b = np.ones(Ny - 2) * -a_P
        c = np.ones(Ny - 2) * a_N
        d = -(p[i, 1:-1] - p[i + 1, 1:-1]) / (2 * dy) + a_W * v_new[i - 1, 1:-1] + a_E * v_new[i + 1, 1:-1]
        v_new[i, 1:-1] = tdma(a, b, c, d)
    return v_new


def ADI_p(b, Aw, Ae, As, An, Ap):
    p_correction = np.zeros_like(b)
    for j in range(1, Ny - 1):
        a = np.ones(Nx - 2) * Aw
        b_diag = np.ones(Nx - 2) * -Ap
        c = np.ones(Nx - 2) * Ae
        d = b[1:-1, j] - (As * p_correction[1:-1, j - 1] + An * p_correction[1:-1, j + 1])
        p_correction[1:-1, j] = tdma(a, b_diag, c, d)
    return p_correction


# SIMPLE loop for velocity and pressure correction
max_b = float('inf')
iteration = 0

while max_b > 1e-10 and iteration < max_iter:
    # Compute intermediate (star) velocities
    u_star = ADI_u(u, v, p, dx, dy, a_E, a_W, a_N, a_S, a_P)
    v_star = ADI_v(u, v, p, dx, dy, a_E, a_W, a_N, a_S, a_P)

    # Calculate pressure corrections
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            b[i, j] = ((u_star[i + 1, j] - u_star[i, j]) / dx + (v_star[i, j + 1] - v_star[i, j]) / dy)

    max_b = np.max(np.abs(b))  # Use maximum absolute value of b as the criterion

    if max_b < 1e-10:  # Early termination if b is very small
        print("Source term b is very small. Ending iteration.")
        break

    p_correction = ADI_p(b, Aw, Ae, As, An, Ap)

    # Update pressure and apply under-relaxation
    p += alpha_p * p_correction

    # Correct velocities
    u = u_star - (alpha_v / a_P) * (np.roll(p_correction, -1, axis=0) - p_correction) / dx
    v = v_star - (alpha_v / a_P) * (np.roll(p_correction, -1, axis=1) - p_correction) / dy

    iteration += 1
    print(f"Iteration {iteration}: Maximum b = {max_b}")

if iteration >= max_iter:
    print("Maximum iterations reached without full convergence.")

# Visualization
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

# Compute velocity magnitude
velocity_magnitude = np.sqrt(u**2 + v**2)

# Velocity Magnitude Contour Plot
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, velocity_magnitude.T, 20, cmap='viridis')
plt.title("Velocity Magnitude Contours")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Velocity Magnitude")
plt.show()
