import numpy as np
import matplotlib.pyplot as plt

# Domain and fluid properties
L = 1.0
Nx, Ny = 60, 60
dx, dy = L / (Nx - 1), L / (Ny - 1)
vis = 0.01  
Re = 200
conv_crit, max_iter = 1e-6, 300
alpha_v, alpha_p = 0.5, 0.5  

# Coefficients for momentum equations
a_E = vis / dx**2
a_W = vis / dx**2
a_N = vis / dy**2
a_S = vis / dy**2
a_P = 2 * (vis / dx**2 + vis / dy**2) 

# Coefficients for pressure correction equation
Aw = -1 / dx**2
Ae = -1 / dx**2
As = -1 / dy**2
An = -1 / dy**2
Ap = -2 * (1/dx**2 + 1/dy**2)  

# Initialize fields
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))
p = np.zeros((Nx, Ny))
b = np.zeros((Nx, Ny))

# Boundary conditions
u[:, Ny-1] = 1.0  # Top lid velocity
u[0, :] = u[Nx-1, :] = u[:, 0] = 0  # No-slip walls
v[0, :] = v[Nx-1, :] = v[:, 0] = v[:, Ny-1] = 0  # No-slip walls

#--------------------------------------------------------------------------------------------------------------------------------------

# three blocks of code for the two components of velocity, and pressure solved using ADI scheme

def tdma(a, b, c, d):
    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    
    # Forward elimination
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denominator = (b[i] - a[i] * c_prime[i-1])
        c_prime[i] = c[i] / denominator
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denominator
    
    # Back substitution
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    return x

def ADI_u(u, v, p):
    u_new = u.copy()
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            # Computing face velocities
            ue = 0.5 * (u[i, j] + u[i+1, j])
            uw = 0.5 * (u[i-1, j] + u[i, j])
            vn = 0.5 * (v[i, j] + v[i, j+1])
            vs = 0.5 * (v[i, j-1] + v[i, j])
            
            # Computing convective fluxes
            Fe = ue * dy
            Fw = uw * dy
            Fn = vn * dx
            Fs = vs * dx
            
            # Computing coefficients
            ae = a_E + np.maximum(0, -Fe)
            aw = a_W + np.maximum(0, Fw)
            an = a_N + np.maximum(0, -Fn)
            as_ = a_S + np.maximum(0, Fs)
            ap = ae + aw + an + as_
            
            # Source term
            Su = -(p[i+1, j] - p[i, j]) * dy
            
            # Update velocity
            u_new[i, j] = (ae * u[i+1, j] + aw * u[i-1, j] + 
                          an * u[i, j+1] + as_ * u[i, j-1] + Su) / ap
    
    return u_new

def ADI_v(u, v, p):
    v_new = v.copy()
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            # Compute face velocities
            ue = 0.5 * (u[i, j] + u[i+1, j])
            uw = 0.5 * (u[i-1, j] + u[i, j])
            vn = 0.5 * (v[i, j] + v[i, j+1])
            vs = 0.5 * (v[i, j-1] + v[i, j])
            
            # Compute convective fluxes
            Fe = ue * dy
            Fw = uw * dy
            Fn = vn * dx
            Fs = vs * dx
            
            # Compute coefficients
            ae = a_E + np.maximum(0, -Fe)
            aw = a_W + np.maximum(0, Fw)
            an = a_N + np.maximum(0, -Fn)
            as_ = a_S + np.maximum(0, Fs)
            ap = ae + aw + an + as_
            
            # Source term
            Sv = -(p[i, j+1] - p[i, j]) * dx
            
            # Update velocity
            v_new[i, j] = (ae * v[i+1, j] + aw * v[i-1, j] + 
                          an * v[i, j+1] + as_ * v[i, j-1] + Sv) / ap
    
    return v_new

def solve_pressure_correction(b):
    p_corr = np.zeros_like(b)
    for j in range(1, Ny-1):
        a = np.ones(Nx-2) * Aw
        b_diag = np.ones(Nx-2) * Ap
        c = np.ones(Nx-2) * Ae
        d = -b[1:-1, j]
        p_corr[1:-1, j] = tdma(a, b_diag, c, d)
    return p_corr

#-------------------------------------------------------------------------------------------------------------------------------------

# SIMPLE algorithm main loop
residuals = []
iteration = 0

while iteration < max_iter:
    # Store old velocities for convergence check
    u_old = u.copy()
    v_old = v.copy()
    
    # Solving momentum equations
    u_star = ADI_u(u, v, p)
    v_star = ADI_v(u, v, p)
    
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            b[i,j] = (u_star[i-1,j] - u_star[i,j])/dx + (v_star[i,j-1] - v_star[i,j])/dy
    
    # Pressure Correction Equation 
    p_corr = solve_pressure_correction(b)
    
    # Correcting pressure and velocities
    p += alpha_p * p_corr
    
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u[i,j] = u_star[i,j] + alpha_v * (p_corr[i,j] - p_corr[i+1,j])/dx
            v[i,j] = v_star[i,j] + alpha_v * (p_corr[i,j] - p_corr[i,j+1])/dy
    
    residual = np.sqrt(np.sum((u - u_old)**2 + (v - v_old)**2))/(Nx*Ny)
    residuals.append(residual)
    
    # Checking convergence
    if residual < conv_crit:
        print(f"Converged after {iteration} iterations")
        break
        
    iteration += 1
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Residual: {residual:.2e}")

# ------------------------------------------------------------------------------------------------------------------------------

x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(12, 5))

def plot_contours(X, Y, u, v, p):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # X-velocity contours
    c1 = axes[0].contourf(X, Y, u.T, 20, cmap='RdBu_r')
    axes[0].set_title('X-Velocity')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(c1, ax=axes[0])
    
    # Y-velocity contours
    c2 = axes[1].contourf(X, Y, v.T, 20, cmap='RdBu_r')
    axes[1].set_title('Y-Velocity')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(c2, ax=axes[1])
    
    # Pressure contours
    c3 = axes[2].contourf(X, Y, p.T, 20, cmap='RdBu_r')
    axes[2].set_title('Pressure')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(c3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

def plot_centerlines(x, y, u, v, p):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get centerline indices
    mid_x = len(x) // 2
    mid_y = len(y) // 2
    
    # X-velocity along centerlines
    axes[0,0].plot(x, u[:, mid_y])
    axes[0,0].set_title('X-Velocity along y = 0.5')
    axes[0,0].set_xlabel('X')
    axes[0,0].set_ylabel('U')
    axes[0,0].grid(True)
    
    axes[1,0].plot(y, u[mid_x, :])
    axes[1,0].set_title('X-Velocity along x = 0.5')
    axes[1,0].set_xlabel('Y')
    axes[1,0].set_ylabel('U')
    axes[1,0].grid(True)
    
    # Y-velocity along centerlines
    axes[0,1].plot(x, v[:, mid_y])
    axes[0,1].set_title('Y-Velocity along y = 0.5')
    axes[0,1].set_xlabel('X')
    axes[0,1].set_ylabel('V')
    axes[0,1].grid(True)
    
    axes[1,1].plot(y, v[mid_x, :])
    axes[1,1].set_title('Y-Velocity along x = 0.5')
    axes[1,1].set_xlabel('Y')
    axes[1,1].set_ylabel('V')
    axes[1,1].grid(True)
    
    # Pressure along centerlines
    axes[0,2].plot(x, p[:, mid_y])
    axes[0,2].set_title('Pressure along y = 0.5')
    axes[0,2].set_xlabel('X')
    axes[0,2].set_ylabel('P')
    axes[0,2].grid(True)
    
    axes[1,2].plot(y, p[mid_x, :])
    axes[1,2].set_title('Pressure along x = 0.5')
    axes[1,2].set_xlabel('Y')
    axes[1,2].set_ylabel('P')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_streamlines(X, Y, u, v):
    """
    Plot velocity streamlines
    """
    plt.figure(figsize=(8, 8))
    
    # Calculate stream function
    speed = np.sqrt(u**2 + v**2)
    lw = 5 * speed / speed.max()
    
    plt.streamplot(X, Y, u.T, v.T, density=2, linewidth=lw.T, color='b', 
                  arrowsize=1.5, arrowstyle='->')
    plt.title('Velocity Streamlines')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Add boundary indication
    plt.plot([0, 1], [1, 1], 'r-', linewidth=2, label='Moving lid')
    plt.plot([0, 1], [0, 0], 'k-', linewidth=2, label='Wall')
    plt.plot([0, 0], [0, 1], 'k-', linewidth=2)
    plt.plot([1, 1], [0, 1], 'k-', linewidth=2)
    
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Plot convergence history
plt.subplot(122)
plt.semilogy(residuals)
plt.title('Convergence History')
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.grid(True)

plt.tight_layout()
plt.show()

plot_contours(X, Y, u, v, p)

# Plot centerline variations
plot_centerlines(x, y, u, v, p)

# Plot streamlines
plot_streamlines(X, Y, u, v)