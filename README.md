# Lid-Driven Cavity Flow Simulation using the SIMPLE Algorithm and ADI Scheme

This repository contains a Python implementation of the lid-driven cavity flow simulation using the SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm and an Alternating Direction Implicit (ADI) scheme for solving the momentum and pressure correction equations.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [License](#license)

## Overview
The lid-driven cavity problem is a benchmark computational fluid dynamics (CFD) problem where fluid flow is driven by the motion of one side of a square cavity. The simulation models incompressible, two-dimensional laminar flow within the cavity.

## Features
- Solves Navier-Stokes equations using the SIMPLE algorithm.
- Alternating Direction Implicit (ADI) scheme for numerical stability.
- Visualizations:
  - Contour plots for velocity and pressure.
  - Streamline plots for flow visualization.
  - Centerline plots for velocity and pressure profiles.
  - Convergence history plot.

## Dependencies
This project uses the following Python libraries:
- `numpy`: For numerical computations.
- `matplotlib`: For plotting results.

Install the dependencies using:
```bash
pip install numpy matplotlib
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/<your_username>/<repository_name>.git
   cd <repository_name>
   ```

2. Run the script:
   ```bash
   python lid_driven_cavity.py
   ```

3. The script generates visualizations for the velocity, pressure, and streamline plots. Convergence history is also displayed.

## Implementation Details
- **Domain Discretization**: The cavity is discretized into a uniform grid with `Nx = 60` and `Ny = 60` nodes.
- **Boundary Conditions**:
  - Top lid: Moving with a velocity of 1.0.
  - Other walls: No-slip condition.
- **Numerical Methods**:
  - Momentum equations solved using ADI.
  - Pressure correction equation solved using the TDMA (Tridiagonal Matrix Algorithm).
- **Convergence Criteria**: The simulation stops when the residual drops below `1e-6` or after 300 iterations.

## Results
The script generates the following:
1. **Contour Plots**:
   - X-Velocity, Y-Velocity, and Pressure.
2. **Centerline Profiles**:
   - Velocity and pressure profiles along the vertical and horizontal centerlines.
3. **Streamlines**:
   - Visualization of the flow patterns.
4. **Convergence History**:
   - Residuals vs. Iterations.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Please feel free to contact me at 24250006@iitgn.ac.in for any clarifications. Thanks! 

