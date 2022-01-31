from numba import cuda
import math

if __name__ == "__main__":
    Re = 150.0          # Reynolds number.
    nx, ny = 420, 180   # Numer of lattice nodes.
    ly = ny-1           # Height of the domain in lattice units.
    cx, cy, r = nx//4, ny//2, ny//9 # Coordinates of the cylinder.
    uLB     = 0.04                  # Velocity in lattice units.
    nulb    = uLB*r/Re;             # Viscoscity in lattice units.
    omega = 1 / (3*nulb+0.5);    # Relaxation parameter.


