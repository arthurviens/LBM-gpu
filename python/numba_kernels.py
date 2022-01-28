from numba import cuda
import math

Re = 150.0          # Reynolds number.
nx, ny = 420, 180   # Numer of lattice nodes.
ly = ny-1           # Height of the domain in lattice units.
cx, cy, r = nx//4, ny//2, ny//9 # Coordinates of the cylinder.
uLB     = 0.04                  # Velocity in lattice units.
nulb    = uLB*r/Re;             # Viscoscity in lattice units.
omega = 1 / (3*nulb+0.5);    # Relaxation parameter.


##### Right Wall #####
rig_threadsperblock = 8
rig_blockspergrid_y = math.ceil(ny / rig_threadsperblock)
rig_blockspergrid = (rig_blockspergrid_y)

@cuda.jit
def rightwall_cuda(fin):
    y = cuda.grid(1)
    if y < fin.shape[2]:
        fin[6, nx-1, y] = fin[6, nx-2, y]
        fin[7, nx-1, y] = fin[7, nx-2, y]
        fin[8, nx-1, y] = fin[8, nx-2, y]


##### Macroscopic #####
mac_threadsperblock = (16, 16)
mac_blockspergrid_x = math.ceil(nx / mac_threadsperblock[0])
mac_blockspergrid_y = math.ceil(ny / mac_threadsperblock[1])
mac_blockspergrid = (mac_blockspergrid_x, mac_blockspergrid_y)

@cuda.jit
def macroscopic_cuda(fin, v, rho_out, u_out):
    x, y = cuda.grid(2)
    if x < rho_out.shape[0] and y < rho_out.shape[1]:
        rho_tmp = 0
        ux_tmp = 0
        uy_tmp = 0
        for ipop in range(9):
            f_tmp = fin[ipop, x, y]
            rho_tmp += f_tmp
            ux_tmp += v[ipop, 0] * f_tmp
            uy_tmp += v[ipop, 1] * f_tmp
        rho_out[x, y] = rho_tmp
        u_out[0, x, y] = ux_tmp / rho_tmp
        u_out[1, x, y] = uy_tmp / rho_tmp


##### Left Wall ##### 
lef_threadsperblock = 8
lef_blockspergrid_y = math.ceil(ny / lef_threadsperblock)
lef_blockspergrid = (lef_blockspergrid_y)

@cuda.jit
def leftwall_cuda(fin, vel, u_out, rho_out):
    y = cuda.grid(1)
    if y < rho_out.shape[1]:
        ux = vel[0, 0, y]
        u_out[0, 0, y] = ux
        u_out[1, 0, y] = vel[1, 0, y]
        rho_out[0, y] = (1 / (1 - ux)) * (fin[3, 0, y] + fin[4, 0, y] + fin[5, 0, y] + \
                                        2 * (fin[6, 0, y] + fin[7, 0, y] + fin[8, 0, y]))

##### Equilibrium #####
equ_threadsperblock = (16, 16)
equ_blockspergrid_x = math.ceil(nx / equ_threadsperblock[0])
equ_blockspergrid_y = math.ceil(ny / equ_threadsperblock[1])
equ_blockspergrid   = (equ_blockspergrid_x, equ_blockspergrid_y)

@cuda.jit
def equilibrium_cuda(rho, u, v, t, feq_out):
    x, y = cuda.grid(2)
    if x < rho.shape[0] and y < rho.shape[1]:
        ux = u[0, x, y]
        uy = u[1, x, y]
        usqr = 1.5 * (ux * ux + uy * uy) 
        for ipop in range(9):
            cu = 3 * (v[ipop, 0] * ux + v[ipop, 1] * uy)
            feq_out[ipop, x, y] = rho[x, y] * t[ipop] * (1 + cu + 0.5 * cu * cu - usqr)


##### Fin_Inflow #####
inf_threadsperblock = 8
inf_blockspergrid_y = math.ceil(ny / inf_threadsperblock)
inf_blockspergrid = (inf_blockspergrid_y)

@cuda.jit
def fin_inflow_cuda(feq, fin_out):
    y = cuda.grid(1)
    if y < fin_out.shape[2]:
        fin_out[0, 0, y] = feq[0, 0, y] + fin_out[8, 0, y] - feq[8, 0, y]
        fin_out[1, 0, y] = feq[1, 0, y] + fin_out[7, 0, y] - feq[7, 0, y]
        fin_out[2, 0, y] = feq[2, 0, y] + fin_out[6, 0, y] - feq[6, 0, y]


##### Collision ##### 
col_threadsperblock = (16, 16)
col_blockspergrid_x = math.ceil(nx / col_threadsperblock[0])
col_blockspergrid_y = math.ceil(ny / col_threadsperblock[1])
col_blockspergrid   = (col_blockspergrid_x, col_blockspergrid_y)

@cuda.jit
def collision_cuda(fin, feq, f_out):
    x, y = cuda.grid(2)
    if x < fin.shape[1] and y < fin.shape[2]:
        for ipop in range(9):
            f_out[ipop, x, y] = fin[ipop, x, y] - omega * (fin[ipop, x, y] - feq[ipop, x, y])


##### BounceBack #####
bou_threadsperblock = (16, 16)
bou_blockspergrid_x = math.ceil(nx / bou_threadsperblock[0])
bou_blockspergrid_y = math.ceil(ny / bou_threadsperblock[1])
bou_blockspergrid   = (bou_blockspergrid_x, bou_blockspergrid_y)

@cuda.jit
def bounceback_cuda(fin, obstacle, f_out):
    x, y = cuda.grid(2)
    if x < obstacle.shape[0] and y < obstacle.shape[1]:
        if obstacle[x, y] == 1:
            for i in range(9):
                f_out[i, x, y] = fin[8 - i, x, y]


##### Streaming #####
str_threadsperblock = (16, 16)
str_blockspergrid_x = math.ceil(nx / str_threadsperblock[0])
str_blockspergrid_y = math.ceil(ny / str_threadsperblock[1])
str_blockspergrid   = (str_blockspergrid_x, str_blockspergrid_y)

@cuda.jit
def streaming_cuda(fout, v, fin_out):
    x, y = cuda.grid(2)
    if x < fout.shape[1] and y < fout.shape[2]:
        for ipop in range(9):
            i_out = x - v[ipop, 0]
            if i_out < 0:
                i_out += nx
            if i_out > nx - 1:
                i_out -= nx
            j_out = y - v[ipop, 1]
            if j_out < 0:
                j_out += ny
            if j_out > ny - 1:
                j_out -= ny
            fin_out[ipop, x, y] = fout[ipop, i_out, j_out]