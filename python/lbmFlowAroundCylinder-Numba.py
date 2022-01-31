#!/usr/bin/env python3
# Copyright (C) 2015 Universite de Geneve, Switzerland
# E-mail contact: jonas.latt@unige.ch
#
# 2D flow around a cylinder
#

#
# Slightly modified from original version
# found here :
#  https://github.com/sidsriv/Simulation-and-modelling-of-natural-processes
#
# The approach used for boundary conditions is link-wise.
# See book : The Lattice Boltzmann Method, Principles and Practice
# https://www.springer.com/gp/book/9783319446479
# section 5.2.4.3
# Please note that Wet-node open boundary conditions is a special case of
# solid boundary (see section 5.3.4)
#

import numpy as np
from numba import cuda
from numpy import format_float_scientific as fs
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import math
import argparse
from Timers import TimersManager

from numba_kernels import *



timers = TimersManager(gpu=True)
timers.add("main")
timers.add("init")
timers.add("equilibrium")
timers.add("collision")
timers.add("streaming")
timers.add("macroscopic")
timers.add("rightwall")
timers.add("leftwall")
timers.add("fin_inflow")
timers.add("bounceback")

###### Flow definition #################################################
maxIter = 2000    # Total number of time iterations.
Re = 150.0          # Reynolds number.
nx, ny = 4200, 300   # Numer of lattice nodes.
ly = ny-1           # Height of the domain in lattice units.
cx, cy, r = nx//4, ny//2, ny//9 # Coordinates of the cylinder.
uLB     = 0.04                  # Velocity in lattice units.
nulb    = uLB*r/Re;             # Viscoscity in lattice units.
omega = 1 / (3*nulb+0.5);    # Relaxation parameter.
save_figures = False
profile = True

# LBM lattice : D2Q9 (numbers are index to v array defined above)
#
# 6   3   0
#  \  |  /
#   \ | /
# 7---4---1
#   / | \
#  /  |  \
# 8   5   2


def main(nx, ny, iter):
    ###### Lattice Constants ###############################################
    v = np.array([ [ 1,  1], [ 1,  0], [ 1, -1], [ 0,  1], [ 0,  0],
                [ 0, -1], [-1,  1], [-1,  0], [-1, -1] ], dtype=np.int32) # 9 vecteurs : 9 directions de déplacement


    t = np.array([ 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36], 
                                dtype=np.float32)

    col1 = np.array([0, 1, 2])
    col2 = np.array([3, 4, 5])
    col3 = np.array([6, 7, 8])
    ###### Function Definitions ############################################
    

    def equilibrium(rho, u):
        """Equilibrium distribution function.
        """
        usqr = 3/2 * (u[0]**2 + u[1]**2)
        feq = np.zeros((9,nx,ny))
        for i in range(9):
            cu = 3 * (v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
            feq[i,:,:] = rho*t[i] * (1 + cu + 0.5*cu**2 - usqr) 
            # feq[i,:,:] : dimension 1 la direction de déplacement de la particule
            #               dimension 2 et 3 : x et y la position
        return feq

    @cuda.jit
    def rightwall_cuda(fin):
        y = cuda.grid(1)
        if y < fin.shape[2]:
            nx = fin.shape[1]
            fin[6, nx-1, y] = fin[6, nx-2, y]
            fin[7, nx-1, y] = fin[7, nx-2, y]
            fin[8, nx-1, y] = fin[8, nx-2, y]


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



    @cuda.jit
    def leftwall_cuda(fin, vel, u_out, rho_out):
        y = cuda.grid(1)
        if y < rho_out.shape[1]:
            ux = vel[0, 0, y]
            u_out[0, 0, y] = ux
            u_out[1, 0, y] = vel[1, 0, y]
            rho_out[0, y] = (1 / (1 - ux)) * (fin[3, 0, y] + fin[4, 0, y] + fin[5, 0, y] + \
                                            2 * (fin[6, 0, y] + fin[7, 0, y] + fin[8, 0, y]))


    @cuda.jit
    def equilibrium_cuda(rho, u, v, t, feq_out):
        x, y = cuda.grid(2)
        if x < feq_out.shape[1] and y < feq_out.shape[2]:
            usqr = 1.5 * (u[0, x, y] * u[0, x, y] + u[1, x, y] * u[1, x, y]) 
            for ipop in range(9):
                cu = 3 * (v[ipop, 0] * u[0, x, y] + v[ipop, 1] * u[1, x, y])
                feq_out[ipop, x, y] = rho[x, y] * t[ipop] * (1 + cu + 0.5 * cu * cu - usqr)



    @cuda.jit
    def fin_inflow_cuda(feq, fin_out):
        y = cuda.grid(1)
        if y < fin_out.shape[2]:
            fin_out[0, 0, y] = feq[0, 0, y] + fin_out[8, 0, y] - feq[8, 0, y]
            fin_out[1, 0, y] = feq[1, 0, y] + fin_out[7, 0, y] - feq[7, 0, y]
            fin_out[2, 0, y] = feq[2, 0, y] + fin_out[6, 0, y] - feq[6, 0, y]



    @cuda.jit
    def collision_cuda(fin, feq, f_out):
        x, y = cuda.grid(2)
        if (x < f_out.shape[1]) and (y < f_out.shape[2]):
            for ipop in range(9):
                f_out[ipop, x, y] = fin[ipop, x, y] - omega * (fin[ipop, x, y] - feq[ipop, x, y])



    @cuda.jit
    def bounceback_cuda(fin, obstacle, f_out):
        x, y = cuda.grid(2)
        if x < obstacle.shape[0] and y < obstacle.shape[1]:
            if obstacle[x, y] == 1:
                for i in range(9):
                    f_out[i, x, y] = fin[8 - i, x, y]



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

    ##### Right Wall ##### 

    rig_threadsperblock = 16
    rig_blockspergrid_y = math.ceil(ny / rig_threadsperblock)
    rig_blockspergrid = (rig_blockspergrid_y)

    ##### Macroscopic ##### 
    mac_threadsperblock = (8, 8)
    mac_blockspergrid_x = math.ceil(nx / mac_threadsperblock[0])
    mac_blockspergrid_y = math.ceil(ny / mac_threadsperblock[1])
    mac_blockspergrid = (mac_blockspergrid_x, mac_blockspergrid_y)

    ##### Left Wall ##### 
    lef_threadsperblock = 16
    lef_blockspergrid_y = math.ceil(ny / lef_threadsperblock)
    lef_blockspergrid = (lef_blockspergrid_y)


    ##### Equilibrium #####
    equ_threadsperblock = (8, 8)
    equ_blockspergrid_x = math.ceil(nx / equ_threadsperblock[0])
    equ_blockspergrid_y = math.ceil(ny / equ_threadsperblock[1])
    equ_blockspergrid   = (equ_blockspergrid_x, equ_blockspergrid_y)


    ##### Fin_Inflow #####
    inf_threadsperblock = 16
    inf_blockspergrid_y = math.ceil(ny / inf_threadsperblock)
    inf_blockspergrid = (inf_blockspergrid_y)


    ##### Collision ##### 
    col_threadsperblock = (8, 8)
    col_blockspergrid_x = math.ceil(nx / col_threadsperblock[0])
    col_blockspergrid_y = math.ceil(ny / col_threadsperblock[1])
    col_blockspergrid   = (col_blockspergrid_x, col_blockspergrid_y)


    ##### BounceBack #####
    bou_threadsperblock = (8, 8)
    bou_blockspergrid_x = math.ceil(nx / bou_threadsperblock[0])
    bou_blockspergrid_y = math.ceil(ny / bou_threadsperblock[1])
    bou_blockspergrid   = (bou_blockspergrid_x, bou_blockspergrid_y)


    ##### Streaming #####
    str_threadsperblock = (8, 8)
    str_blockspergrid_x = math.ceil(nx / str_threadsperblock[0])
    str_blockspergrid_y = math.ceil(ny / str_threadsperblock[1])
    str_blockspergrid   = (str_blockspergrid_x, str_blockspergrid_y)

    # Setup: cylindrical obstacle and velocity inlet with perturbation
    #
    # Creation of a mask with boolean values, defining the shape of the obstacle.
    #
    def obstacle_fun(x, y):
        return (x-cx)**2+(y-cy)**2<r**2

    # Initial velocity profile: 
    # almost zero, with a slight perturbation to trigger the instability.
    def inivel(d, x, y):
        return (1-d) * uLB * (1 + 1e-4*np.sin(y/ly*2*np.pi))

    #############################################################
    def run_profile(v, t, maxIter):
        
        
        timers.get("init").start()
        v_device = cuda.to_device(v)
        t_device = cuda.to_device(t)
        
        # create obstacle mask array from element-wise function
        obstacle = np.fromfunction(obstacle_fun, (nx,ny), dtype=np.float32)
        obstacle_device = cuda.to_device(obstacle)
        
        # initial velocity field vx,vy from element-wise function
        # vel is also used for inflow border condition
        vel = np.fromfunction(inivel, (2,nx,ny), dtype=np.float32)
        vel_device = cuda.to_device(vel)
        
        
        rho = np.ones(shape=(nx, ny), dtype=np.float32)
        rho_device = cuda.to_device(rho)
        
        
        u = np.zeros((2, nx, ny), dtype=np.float32)
        u_device = cuda.to_device(u)
        
        fin = np.zeros((9, nx, ny), dtype=np.float32)
        fin_device = cuda.to_device(fin)
        
        feq = np.zeros((9, nx, ny), dtype=np.float32)
        feq_device = cuda.to_device(feq)
        
        
        fout = np.zeros((9, nx, ny), dtype=np.float32)
        fout_device = cuda.to_device(fout)
        # Initialization of the populations at equilibrium 
        # with the given velocity.
        equilibrium_cuda[equ_blockspergrid, equ_threadsperblock](rho_device, 
            u_device, v_device, t_device, fin_device)
        timers.get("init").end()

        
        ###### Main time loop ########
        for time in range(maxIter):
            # Right wall: outflow condition.
            # we only need here to specify distrib. function for velocities
            # that enter the domain (other that go out, are set by the streaming step)
            
            timers.get("rightwall").start()
            rightwall_cuda[rig_blockspergrid, rig_threadsperblock,
                        timers.get("rightwall").getStream()](fin_device)
            timers.get("rightwall").end()

            # Compute macroscopic variables, density and velocity.
            timers.get("macroscopic").start()
            macroscopic_cuda[mac_blockspergrid, mac_threadsperblock,
                            timers.get("macroscopic").getStream()](fin_device, v_device, rho_device, u_device)
            #rho, u = macroscopic(fin)
            timers.get("macroscopic").end()
            
            # Left wall: inflow condition.
            timers.get("leftwall").start()
            leftwall_cuda[lef_blockspergrid, lef_threadsperblock,
                        timers.get("leftwall").getStream()](fin_device, vel_device, u_device, rho_device)
            timers.get("leftwall").end()
            
            # Compute equilibrium.
            timers.get("equilibrium").start()
            equilibrium_cuda[equ_blockspergrid, equ_threadsperblock,
                            timers.get("equilibrium").getStream()](rho_device, u_device, 
                                                                v_device, t_device, feq_device) 
            #feq = equilibrium(rho, u)
            timers.get("equilibrium").end()
        
            timers.get("fin_inflow").start()
            fin_inflow_cuda[inf_blockspergrid, inf_threadsperblock, 
                            timers.get("fin_inflow").getStream()](feq_device, fin_device)
            #fin[[0,1,2],0,:] = feq[[0,1,2],0,:] + fin[[8,7,6],0,:] - feq[[8,7,6],0,:]
            timers.get("fin_inflow").end()

            # Collision step.
            timers.get("collision").start()
            #fout = fin - omega * (fin - feq) # Noyau de calcul 1
            collision_cuda[col_blockspergrid, col_threadsperblock](fin_device, feq_device, fout_device)
            timers.get("collision").end()
            
            # Bounce-back condition for obstacle.
            # in python language, we "slice" fout by obstacle
            timers.get("bounceback").start()
            bounceback_cuda[bou_blockspergrid, bou_threadsperblock, 
                            timers.get("bounceback").getStream()](fin_device, obstacle_device, fout_device)
            timers.get("bounceback").end()

            # Streaming step.
            timers.get("streaming").start()
            streaming_cuda[str_blockspergrid, str_threadsperblock, 
                        timers.get("streaming").getStream()](fout_device, v_device, fin_device)
            timers.get("streaming").end()


                
    def oneLoop(obstacle, vel, v, t, fin, rho, u, feq, fout):
        rightwall_cuda[rig_blockspergrid, rig_threadsperblock](fin)
        macroscopic_cuda[mac_blockspergrid, mac_threadsperblock](fin, v, rho, u) 
        leftwall_cuda[lef_blockspergrid, lef_threadsperblock](fin, vel, u, rho)
        equilibrium_cuda[equ_blockspergrid, equ_threadsperblock](rho, u, v, t, feq) 
        fin_inflow_cuda[inf_blockspergrid, inf_threadsperblock](feq, fin)
        collision_cuda[col_blockspergrid, col_threadsperblock](fin, feq, fout)
        bounceback_cuda[bou_blockspergrid, bou_threadsperblock](fin, obstacle, fout)
        streaming_cuda[str_blockspergrid, str_threadsperblock](fout, v, fin)


    def run(v, t, maxIter):
        # create obstacle mask array from element-wise function
        
        v_device = cuda.to_device(v)
        t_device = cuda.to_device(t)
        
        # create obstacle mask array from element-wise function
        obstacle = np.fromfunction(obstacle_fun, (nx,ny), dtype=np.float32)
        obstacle_device = cuda.to_device(obstacle)
        
        # initial velocity field vx,vy from element-wise function
        # vel is also used for inflow border condition
        vel = np.fromfunction(inivel, (2,nx,ny), dtype=np.float32)
        vel_device = cuda.to_device(vel)
        
        
        rho = np.ones(shape=(nx, ny), dtype=np.float32)
        rho_device = cuda.to_device(rho)
        
        
        u = np.zeros((2, nx, ny), dtype=np.float32)
        u_device = cuda.to_device(u)
        
        fin = np.zeros((9, nx, ny), dtype=np.float32)
        fin_device = cuda.to_device(fin)
        
        feq = np.zeros((9, nx, ny), dtype=np.float32)
        feq_device = cuda.to_device(feq)
        
        
        fout = np.zeros((9, nx, ny), dtype=np.float32)
        fout_device = cuda.to_device(fout)
        # Initialization of the populations at equilibrium 
        # with the given velocity.
        equilibrium_cuda[equ_blockspergrid, equ_threadsperblock](rho_device, 
            u_device, v_device, t_device, fin_device)
        
        for time in range(maxIter):
            oneLoop(obstacle_device, vel_device, v_device, t_device, fin_device,
                        rho_device, u_device, feq_device, fout_device)
            
            if ((time%100==0) and save_figures):
                plt.clf()
                u_device.copy_to_host(u)
                plt.imshow(np.sqrt(u[0]**2+u[1]**2).transpose(), cmap=cm.Reds)
                plt.show()
                #plt.savefig("figures/vel.{0:04d}.png".format(time//100))

    if args.profile:
        timers.get("main").start()
        run_profile(v, t, iter)
        timers.get("main").end()
    else:
        timers.get("main").start()
        run(v, t, iter)
        timers.get("main").end()
    
    total = np.sum(timers.get("main").getMeasures())
    print(f"Total time : {total:4.2f}s")
    
    if args.profile:
        timers.printInfo()
        timers.printBd(nx, ny, 4)
        timers.printGflops(nx, ny)

            
def parse_all(args):
    if args.i is not None:
        global maxIter
        maxIter = args.i
    if args.nx is not None:
        global nx
        nx = args.nx
    if args.ny is not None:
        global ny
        ny = args.ny
    
    global ly, cx, cy, r, omega
    ly = ny-1           # Height of the domain in lattice units.
    cx, cy, r = nx//4, ny//2, ny//9 # Coordinates of the cylinder.
    omega = 1 / (3*nulb+0.5);    # Relaxation parameter.
    
        


if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser(description='Process options')
    parser.add_argument('-i', type=int, const=2000, nargs="?",
                        help='Number of iterations')
    parser.add_argument('-nx', type=int, const=420, nargs="?",
                        help='Size of grid in x')
    parser.add_argument('-ny', type=int, const=180, nargs="?",
                        help='Size of grid in y')
    parser.add_argument('--profile', action="store_true",
                        help='Wether or not to use timers')
    parser.add_argument('--savefigs', action="store_true",
                        help='Wether or not to save figures')

    args = parser.parse_args()
    
    print(f"Launching computation with profile = {args.profile} and savefigs = {args.savefigs}") 
    save_figures = args.savefigs
    
    parse_all(args)
    
    
    print("Using following parameters")
    print(f"    maxIter = {maxIter}")
    print(f"    nx      = {nx}")
    print(f"    ny      = {ny}")
    
    main(nx, ny, maxIter)
    
