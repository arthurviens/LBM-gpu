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
from numba import cuda, vectorize
from numpy import format_float_scientific as fs
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import math
import argparse

from numba_kernels import *


class Timer():
    def __init__(self, name):
        self.name = name
        self.measures = []

    def getName(self):
        return self.name

    def getMeasures(self):
        return self.measures

    def start(self):
        self.start = time.time()

    def end(self):
        self.end = time.time()
        self.measures.append(self.end - self.start)
        del self.start 
        del self.end

class TimersManager():
    def __init__(self):
        self.timers = []

    def add(self, name):
        self.timers.append(Timer(name))

    def get(self, name):
        for t in self.timers:
            if t.getName() == name:
                return t

    def printInfo(self):
        main = self.get("main")
        main_m = np.sum(main.getMeasures())
        not_computed = main_m
                
        for t in self.timers:
            name = t.getName()
            measures = t.getMeasures()
            if name != "main":
                not_computed -= np.sum(measures)
            if len(measures) > 0:
                percent = np.round((np.sum(measures) / main_m)*100, 2)
                print(f"--> Timer '{name:13}' : N = {len(measures):4} | Mean "\
                    f"{fs(np.mean(measures), precision=3):9} +- {fs(np.std(measures), precision=3):9} "\
                    f" | {percent:5}% of total time.")
            else:
                print(f"--> Timer '{name:12}' : N = {len(measures):4}")
        r = np.round((not_computed/main_m)*100, 2)
        print(f"--> Remaining {fs(np.mean(not_computed), precision=3):9}s not monitored represent {r:5}% of total time")

timers = TimersManager()
timers.add("main")
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
nx, ny = 420, 180   # Numer of lattice nodes.
ly = ny-1           # Height of the domain in lattice units.
cx, cy, r = nx//4, ny//2, ny//9 # Coordinates of the cylinder.
uLB     = 0.04                  # Velocity in lattice units.
nulb    = uLB*r/Re;             # Viscoscity in lattice units.
omega = 1 / (3*nulb+0.5);    # Relaxation parameter.
save_figures = False
profile = True

###### Lattice Constants ###############################################
v = cuda.to_device(np.array([ [ 1,  1], [ 1,  0], [ 1, -1], [ 0,  1], [ 0,  0],
               [ 0, -1], [-1,  1], [-1,  0], [-1, -1] ], dtype=np.int32)) # 9 vecteurs : 9 directions de déplacement
v_np = np.empty(shape=v.shape, dtype=v.dtype)
v.copy_to_host(v_np)

t = cuda.to_device(np.array([ 1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36], 
                            dtype=np.float32))
t_np = np.empty(shape=t.shape, dtype=t.dtype)
t.copy_to_host(t_np)

# LBM lattice : D2Q9 (numbers are index to v array defined above)
#
# 6   3   0
#  \  |  /
#   \ | /
# 7---4---1
#   / | \
#  /  |  \
# 8   5   2

###### Function Definitions ############################################
# Lien avec les planches : Avec la fonction de densité de f on peut remonter aux 3 valeurs
# grâce à cette fonction 'macroscopic'
def macroscopic(fin): 
    """Compute macroscopic variables (density, velocity)

    fluid density is 0th moment of distribution functions 
    fluid velocity components are 1st order moments of dist. functions
    """
    rho = np.sum(fin, axis=0)
    u = np.zeros((2, nx, ny))
    for i in range(9):
        u[0,:,:] += v[i,0] * fin[i,:,:]
        u[1,:,:] += v[i,1] * fin[i,:,:]
    u /= rho
    return rho, u

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

#
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
def main_profile():
    # create obstacle mask array from element-wise function
    obstacle = np.fromfunction(obstacle_fun, (nx,ny))
    obstacle_device = cuda.to_device(obstacle)
    
    # initial velocity field vx,vy from element-wise function
    # vel is also used for inflow border condition
    vel = np.fromfunction(inivel, (2,nx,ny)) 
    vel_device = cuda.to_device(vel)
    
    # Initialization of the populations at equilibrium 
    # with the given velocity.
    fin = equilibrium(1, vel)
    fin_device = cuda.to_device(fin)
    
    rho = np.zeros(shape=(fin_device.shape[1], fin_device.shape[2]))
    rho_device = cuda.to_device(rho)
    
    u = np.zeros((2, nx, ny), dtype=np.float32)
    u_device = cuda.to_device(u)
    
    feq = np.zeros_like(fin)
    feq_device = cuda.to_device(feq)
    
    fout = np.zeros_like(fin)
    fout_device = cuda.to_device(fout)
    

    ###### Main time loop ########
    for time in range(maxIter):
        # Right wall: outflow condition.
        # we only need here to specify distrib. function for velocities
        # that enter the domain (other that go out, are set by the streaming step)
        
        timers.get("rightwall").start()
        #fin[col3,nx-1,:] = fin[col3,nx-2,:] 
        rightwall_cuda[rig_blockspergrid, rig_threadsperblock](fin_device)
        timers.get("rightwall").end()

        
        # Compute macroscopic variables, density and velocity.
        timers.get("macroscopic").start()
        macroscopic_cuda[mac_blockspergrid, mac_threadsperblock](fin_device, v, rho_device, u_device) # Timer in func
        #rho, u = macroscopic(fin)
        timers.get("macroscopic").end()
        
        # Left wall: inflow condition.
        timers.get("leftwall").start()
        ########################## NE FONCTIONNE PAS
        #u[:,0,:] = vel[:,0,:]                                             # A remettre pour exécution sur CPU
        #rho[0,:] = 1/(1-u[0,0,:]) * ( np.sum(fin[col2,0,:], axis=0) +     # A remettre pour exécution sur CPU
        #                                2*np.sum(fin[col3,0,:], axis=0) ) # A remettre pour exécution sur CPU
        leftwall_cuda[lef_blockspergrid, lef_threadsperblock](fin_device, vel_device, u_device, rho_device)
        timers.get("leftwall").end()
        

        
        # Compute equilibrium.
        timers.get("equilibrium").start()
        equilibrium_cuda[equ_blockspergrid, equ_threadsperblock](rho_device, u_device, v, t, feq_device) # Timer in func
        #feq = equilibrium(rho, u)
        timers.get("equilibrium").end()
    

    
    
        timers.get("fin_inflow").start()
        fin_inflow_cuda[inf_blockspergrid, inf_threadsperblock](feq_device, fin_device)
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
        bounceback_cuda[bou_blockspergrid, bou_threadsperblock](fin_device, obstacle_device, fout_device)
        #for i in range(9):
        #    fout[i, obstacle] = fin[8-i, obstacle]
        timers.get("bounceback").end()


        # Streaming step.
        timers.get("streaming").start()
        #for i in range(9):
        #    fin[i,:,:] = np.roll(np.roll(fout[i,:,:], v_np[i,0], axis=0),
        #                         v_np[i,1], axis=1 ) # Noyau de calcul 2
        streaming_cuda[str_blockspergrid, str_threadsperblock](fout_device, v, fin_device)
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


def main():
    # create obstacle mask array from element-wise function
    obstacle_device = cuda.to_device(np.fromfunction(obstacle_fun, (nx,ny)))
    
    # initial velocity field vx,vy from element-wise function
    # vel is also used for inflow border condition
    vel = np.fromfunction(inivel, (2,nx,ny)) 
    vel_device = cuda.to_device(vel)
    
    # Initialization of the populations at equilibrium 
    # with the given velocity.
    fin_device = cuda.to_device(equilibrium(1, vel))
    
    rho_device = cuda.to_device(np.zeros(shape=(fin_device.shape[1], fin_device.shape[2])))
    
    u = np.zeros((2, nx, ny), dtype=np.float32)
    u_device = cuda.to_device(u)
    
    feq_device = cuda.to_device(np.zeros_like(fin_device))
    
    fout_device = cuda.to_device(np.zeros_like(fin_device))
    
    for time in range(maxIter):
        oneLoop(obstacle_device, vel_device, v, t, fin_device,
                     rho_device, u_device, feq_device, fout_device)
        
        if ((time%100==0) and save_figures):
            plt.clf()
            u_device.copy_to_host(u)
            plt.imshow(np.sqrt(u[0]**2+u[1]**2).transpose(), cmap=cm.Reds)
            plt.show()
            #plt.savefig("figures/vel.{0:04d}.png".format(time//100))

            
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
    
    
    if args.profile:
        timers.get("main").start()
        main_profile()
        timers.get("main").end()
    else:
        timers.get("main").start()
        main()
        timers.get("main").end()
    
    total = np.sum(timers.get("main").getMeasures())
    print(f"Total time : {total:4.2f}s")
    
    if args.profile:
        timers.printInfo()
