import time
import numpy as np
from numpy import format_float_scientific as fs
import numba


def generate_memory_table(nx, ny):
    memory_table = {}
    memory_table["rightwall"] = ny * 2 * 3 * 2
    memory_table["macroscopic"] = ny * nx * 9 + 9 * 2 + 2 * nx * ny
    memory_table["leftwall"] = 2 * 2 * ny + ny * 7
    memory_table["equilibrium"] = 2 * ny * nx + 9 * (2 + 2*nx*ny + 1)
    memory_table["fin_inflow"] = ny * 4 * 3
    memory_table["collision"] = 9 * ny * nx * 4
    memory_table["bounceback"] = 9 * 2 * nx * ny
    memory_table["streaming"] = 9 * (2 + 2 * nx * ny)
    return memory_table
    

def generate_calc_table(nx, ny):
    calc_table = {}
    calc_table["rightwall"] = 6 * ny
    calc_table["macroscopic"] = nx * ny * (3 + 7 * 9)
    calc_table["leftwall"] = ny * (1 + 9)
    calc_table["equilibrium"] = nx * ny * (3 + 4 + 9 * 11)
    calc_table["fin_inflow"] = ny * (1 + 3 * 2)
    calc_table["collision"] = nx * ny * (3 + 9 * 3)
    calc_table["bounceback"] = nx * ny * (3 + 1 + 9 * 1)
    calc_table["streaming"] = nx * ny * (3 + 9 * 12)
    return calc_table
    
class TimerGPU():
    def __init__(self, name):
        self.name = name
        self.measures = []

    def getName(self):
        return self.name

    def getMeasures(self):
        return self.measures
    
    def getStream(self):
        return self.stream

    def start(self):
        self.stream = numba.cuda.stream()
        self.start_t = numba.cuda.event()
        self.end_t = numba.cuda.event()
        self.start_t.record(stream = self.stream)

    def end(self):
        self.end_t.record(stream = self.stream)
        self.end_t.wait(stream = self.stream)
        numba.cuda.synchronize()
        self.measures.append(self.start_t.elapsed_time(self.end_t) / 1000)
        del self.start_t
        del self.end_t
        del self.stream
        
class Timer():
    def __init__(self, name):
        self.name = name
        self.measures = []

    def getName(self):
        return self.name

    def getMeasures(self):
        return self.measures

    def start(self):
        self.start_t = time.time()

    def end(self):
        self.end_t = time.time()
        self.measures.append(self.end_t - self.start_t)
        del self.start_t
        del self.end_t
        

class TimersManager():
    def __init__(self, gpu=False):
        self.gpu = gpu
        self.timers = []

    def add(self, name):
        if self.gpu:
            self.timers.append(TimerGPU(name))
        else:
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
        print(f"--> Timer 'remains      ' : N =    1 | Mean {fs(np.mean(not_computed), precision=3):9} | {r:5}% of total time")
        
        
    def printBd(self, nx, ny, size):
        memory_table = generate_memory_table(nx, ny)
        for t in self.timers:
            name = t.getName()
            measures = t.getMeasures()
            if name != "main":
                try:
                    memory_ops = memory_table[name]
                    bytesize = size * memory_ops
                    t = np.mean(measures)
                    GBs = (bytesize * (1e-9)) / t
                    print(f"mem bandwidth {name:13} : {np.round(GBs, 3)} GB/s")
                except KeyError:
                    pass
    
    def printGflops(self, nx, ny):
        calc_table = generate_calc_table(nx, ny)
        for t in self.timers:
            name = t.getName()
            measures = t.getMeasures()
            if name != "main":
                try:
                    calc_ops = calc_table[name]
                    t = np.mean(measures)
                    Gflops = calc_ops * (10 ** (-9)) / t
                    print(f"flops intensity {name:13} : {np.round(Gflops, 3)} Gflops/s")
                except KeyError:
                    pass

        
        
        