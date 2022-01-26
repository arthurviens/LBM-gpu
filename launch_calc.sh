#!/bin/bash

for nx in 400 500 #600 700 800 900 1000 1500 2000 2500 3000
do
	python python/lbmFlowAroundCylinder-Numba.py -nx ${nx} -ny 300 --profile > logs/numba/run_nx${nx}_ny300_i2000.log
done

for ny in 200 350 #500 750 1000 1250 1500 2000 2500 3000
do
	python python/lbmFlowAroundCylinder-Numba.py -nx 400 -ny ${ny} --profile > logs/numba/run_nx400_ny${ny}_i2000.log
done
