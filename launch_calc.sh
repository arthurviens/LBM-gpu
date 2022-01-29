#!/bin/bash
conda activate numba2021
# NumBa
echo "Numba runs for nx"
for nx in 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000
do
	python python/lbmFlowAroundCylinder-Numba.py -nx ${nx} -ny 300 --profile > logs/numba/run_nx${nx}_ny300_i2000.log 2>/dev/null
done

echo "Numba runs for ny"
for ny in 200 350 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000
do
	python python/lbmFlowAroundCylinder-Numba.py -nx 420 -ny ${ny} --profile > logs/numba/run_nx420_ny${ny}_i2000.log 2>/dev/null
done

echo "Numba runs for iter"
for iter in 1000 1500 2000 2500 3000 3500 4000 4500 5000
do
	python python/lbmFlowAroundCylinder-Numba.py -nx 420 -ny 300 -i ${iter} --profile > logs/numba/run_nx420_ny300_i${iter}.log 2>/dev/null
done

# For bandwidth purposes
python python/lbmFlowAroundCylinder-Numba.py -nx 5000 -ny 5000 -i 5000 --profile > logs/numba_run_nx5000_ny5000_i5000.log 2>/dev/null
: '
# CuPy
echo "Cupy runs for nx"
for nx in 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000
do
	python python/lbmFlowAroundCylinder-CuPy.py -nx ${nx} -ny 300 --profile > logs/cupy/run_nx${nx}_ny300_i2000.log 2>/dev/null
done

echo "Cupy runs for ny"
for ny in 200 350 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000
do
	python python/lbmFlowAroundCylinder-CuPy.py -nx 420 -ny ${ny} --profile > logs/cupy/run_nx420_ny${ny}_i2000.log 2>/dev/null
done

echo "Cupy runs for iter"
for iter in 1000 1500 2000 2500 3000 3500 4000 4500 5000
do
	python python/lbmFlowAroundCylinder-CuPy.py -nx 420 -ny 300 -i ${iter} --profile > logs/cupy/run_nx420_ny300_i${iter}.log 2>/dev/null
done

# Base
echo "Base runs for nx"
for nx in 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000
do
	python python/lbmFlowAroundCylinder.py -nx ${nx} -ny 300 --profile > logs/base/run_nx${nx}_ny300_i2000.log 2>/dev/null
done

echo "Base runs for ny"
for ny in 200 350 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500 3750 4000
do
	python python/lbmFlowAroundCylinder.py -nx 420 -ny ${ny} --profile > logs/base/run_nx420_ny${ny}_i2000.log 2>/dev/null
done

echo "Base runs for iter"
for iter in 1000 1500 2000 2500 3000 3500 4000 4500 5000
do
	python python/lbmFlowAroundCylinder.py -nx 420 -ny 300 -i ${iter} --profile > logs/base/run_nx420_ny300_i${iter}.log 2>/dev/null
done
'