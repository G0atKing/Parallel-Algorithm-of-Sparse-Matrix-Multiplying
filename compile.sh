#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
echo "########## Compiling"
# icpx -fiopenmp -fopenmp-targets=spir64 -lsycl lab/simple.cpp -o bin/simple 
dpcpp -fiopenmp -fopenmp-targets=spir64 -lsycl -mavx2 -mavx512f lab/simple.cpp -o bin/simple 
# mpiicpc -cxx=icpx lab/simple.cpp -fiopenmp -fopenmp-targets=spir64 -o bin/simple 

echo "########## Done"