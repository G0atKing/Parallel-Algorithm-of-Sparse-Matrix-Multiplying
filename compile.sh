#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
echo "########## Compiling"
dpcpp -fiopenmp -fopenmp-targets=spir64 -lsycl -mavx2 -mavx512f DPSPMM.cpp -o bin/DPSPMM
echo "########## Done"