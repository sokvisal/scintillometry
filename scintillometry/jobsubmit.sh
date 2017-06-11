#!/bin/sh
# @ job_name           = test_m_newtoeplitz
# @ job_type           = bluegene
# @ comment            = "n=250, m=150, zero-padded"
# @ error              = $(job_name).$(Host).$(jobid).err
# @ output             = $(job_name).$(Host).$(jobid).out
# @ bg_size            = 64
# @ wall_clock_limit   = 0:30:00
# @ bg_connectivity    = Torus
# @ queue 

source /scratch/s/scinet/nolta/venv-numpy/setup

NP=128
OMP=8 ## Each core has 4 threads. Since RPN = 16, OMP = 4?
RPN=8
n=64
m=64
p=64
nodes=64

source /scratch/s/scinet/nolta/venv-numpy/setup
module purge
module unload mpich2/xl
module load binutils/2.23 bgqgcc/4.8.1 mpich2/gcc-4.8.1 
module load python/2.7.3 
export OMP_NUM_THREADS=${OMP}

cd /scratch/d/djones/sokvisal/new_test_factorize/ # directory of the code

echo "----------------------"
echo "STARTING in directory $PWD"
date
echo "n ${n}, m ${m}, bg ${nodes}, np ${NP}, rpn ${RPN}, omp ${OMP}"
time runjob --np ${NP} --ranks-per-node=${RPN} --envs  HOME=$HOME LD_LIBRARY_PATH=/scinet/bgq/Libraries/HDF5-1.8.12/mpich2-gcc4.8.1//lib:/scinet/bgq/Libraries/fftw-3.3.4-gcc4.8.1/lib:$LD_LIBRARY_PATH PYTHONPATH=/scinet/bgq/tools/Python/python2.7.3-20131205/lib/python2.7/site-packages/ : /scratch/s/scinet/nolta/venv-numpy/bin/python /scratch/d/djones/sokvisal/new_test_factorize/run_real_new.py yty2 30 30 ${n} ${m} ${p} 1

echo "ENDED"

