#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

export CXX=icpx
export CC=icx

cmake -B build
cmake --build build -j

mpirun -n 3 ./build/src/example1
mpirun -n 3 ./build/src/example2
mpirun -n 3 ./build/src/example3
mpirun -n 3 ./build/src/example4
mpirun -n 3 ./build/src/example5
mpirun -n 3 ./build/src/example6
