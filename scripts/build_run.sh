#! /bin/bash

# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

export CXX=icpx
export CC=icx

cmake -B build
cmake --build build -j

mpirun -n 2 ./build/src/example1
mpirun -n 2 ./build/src/example2
mpirun -n 2 ./build/src/example3
