# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

FROM docker.io/intel/oneapi:latest

COPY . .
RUN cmake -B build -DENABLE_SYCL=on \
&& cmake --build build -j \
&& mpirun -n 2 ./build/src/example1 \
&& mpirun -n 2 ./build/src/example2 \
&& mpirun -n 2 ./build/src/example3

CMD ["/bin/bash"]
