# Distributed-ranges tutorial

## Introduction

The distributed-ranges (dr) library is a C++20 library for multi-CPU and multi-GPU computing environments. It provides algorithms, data structures and views tailored to use in multi-node HPC systems and servers with many CPUs and/or GPUs. It takes advantage of parallel processing and MPI communication in distributed memory model as well as parallel processing in shared memory model with many GPUs.
The library is designed as replacement for chosen data structures, containers, and algorithms of the C++20 Standard Template Library. If you are familiar with the C++ Template Libraries, and in particular std::ranges (C++20) or ranges-v3 (C++11 -- C++17), switching to dr will be straightforward, but this tutorial will help you get started even if you have never used them. However, we assume that you are familiar with C++, at least in the C++11 standard (C++20 is recommended).

## Getting started

### Prerequisites

The distributed-ranges library can be used on any system with a working SYCL or g++ compiler. _Intel's DPC++ is recommended, and it is required by this tutorial_. g++ v. 10, 11 or 12 is also supported, but GPU usage is not possible.
Distributed-ranges depends on MPI and oneDPL libraries. DPC++, oneDPL and oneMPI are part of the [oneAPI](whttps://www.oneapi.io/) - open-standards based industry initiative. OneAPI and the associated [Intel® oneAPI Toolkits and products](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html), help to provide a unified approach to mixed-architecture offload computing. Its approach also ensures interoperability with existing distributed computing standards. It is recommended to install oneAPI components before downloading distributed-ranges.

### First steps

Currently, there are two ways to start work with distributed-ranges.

#### Users

If you want to use dr in your application, and your development environment is connected to the Internet, we encourage you to clone the [distributed-ranges-tutorial repository](https://github.com/intel/distributed-ranges-tutorial) and modify examples provided. The cmake files provided in the skeleton repo will download the dr library as a source code and build the examples, there is no need for separate install.

In Linux system (bash shell) download distributed-ranges-tutorial from GitHub and build with the following commands

 ```shell
 git clone https://github.com/mateuszpn/distributed-ranges-tutorial
 cd distributed-ranges-tutorial
 CXX=icpx CC=icx cmake -B build
 cmake --build build
 mpirun -n N ./build/src/example_name
 ```

If you have a compiler different than DPC++, change CXX and CC values respectively.
Modify the call of mpirun, replacing N with number of mpi processes you want to start, and _example_name_ with an actual example name.

Now you can:

- modify provided examples
- add new source files, modifying src/CMakeList.txt accordingly
- start a new project, using the tutorial as a template

In case your environment is not configured properly or you just prefer a hassle-free code exploration you can use Docker.

 ```shell
 git clone https://github.com/mateuszpn/distributed-ranges-tutorial
 cd distributed-ranges-tutorial
 docker run -it -v $(pwd):/custom-directory-name -u root docker.io/intel/oneapi:latest /bin/bash
 cd custom-directory-name
 CXX=icpx CC=icx cmake -B build -DENABLE_SYCL=ON
 cmake --build build -j
 ```

where 'custom-directory-name' stands for the name of a directory containing local repo data on a docker volume

#### Contributors

If you want to contribute to distributed-ranges or go through more advanced examples, please go to original [distributed-ranges GitHub repository](https://github.com/oneapi-src/distributed-ranges/)

 ```shell
 git clone https://github.com/oneapi-src/distributed-ranges
 cd distributed-ranges
 CXX=icpx CC=icx cmake -B build -DENABLE_SYCL=ON
 cmake --build build -j
 ```

## Distributed-ranges library

The distributed-ranges library provides data-structures, algorithms and views designed to be used in two memory models - distributed memory and shared (common) memory. For distributed memory model, MPI is used as communication library between processes. Both model are able to use SYCL devices (GPUs and multi-core CPUs) for calculations.

Algorithms and data structures are designed to take the user off the need to worry about the technical details of their parallelism. An example would be the definition of a distributed vector in memory of multiple nodes connected using MPI.

```cpp
dr::mhp::distributed_vector<double> dv(N);
```

Such a vector, containing N elements, is automatically distributed among all the nodes involved in the calculation, with individual nodes storing an equal (if possible) amount of data.
Then again, functions such as `for_each()` or `transform()` allow you to perform in parallel operations on each element of a data structure conforming to dr.

In this way, many of the technical details related to the parallel execution of calculations can remain hidden from the user. On the other hand, a programmer aware of the capabilities of the environment in which the application is run has access to the necessary information.

### Namespaces

General namespace used in the library is `dr::`
For program using a single node with shared memory available for multiple CPUs and one or more GPUs, data structures and algoritms from `dr::shp::` namespace are provided.
For distributed memory model, use the `dr::mhp::` namespace.

### Data structures

Content of distributes-ranges' data structures is distributed over available nodes. For example, segments of `dr::mhp::distributed_vector` are located in memory of different nodes (mpi processes). Still, global view of the `distributed_vector` is uniform, with contigous indices.
<!-- TODO: some pictures here -->

#### Halo concept

When implementing an algorithm using a distributed data structure such as `distributed_vector`, its segmented internal structure must be kept in mind. The issue comes up when the algorithm references cells adjacent to the current one, and the local loop reaches the beginning or end of the segment. At this point, the neighboring cells are in the physical memory of another node!
To support this situation, the concept of halo was introduced. A halo is an area into which the contents of the edge elements of a neighboring segment are copied. Also, changes in the halo are copied to cells in the corresponding segment to maintain the consistency of the entire vector.
<!-- TODO: picture here -->

### Algorithms

Follwing algorithms are included in distributed-ranges, both in mhp and shp versions:

```cpp
 copy()
 exclusive_scan()
 fill()
 for_each()
 inclusive_scan()
 iota()
 reduce()
 sort()
 transform()
```

Refer to C++20 documentation for detailed description of how the above functions work.

## Examples

The examples should be compiled with SYCL compiler and run with.

```shell
mpirun -n N ./build/src/example_name
```

where `N` - number of MPI processes. Replace _example_name_ with appropiate name of a file tu run.

### Example 1

[./src/example1.cpp](src/example1.cpp)

The example, performing very simple decoding of encoded string, presents copying data between local and distributed data structures, and a `for_each()` loop performing a lambda on each element of the `distributed_vector<>`. Please note, that the copy operation affects only local vector on the node 0 (the _root_ argument of `copy()` function is 0), and only the node prints the decoded message.

### Example 2

[./src/example2.cpp](src/example2.cpp)

The example shows the distributed nature of dr data structures. The distributed_vector has segments located in each of the nodes performing the example. The nodes introduce themselves at the beginning. You can try different numbers on MPI processes when calling `mpirun`.
`iota()` function is aware what distributed_vector is, and fills the segments accordingly. Then node 0 prints out the general information about the vector, and every node presents size and content of its local part.

### Example 3

[./src/example3.cpp](src/example3.cpp)

 The example simulates the elementary 1-d cellular automaton (ECA). Description of what the automaton is and how it works can be found in [wikipedia](https://en.wikipedia.org/wiki/Elementary_cellular_automaton). Visulisation of the automaton work is available in [ASU team webpage](https://elife-asu.github.io/wss-modules/modules/1-1d-cellular-automata).

 The ECA calculates the new value of a cell using old value of the cell and old values of the cell's neighbors. Therefore a halo of 1-cell width is used, to get access to neighboring cells' values when the loop eaches end of local segment of a vector.
 Additionally, a use of a subrange is presented, and `transform()` function, which puts transformed values of input structure to the output structure, element by element. The transforming function is given as lambda `newvalue`.
 _Please note: after each loop the vector content is printed with `fmt::print()`. The formatter function for `distributed_vector` is rather slow, as it gets the vector element by element, both from local node and remote nodes. You can think about customised, more effective way of results presentation._

<!--
Consider adding one more example:
*Simple 2-D operation - Find a pattern in the randomly filled array*
-->
