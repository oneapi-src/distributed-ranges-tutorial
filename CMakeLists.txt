# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(
  distributed_ranges_tutorial
  VERSION 0.1
  DESCRIPTION "Distributed ranges tutorial")



find_package(MPI REQUIRED)

add_subdirectory(src)

option(ENABLE_CUDA "Build for cuda" OFF)
# required by distributed-ranges
option(ENABLE_FORMAT "Build with format library" ON)

include(FetchContent)

FetchContent_Declare(
    distributed-ranges
    GIT_REPOSITORY https://github.com/oneapi-src/distributed-ranges.git
    GIT_TAG c618154a1bceda33e5d61e1536179aeaa11b68f4)
FetchContent_MakeAvailable(distributed-ranges)

FetchContent_Declare(
  cpp-format
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 0b0f7cf)
FetchContent_MakeAvailable(cpp-format)
