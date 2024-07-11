# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(
  distributed_ranges_tutorial
  LANGUAGES CXX
  VERSION 0.1
  DESCRIPTION "Distributed ranges tutorial")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fsycl COMPILER_SUPPORTS_FSYCL)

if(NOT COMPILER_SUPPORTS_FSYCL)
  message(
    FATAL_ERROR
      "A compiler with SYCL support is required. Configure Intel(R) oneAPI DPC++/C++ or any other compiler with SYCL support in your system."
  )
endif()

find_package(MPI REQUIRED)

add_subdirectory(src)

option(ENABLE_CUDA "Build for cuda" OFF)
# required by distributed-ranges
option(ENABLE_FORMAT "Build with format library" ON)

include(FetchContent)

FetchContent_Declare(
  distributed-ranges
  GIT_REPOSITORY https://github.com/oneapi-src/distributed-ranges.git
  GIT_TAG main)
FetchContent_MakeAvailable(distributed-ranges)

FetchContent_Declare(
  cpp-format
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG master)
FetchContent_MakeAvailable(cpp-format)
