# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

add_compile_options(-fsycl)
add_link_options(-fsycl)

if(ENABLE_CUDA)
  add_compile_options(-fsycl-targets=nvptx64-nvidia-cuda
                      -Wno-error=unknown-cuda-version)
  add_link_options(-fsycl-targets=nvptx64-nvidia-cuda
                   -Wno-error=unknown-cuda-version)
endif()

add_executable(example1 example1.cpp)

target_compile_definitions(example1 INTERFACE DR_FORMAT)
target_link_libraries(example1 DR::mpi fmt::fmt)

add_executable(example2 example2.cpp)

target_compile_definitions(example2 INTERFACE DR_FORMAT)
target_link_libraries(example2 DR::mpi fmt::fmt)

add_executable(example3 example3.cpp)

target_compile_definitions(example3 INTERFACE DR_FORMAT)
target_link_libraries(example3 DR::mpi fmt::fmt)

add_executable(example4 example4.cpp)

target_compile_definitions(example4 INTERFACE DR_FORMAT)
target_link_libraries(example4 DR::mpi fmt::fmt)

add_executable(example5 example5.cpp)

target_compile_definitions(example5 INTERFACE DR_FORMAT)
target_link_libraries(example5 DR::mpi fmt::fmt)

add_executable(example6 example6.cpp)

target_compile_definitions(example6 INTERFACE DR_FORMAT)
target_link_libraries(example6 DR::mpi fmt::fmt)
