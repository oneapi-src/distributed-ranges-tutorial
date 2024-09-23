// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mp.hpp>
#include <fmt/core.h>

int main(int argc, char **argv) {

  dr::mp::init(sycl::default_selector_v);

  fmt::print(
      "Hello, World! Distributed ranges process is running on rank {} / {} on "
      "host {}\n",
      dr::mp::rank(), dr::mp::nprocs(), dr::mp::hostname());

  std::size_t n = 100;

  dr::mp::distributed_vector<int> v(n);
  dr::mp::iota(v, 1);

  if (dr::mp::rank() == 0) {
    auto &&segments = v.segments();
    fmt::print("Created distributed vector of size {} with {} segments.\n",
               v.size(), segments.size());
  }

  fmt::print("Rank {} owns segment of size {} and content {}\n", dr::mp::rank(),
             dr::mp::local_segment(v).size(), dr::mp::local_segment(v));

  dr::mp::finalize();

  return 0;
}
