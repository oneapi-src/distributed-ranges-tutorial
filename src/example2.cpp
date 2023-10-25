// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mhp.hpp>
#include <fmt/core.h>

namespace mhp = dr::mhp;

int main(int argc, char **argv) {

  mhp::init(sycl::default_selector_v);

  fmt::print(
      "Hello, World! Distributed ranges process is running on rank {} / {} on "
      "host {}\n",
      mhp::rank(), mhp::nprocs(), mhp::hostname());

  std::size_t n = 100;

  mhp::distributed_vector<int> v(n);
  mhp::iota(v, 1);

  if (mhp::rank() == 0) {
    auto &&segments = v.segments();
    fmt::print("Created distributed vector of size {} with {} segments.\n",
               v.size(), segments.size());
  }

  fmt::print("Rank {} owns segment of size {} and content {}\n", mhp::rank(),
             mhp::local_segment(v).size(), mhp::local_segment(v));

  mhp::finalize();

  return 0;
}
