// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mhp.hpp>
#include <fmt/core.h>

namespace mhp = dr::mhp;
using T = int;

int main(int argc, char **argv) {

  mhp::init(sycl::default_selector_v);
  std::size_t xdim = 9, ydim = 5;

  std::array<std::size_t, 2> extents2d = {xdim, ydim};

  // any array with corresponding dimensions can be used
  mhp::distributed_mdarray<T, 2> a(extents2d);
  mhp::distributed_mdarray<T, 2> b(extents2d);
  mhp::distributed_mdarray<T, 2> c(extents2d);

  // try populating the arrays with any data
  mhp::iota(a, 100);
  mhp::iota(b, 200);

  auto sum_op = [](auto v) {
    auto [in1, in2, out] = v;
    out = in1 + in2;
  };
  mhp::for_each(sum_op, a, b, c);

  if (mhp::rank() == 0) {
    fmt::print("A:\n{}\n", a.mdspan());
    fmt::print("B:\n{}\n", b.mdspan());
    fmt::print("C:\n{}\n", c.mdspan());
  }

  mhp::finalize();

  return 0;
}
