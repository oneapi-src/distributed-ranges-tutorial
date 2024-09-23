// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mp.hpp>
#include <fmt/core.h>

using T = float;
using MDA = dr::mhp::distributed_mdarray<T, 2>;

/* 2d stencil - simple operation on multi-dimensional array */
int main() {
  dr::mp::init(sycl::default_selector_v);

  std::size_t arr_size = 4;
  std::size_t radius = 1;
  std::array slice_starts{radius, radius};
  std::array slice_ends{arr_size - radius, arr_size - radius};

  auto dist = dr::mp::distribution().halo(radius);
  MDA a({arr_size, arr_size}, dist);
  MDA b({arr_size, arr_size}, dist);
  dr::mp::iota(a, 1);
  dr::mp::iota(b, 1);

  auto in = dr::mp::views::submdspan(a.view(), slice_starts, slice_ends);
  auto out = dr::mp::views::submdspan(b.view(), slice_starts, slice_ends);

  auto mdspan_stencil_op = [](auto &&v) {
    auto [in, out] = v;
    out(0, 0) = (in(-1, 0) + in(0, -1) + in(0, 0) + in(0, 1) + in(1, 0)) / 4;
  };

  dr::mp::halo(a).exchange();
  dr::mp::stencil_for_each(mdspan_stencil_op, in, out);

  if (dr::mp::rank() == 0) {
    fmt::print("a: \n{} \n", a.mdspan());
    fmt::print("b: \n{} \n", b.mdspan());
    fmt::print("in: \n{} \n", in.mdspan());
    fmt::print("out: \n{} \n", out.mdspan());
  }

  dr::mp::finalize();

  return 0;
}
