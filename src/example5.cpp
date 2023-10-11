// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mhp.hpp>
#include <fmt/core.h>

namespace mhp = dr::mhp;

using T = uint16_t;
using MDA = dr::mhp::distributed_mdarray<T, 2>;

int main() {
#ifdef SYCL_LANGUAGE_VERSION
  mhp::init(sycl::default_selector_v);
#else
  mhp::init();
#endif
  std::size_t arr_size = 4;
  std::size_t radius = 1;
  std::array slice_starts{radius, radius};
  std::array slice_ends{arr_size - radius, arr_size - radius};

  auto dist = dr::mhp::distribution().halo(radius);
  MDA a({arr_size, arr_size}, dist);
  MDA b({arr_size, arr_size}, dist);
  mhp::iota(a, 1);
  mhp::iota(b, 1);

  auto in = dr::mhp::views::submdspan(a.view(), slice_starts, slice_ends);
  auto out = dr::mhp::views::submdspan(b.view(), slice_starts, slice_ends);
  auto in_array = &a;
  auto out_array = &b;

  auto mdspan_stencil_op = [](auto &&v) {
    auto [in, out] = v;
    out(0, 0) = (in(-1, 0) + in(0, -1) + in(0, 0) + in(0, 1) + in(1, 0)) / 4;
  };

  mhp::halo(*in_array).exchange();
  mhp::stencil_for_each(mdspan_stencil_op, in, out);

  if (mhp::rank() == 0) {
    fmt::print("a: \n{} \n", a.mdspan());
    fmt::print("b: \n{} \n", b.mdspan());
    fmt::print("in: \n{} \n", in.mdspan());
    fmt::print("out: \n{} \n", out.mdspan());
  }

  mhp::finalize();

  return 0;
}
