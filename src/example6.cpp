// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mp.hpp>
#include <fmt/core.h>

using T = uint16_t;
using MDA = dr::mp::distributed_mdarray<T, 2>;

/* 2D pattern search in a distributed multidimensional (2D) array */
int main() {
  dr::mp::init(sycl::default_selector_v);

  std::size_t arr_size = 7;
  // keep in mind that if you change the pattern size, you have to also change
  // the pattern array
  const std::size_t pattern_size = 2;
  const std::size_t radius = pattern_size - 1;
  std::array slice_starts{radius - 1, radius - 1};
  std::array slice_ends{arr_size - radius, arr_size - radius};

  auto dist = dr::mp::distribution().halo(radius);
  MDA a({arr_size, arr_size}, dist);
  MDA occurrences_coords({arr_size, arr_size});

  dr::mp::iota(a, 1);
  dr::mp::transform(a, a.begin(), [](auto &&v) { return v % 2; });
  dr::mp::fill(occurrences_coords, 0);

  auto a_submdspan =
      dr::mp::views::submdspan(a.view(), slice_starts, slice_ends);
  int pattern[pattern_size][pattern_size] = {{1, 0}, {0, 1}};

  auto mdspan_pattern_op = [pattern](auto &&v) {
    auto [a_submdspan, occurrences] = v;
    if (pattern[0][0] == a_submdspan(0, 0) &&
        pattern[0][1] == a_submdspan(0, 1) &&
        pattern[1][0] == a_submdspan(1, 0) &&
        pattern[1][1] == a_submdspan(1, 1)) {
      occurrences(0, 0) = 1;
    }
  };

  dr::mp::halo(a).exchange();
  dr::mp::stencil_for_each(mdspan_pattern_op, a_submdspan, occurrences_coords);

  if (dr::mp::rank() == 0) {
    fmt::print("a: \n{} \n", a.mdspan());
    fmt::print("pattern: \n{} \n", pattern);
    fmt::print("occurrences: \n{} \n", occurrences_coords.mdspan());
  }

  dr::mp::finalize();

  return 0;
}
