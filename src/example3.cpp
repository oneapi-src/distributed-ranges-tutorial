// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mp.hpp>
#include <fmt/core.h>

namespace mhp = dr::mhp;

/* The example simulates the elementary 1-d cellular automaton. Description of
 * what the automaton is and how it works can be found at
 * https://en.wikipedia.org/wiki/Elementary_cellular_automaton
 * Visualisation of the automaton work is available
 * https://elife-asu.github.io/wss-modules/modules/1-1d-cellular-automata
 * (credit: Emergence team @ Arizona State University)*/

constexpr std::size_t asize = 60;
constexpr std::size_t steps = 60;

constexpr uint8_t ca_rule = 28;

auto newvalue = [](auto &&p) {
  auto v = &p;
  uint8_t pattern = 4 * v[-1] + 2 * v[0] + v[1];
  return (ca_rule >> pattern) % 2;
};

int main(int argc, char **argv) {

  dr::mp::init(sycl::default_selector_v);

  auto dist = dr::mp::distribution().halo(1);
  dr::mp::distributed_vector<uint8_t> a1(asize + 2, 0, dist),
      a2(asize + 2, 0, dist);

  auto in = rng::subrange(a1.begin() + 1, a1.end() - 1);
  auto out = rng::subrange(a2.begin() + 1, a2.end() - 1);

  /* initial value of the automaton - customize it if you want to */
  in[0] = 1;

  if (dr::mp::rank() == 0)
    fmt::print("{}\n", in);

  for (std::size_t s = 0; s < steps; s++) {
    dr::mp::halo(in).exchange();

    dr::mp::transform(in, out.begin(), newvalue);

    std::swap(in, out);

    /* fmt::print() is rather slow here, as it gets element by element from
     * remote nodes. Use with care. */
    if (dr::mp::rank() == 0)
      fmt::print("{}\n", in);
  }

  dr::mp::finalize();

  return 0;
}
