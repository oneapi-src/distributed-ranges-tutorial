// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mp.hpp>
#include <fmt/core.h>

int main(int argc, char **argv) {

  dr::mp::init(sycl::default_selector_v);

  dr::mp::distributed_vector<char> dv(81);
  std::string decoded_string(80, 0);

  dr::mp::copy(
      0,
      std::string("Mjqqt%|twqi&%Ymnx%nx%ywfsxrnxnts%kwtr%ymj%tsj%fsi%tsq~%"
                  "Inxywngzyji%Wfsljx%wjfqr&"),
      dv.begin());

  dr::mp::for_each(dv, [](char &val) { val -= 5; });
  dr::mp::copy(0, dv, decoded_string.begin());

  if (dr::mp::rank() == 0)
    fmt::print("{}\n", decoded_string);

  dr::mp::finalize();

  return 0;
}
