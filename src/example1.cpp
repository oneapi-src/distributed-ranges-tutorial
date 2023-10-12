// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mhp.hpp>
#include <fmt/core.h>

namespace mhp = dr::mhp;

int main(int argc, char **argv) {

  mhp::init(sycl::default_selector_v);

  mhp::distributed_vector<char> dv(81);
  std::string decoded_string(80, 0);

  mhp::copy(
      0,
      std::string("Mjqqt%|twqi&%Ymnx%nx%ywfsxrnxnts%kwtr%ymj%tsj%fsi%tsq~%"
                  "Inxywngzyji%Wfsljx%wjfqr&"),
      dv.begin());

  mhp::for_each(dv, [](char &val) { val -= 5; });
  mhp::copy(0, dv, decoded_string.begin());

  if (mhp::rank() == 0)
    fmt::print("{}\n", decoded_string);

  mhp::finalize();

  return 0;
}
