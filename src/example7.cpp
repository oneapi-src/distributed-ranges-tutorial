// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mp.hpp>
#include <fmt/core.h>
#include <ranges>

/* Sparse band matrix vector multiplication */
int main() {
  dr::mp::init(sycl::default_selector_v);
  using I = long;
  using V = double;

  dr::views::csr_matrix_view<V, I> local_data;
  auto root = 0;
  if (root == dr::mp::rank()) {
    // x x 0 0 ... 0
    // 0 x x 0 ... 0
    // .............
    // 0 ... 0 0 x x
    auto source = "./resources/example.mtx";
    local_data = dr::read_csr<double, long>(source);
  }

  dr::mp::distributed_sparse_matrix<
      V, I, dr::mp::MpiBackend,
      dr::mp::csr_eq_distribution<V, I, dr::mp::MpiBackend>>
      matrix(local_data, root);

  dr::mp::broadcasted_vector<double> broadcasted_b;
  std::vector<double> b;
  if (root == dr::mp::rank()) {
    b.resize(matrix.shape().second);
    std::iota(b.begin(), b.end(), 1);

    broadcasted_b.broadcast_data(matrix.shape().second, 0, b,
                                 dr::mp::default_comm());
  } else {
    broadcasted_b.broadcast_data(matrix.shape().second, 0,
                                 std::ranges::empty_view<V>(),
                                 dr::mp::default_comm());
  }
  std::vector<double> res(matrix.shape().first);
  gemv(root, res, matrix, broadcasted_b);

  if (root == dr::mp::rank()) {
    fmt::print("Matrix imported from {}\n", "./resources/example.mtx");
    fmt::print("Input: ");
    for (auto x : b) {
      fmt::print("{} ", x);
    }
    fmt::print("\n");
    fmt::print("Matrix vector multiplication res: ");
    for (auto x : res) {
      fmt::print("{} ", x);
    }
    fmt::print("\n");
  }

  dr::mp::finalize();

  return 0;
}
