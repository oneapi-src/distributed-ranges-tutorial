// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mp.hpp>
#include <fmt/core.h>

/* Sparse band matrix vector multiplication */
int main() {
  dr::mp::init(sycl::default_selector_v);
  using I = long;
  using V = double;

  dr::views::csr_matrix_view<V, I> local_data;
  auto root = 0;
  if (root == dr::mp::rank()) {
    auto source = "./resources/example.mtx";
    local_data = dr::read_csr<double, long>(source);
  }

  dr::mp::distributed_sparse_matrix<
      V, I, dr::mp::MpiBackend,
      dr::mp::csr_eq_distribution<V, I, dr::mp::MpiBackend>>
      matrix(local_data, root);

  std::vector<double> b;
  b.reserve(matrix.shape().second);
  std::vector<double> res(matrix.shape().first);
  for (auto i = 0; i < matrix.shape().second; i++) {
    b.push_back(i);
  }

  dr::mp::broadcasted_vector<double> broadcasted_b;
  broadcasted_b.broadcast_data(matrix.shape().second, 0, b,
                               dr::mp::default_comm());

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
