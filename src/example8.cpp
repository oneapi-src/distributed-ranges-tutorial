// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mp.hpp>
#include <fmt/core.h>
#include <random>

/* Sparse band matrix vector multiplication */
int main() {
  dr::mp::init(sycl::default_selector_v);
  using I = long;
  using V = double;
  dr::views::csr_matrix_view<V, I> local_data;
  auto root = 0;
  if (root == dr::mp::rank()) {
    auto size = 10;
    auto nnz = 20;
    auto colInd = new I[nnz];
    auto rowInd = new I[size + 1];
    auto values = new V[nnz];
    std::uniform_real_distribution<double> unif(0, 1);
    std::default_random_engine re;
    for (auto i = 0; i <= size; i++) {
      rowInd[i] = i * 2; // two elements per row
    }
    for (auto i = 0; i < nnz; i++) {
      colInd[i] =
          (i % 2) * (std::max(i / 2, 1)); // column on 0 and diagonal (with
                                          // additional entry in first row)
      values[i] = unif(re);
    }

    local_data = dr::views::csr_matrix_view<V, I>(values, rowInd, colInd,
                                                  {size, size}, nnz, root);
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
    fmt::print("Matrix with {} x {} and number of non-zero entries equal to {} "
               "and entries:\n",
               matrix.shape().first, matrix.shape().second, matrix.size());
    for (auto [i, v] : matrix) {
      auto [n, m] = i;
      fmt::print("Matrix entry <{}, {}, {}>\n", n, m, v);
    }
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
