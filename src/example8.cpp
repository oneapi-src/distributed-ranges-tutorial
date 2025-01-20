// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mp.hpp>
#include <fmt/core.h>
#include <random>
#include <ranges>

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
    // x x 0 0 ... 0
    // x x 0 0 ... 0
    // x 0 x 0 ... 0
    // x 0 0 x ... 0
    // .............
    // x ... 0 0 0 x
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

  dr::mp::broadcasted_vector<double> broadcasted_b;
  std::vector<double> b;
 if (root == dr::mp::rank()) {
  b.resize(matrix.shape().second);
  std::iota(b.begin(), b.end(), 1);

  broadcasted_b.broadcast_data(matrix.shape().second, 0, b,
                               dr::mp::default_comm());
  }
  else {
  broadcasted_b.broadcast_data(matrix.shape().second, 0, std::ranges::empty_view<V>(),
                               dr::mp::default_comm());
  }

  std::vector<double> res(matrix.shape().first);
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
