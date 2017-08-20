#include <benchmark/benchmark.h>

#include "NerdNet/convert.h"

using arma::Mat;
using arma::Cube;
using arma::field;
using nerd::nn::Tensor;

void BM_Matrix2Tensor(benchmark::State& state) {
  int size = state.range();
  Mat<float> m(size, size);
  Tensor<float> m_tensor;
  while (state.KeepRunning()) {
    Matrix2Tensor(m, m_tensor);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

BENCHMARK(BM_Matrix2Tensor)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(8, 8 << 10);

void BM_Cube2Tensor(benchmark::State& state) {
  int size = state.range();
  Cube<float> c(size, size, size);
  Tensor<float> c_tensor;
  while (state.KeepRunning()) {
    Cube2Tensor(c, c_tensor);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

BENCHMARK(BM_Cube2Tensor)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1, 256);

void BM_Tensor2Matrix(benchmark::State& state) {
  int size = state.range();
  Mat<float> m(size, size);
  Tensor<float> m_tensor;
  Matrix2Tensor(m, m_tensor);
  while (state.KeepRunning()) {
    Tensor2Matrix(m_tensor, m);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

BENCHMARK(BM_Tensor2Matrix)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(8, 8 << 10);

void BM_Tensor2Cube(benchmark::State& state) {
  int size = state.range();
  Cube<float> c(size, size, size);
  Tensor<float> c_tensor;
  Cube2Tensor(c, c_tensor);
  while (state.KeepRunning()) {
    Tensor2Cube(c_tensor, c);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

BENCHMARK(BM_Tensor2Cube)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1, 256);

BENCHMARK_MAIN();
