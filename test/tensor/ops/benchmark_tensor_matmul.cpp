#include <benchmark/benchmark.h>
#include "tensor/ops/matmul.h"
#include "tensor/tensor.h"

using nn::tensor::Tensor;
using nn::tensor::TensorShape;

void BM_TensorMatMul(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t1 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> t2 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> result = t1 % t2;
  while (state.KeepRunning()) {
    matmul(t1, t2, result);
  }
}

BENCHMARK(BM_TensorMatMul)
    ->Unit(benchmark::kMicrosecond)
    ->MinTime(3.6)
    ->Range(1, 1 << 10);

BENCHMARK_MAIN();
