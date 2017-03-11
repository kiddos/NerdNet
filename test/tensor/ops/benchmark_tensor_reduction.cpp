#include <benchmark/benchmark.h>
#include "tensor/ops/reduce_mean.h"
#include "tensor/ops/reduce_sum.h"
#include "tensor/tensor.h"

using nn::tensor::Tensor;
using nn::tensor::TensorShape;

void BM_TensorReduceMeanAxis(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> result = reduce_mean(t, 0);
  while (state.KeepRunning()) {
    result = reduce_mean(t, 0);
  }
}

void BM_TensorReduceMean(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  reduce_mean(t);
  while (state.KeepRunning()) {
    reduce_mean(t);
  }
}

BENCHMARK(BM_TensorReduceMeanAxis)
    ->Unit(benchmark::kMicrosecond)
    ->Range(8, 1 << 10);
BENCHMARK(BM_TensorReduceMean)
    ->Unit(benchmark::kMicrosecond)
    ->Range(8, 1 << 10);

void BM_TensorReduceSumAxis(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> result = reduce_mean(t, 0);
  while (state.KeepRunning()) {
    result = reduce_mean(t, 0);
  }
}

void BM_TensorReduceSum(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  reduce_sum(t);
  while (state.KeepRunning()) {
    reduce_mean(t);
  }
}

BENCHMARK(BM_TensorReduceMeanAxis)
    ->Unit(benchmark::kMicrosecond)
    ->Range(8, 1 << 10);
BENCHMARK(BM_TensorReduceMean)
    ->Unit(benchmark::kMicrosecond)
    ->Range(8, 1 << 10);

BENCHMARK_MAIN();
