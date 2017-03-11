#include <benchmark/benchmark.h>
#include "tensor/tensor.h"
#include "tensor/ops/log.h"
#include "tensor/ops/exp.h"
#include "tensor/ops/sqrt.h"

using nn::tensor::Tensor;
using nn::tensor::TensorShape;

void BM_TensorLog(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> result = log(t);
  while (state.KeepRunning()) {
    result = log(t);
  }
}

BENCHMARK(BM_TensorLog)->Unit(benchmark::kMicrosecond)->Range(1, 1 << 10);

void BM_TensorExp(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> result = exp(t);
  while (state.KeepRunning()) {
    result = exp(t);
  }
}

BENCHMARK(BM_TensorExp)->Unit(benchmark::kMicrosecond)->Range(1, 1 << 10);

void BM_TensorSqrt(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> result = sqrt(t);
  while (state.KeepRunning()) {
    result = sqrt(t);
  }
}

BENCHMARK(BM_TensorSqrt)->Unit(benchmark::kMicrosecond)->Range(1, 1 << 10);

BENCHMARK_MAIN();
