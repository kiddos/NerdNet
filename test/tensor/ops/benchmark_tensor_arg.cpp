#include <benchmark/benchmark.h>
#include "tensor/tensor.h"
#include "tensor/ops/argmin.h"
#include "tensor/ops/argmax.h"

using nn::tensor::Tensor;
using nn::tensor::TensorShape;

void BM_TensorArgmax(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> result = argmax(t, 0);
  while (state.KeepRunning()) {
    result = argmax(t, 0);
  }
}

BENCHMARK(BM_TensorArgmax)->Unit(benchmark::kMicrosecond)->Range(1, 1 << 10);

void BM_TensorArgmin(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> result = argmin(t, 1);
  while (state.KeepRunning()) {
    result = argmin(t, 0);
  }
}

BENCHMARK(BM_TensorArgmin)->Unit(benchmark::kMicrosecond)->Range(1, 1 << 10);

BENCHMARK_MAIN();
