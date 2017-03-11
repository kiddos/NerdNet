#include <benchmark/benchmark.h>
#include "tensor/ops/basic_ops.h"
#include "tensor/tensor.h"

using nn::tensor::Tensor;
using nn::tensor::TensorShape;

void BM_TensorAdd(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t1 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> t2 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> t3 = t1 + t2;
  while (state.KeepRunning()) {
    t3 = t1 + t2;
  }
}

BENCHMARK(BM_TensorAdd)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 10);

void BM_TensorBroadCastAdd(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t1 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> t2 = Tensor<float>::Gaussian({size}, 0.0f, 1.0f);
  Tensor<float> t3 = t1 + t2;
  while (state.KeepRunning()) {
    t3 = t1 + t2;
  }
}

BENCHMARK(BM_TensorBroadCastAdd)
    ->Unit(benchmark::kMicrosecond)
    ->Range(2, 2 << 10);

void BM_TensorSub(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t1 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> t2 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> t3 = t1 - t2;
  while (state.KeepRunning()) {
    t3 = t1 - t2;
  }
}

BENCHMARK(BM_TensorSub)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 10);

void BM_TensorBroadCastSub(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t1 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> t2 = Tensor<float>::Gaussian({size}, 0.0f, 1.0f);
  Tensor<float> t3 = t1 - t2;
  while (state.KeepRunning()) {
    t3 = t1 - t2;
  }
}

BENCHMARK(BM_TensorBroadCastSub)
    ->Unit(benchmark::kMicrosecond)
    ->Range(2, 2 << 10);

void BM_TensorMul(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t1 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> t2 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> t3 = t1 * t2;
  while (state.KeepRunning()) {
    t3 = t1 * t2;
  }
}

BENCHMARK(BM_TensorMul)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 10);

void BM_TensorBroadCastMul(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t1 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> t2 = Tensor<float>::Gaussian({size}, 0.0f, 1.0f);
  Tensor<float> t3 = t1 * t2;
  while (state.KeepRunning()) {
    t3 = t1 * t2;
  }
}

BENCHMARK(BM_TensorBroadCastMul)
    ->Unit(benchmark::kMicrosecond)
    ->Range(2, 2 << 10);

void BM_TensorDiv(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t1 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> t2 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> t3 = t1 / t2;
  while (state.KeepRunning()) {
    t3 = t1 / t2;
  }
}

BENCHMARK(BM_TensorDiv)->Unit(benchmark::kMicrosecond)->Range(2, 2 << 10);

void BM_TensorBroadCastDiv(benchmark::State& state) {
  int size = state.range(0);
  Tensor<float> t1 = Tensor<float>::Gaussian({size, size}, 0.0f, 1.0f);
  Tensor<float> t2 = Tensor<float>::Gaussian({size}, 0.0f, 1.0f);
  Tensor<float> t3 = t1 / t2;
  while (state.KeepRunning()) {
    t3 = t1 / t2;
  }
}

BENCHMARK(BM_TensorBroadCastDiv)
    ->Unit(benchmark::kMicrosecond)
    ->Range(2, 2 << 10);

BENCHMARK_MAIN();
