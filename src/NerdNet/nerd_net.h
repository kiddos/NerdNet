#ifndef NERD_NET_H
#define NERD_NET_H

#include "NerdNet/except/nerdnet_exception.h"
#include "NerdNet/layer/base_layer.h"
#include "NerdNet/layer/input_layer.h"
#include "NerdNet/layer/cost_func_layer.h"
#include "NerdNet/tensor.h"

#include <memory>
#include <vector>

namespace nerd {
namespace nn {

class NerdNet {
 public:
  NerdNet();

  BaseLayer* layer(int index) { return layers_[index].get(); }
  BaseLayer* first() { return layers_[0].get(); }
  BaseLayer* last() { return layers_[layers_.size() - 1].get(); }
  InputLayer* input_layer() { return reinterpret_cast<InputLayer*>(first()); }
  CostFunction* cost_function() {
    return reinterpret_cast<CostFunction*>(last());
  }
  int layer_count() const { return layers_.size(); }

  template <typename T, typename... Args>
  void AddLayer(Args... args) {
    std::unique_ptr<T> ptr(new T(last(), args...));
    layers_.push_back(std::move(ptr));
  }
  bool Init();

  template <typename T>
  Tensor<float> Feed(const T& input) {
    input_layer()->SetInput(input);
    return last()->ForwardProp();
  }

 private:
  std::vector<std::unique_ptr<BaseLayer>> layers_;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: NERD_NET_H */
