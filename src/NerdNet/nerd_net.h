#ifndef NERD_NET_H
#define NERD_NET_H

#include <iostream>
#include <memory>
#include <typeinfo>
#include <vector>

#include "NerdNet/except/nerdnet_exception.h"
#include "NerdNet/layers.h"
#include "NerdNet/tensor.h"

namespace nerd {
namespace nn {

class NerdNet {
 public:
  NerdNet();

  BaseLayer* layer(int index) { return layers_[index].get(); }
  BaseLayer* first() { return layers_[0].get(); }
  BaseLayer* last() { return layers_[layers_.size() - 1].get(); }
  InputLayer* input_layer() { return dynamic_cast<InputLayer*>(first()); }
  CostFunction* cost_function() { return dynamic_cast<CostFunction*>(last()); }
  int layer_count() const { return layers_.size(); }
  bool has_variable(int i) { return has_variables_[i]; }

  template <typename T, typename... Args>
  void AddLayer(Args... args) {
    std::unique_ptr<T> ptr(new T(last(), args...));
    if (dynamic_cast<FCLayer*>(ptr.get()) != nullptr) {
      has_variables_.push_back(true);
    } else {
      has_variables_.push_back(false);
    }
    layers_.push_back(std::move(ptr));
  }
  bool Init();

  template <typename T>
  Tensor<float> Feed(const T& input) {
    input_layer()->SetInput(input);
    return last()->ForwardProp();
  }

 private:
  std::vector<bool> has_variables_;
  std::vector<std::unique_ptr<BaseLayer>> layers_;
};

} /* end of nn namespace */
} /* end of nerd namespace */

#endif /* end of include guard: NERD_NET_H */
