#include "NerdNet/layer/constant_initializer.h"

namespace nerd {
namespace nn {

ConstantInitializer::ConstantInitializer(float value) : value_(value) {}

float ConstantInitializer::Next() { return value_; }

} /* end of nn namespace */
} /* end of nerd namespace */
