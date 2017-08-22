#ifndef LAYERS_H
#define LAYERS_H

// initializers
#include "NerdNet/layer/constant_initializer.h"
#include "NerdNet/layer/normal_initializer.h"

// bases
#include "NerdNet/layer/fc_layer.h"
#include "NerdNet/layer/input_layer.h"

// activations
#include "NerdNet/layer/arctan_layer.h"
#include "NerdNet/layer/leaky_relu_layer.h"
#include "NerdNet/layer/relu_layer.h"
#include "NerdNet/layer/sigmoid_layer.h"
#include "NerdNet/layer/smooth_relu_layer.h"
#include "NerdNet/layer/tanh_layer.h"

// cost functions
#include "NerdNet/layer/mean_square_error.h"
#include "NerdNet/layer/softmax_cross_entropy.h"
#include "NerdNet/layer/sigmoid_cross_entropy.h"
#include "NerdNet/layer/kullback_leibler_divergence.h"

#endif /* end of include guard: LAYERS_H */
