#define BOOST_TEST_MODULE LAYER TEST
#include <boost/test/included/unit_test.hpp>
#include <math.h>
#include <memory>

#include "neuralnet.h"

BOOST_AUTO_TEST_CASE(layer_random_init_weights) {
  nn::LayerParam param;
  param.previous_nodes = 10;
  param.nodes = 10;
  param.standard_dev = 0.05;
  std::shared_ptr<nn::Layer> layer(new nn::Layer(param));
  const nn::mat w = layer->getw();

  double mean = 0;
  for (int i = 0 ; i < static_cast<int>(w.n_rows) ; ++i) {
    for (int j = 0 ; j < static_cast<int>(w.n_cols) ; ++j) {
      mean += w(i, j);
    }
  }

  double stddev = 0;
  for (int i = 0 ; i < static_cast<int>(w.n_rows) ; ++i) {
    for (int j = 0 ; j < static_cast<int>(w.n_cols) ; ++j) {
      const double diff = mean - w(i, j);
      stddev += diff * diff;
    }
  }
  stddev = sqrt(stddev / w.n_rows / w.n_cols);
  BOOST_CHECK(fabs(stddev - param.standard_dev) <= 1e-6);
}

BOOST_AUTO_TEST_CASE(layer_dimension_test) {
  const int input_nodes = 10;
  const int output_nodes = 20;
  nn::LayerParam param;
  param.previous_nodes = input_nodes;
  param.nodes = output_nodes;
  param.standard_dev = 1.0;

  std::shared_ptr<nn::Layer> layer(new nn::Layer(param));
  nn::mat input(1, 10);
  nn::mat delta(1, 20);

  for (int i = 0 ; i < 10 ; ++i) {
    input(0, i) = 1;
  }
  for (int i = 0 ; i < 20 ; ++i) {
    delta(0, i) = 1;
  }

  const nn::mat output1 = layer->forwardprop(input);
  const nn::mat output2 = layer->backprop(delta);
  BOOST_CHECK(output1.n_cols == output_nodes);
  BOOST_CHECK(output2.n_cols == input_nodes);
}
