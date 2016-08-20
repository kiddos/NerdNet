#define BOOST_TEST_MODULE ACTIVATION TEST
#include <boost/test/included/unit_test.hpp>
#include "act.h"
#include <math.h>

namespace nn {

double derivative(double (*f)(double), double val) {
  const double eps = 1e-6;
  return (f(val + eps) - f(val - eps)) / (2 * eps);
}

}

BOOST_AUTO_TEST_CASE(identity_test) {
  BOOST_CHECK(nn::identity.act(6.6) == 6.6);
  BOOST_CHECK(nn::identity.act(0) == 0);
  BOOST_CHECK(nn::identity.act(-6.6) == -6.6);
  BOOST_CHECK(fabs(nn::identity.actd(1.0) -
                   nn::derivative(nn::identity.act, 1.0)) <= 1e-6);
}

BOOST_AUTO_TEST_CASE(sigmoid_test) {
  BOOST_CHECK(fabs(nn::sigmoid.act(6.6) - 0.99864148) <= 1e-6);
  BOOST_CHECK(fabs(nn::sigmoid.act(0) - 0.5) <= 1e-6);
  BOOST_CHECK(fabs(nn::sigmoid.act(-6.6) - 0.0013585) <= 1e-6);
  BOOST_CHECK(fabs(nn::sigmoid.actd(1.0) -
                   nn::derivative(nn::sigmoid.act, 1.0)) <= 1e-6);
}

BOOST_AUTO_TEST_CASE(arctan_test) {
  BOOST_CHECK(fabs(nn::arctan.act(6.6) - 1.42042490) <= 1e-6);
  BOOST_CHECK(nn::arctan.act(0) == 0);
  BOOST_CHECK(fabs(nn::arctan.act(-6.6) + 1.42042490) <= 1e-6);
  BOOST_CHECK(fabs(nn::arctan.actd(1.0) -
                   nn::derivative(nn::arctan.act, 1.0)) <= 1e-6);
}

BOOST_AUTO_TEST_CASE(tanh_test) {
  BOOST_CHECK(fabs(nn::tanhyperbolic.act(6.6) - 0.99999630) <= 1e-6);
  BOOST_CHECK(nn::tanhyperbolic.act(0) == 0);
  BOOST_CHECK(fabs(nn::tanhyperbolic.act(-6.6) + 0.99999630) <= 1e-6);
  BOOST_CHECK(fabs(nn::tanhyperbolic.actd(1.0) -
                   nn::derivative(nn::tanhyperbolic.act, 1.0)) <= 1e-6);
}

BOOST_AUTO_TEST_CASE(bipolarsigmoid_test) {
  BOOST_CHECK(fabs(nn::bipolarsigmoid.act(6.6) - 0.99728296) <= 1e-6);
  BOOST_CHECK(nn::bipolarsigmoid.act(0) == 0);
  BOOST_CHECK(fabs(nn::bipolarsigmoid.act(-6.6) + 0.99728296) <= 1e-6);
  BOOST_CHECK(fabs(nn::bipolarsigmoid.actd(1.0) -
                   nn::derivative(nn::bipolarsigmoid.act, 1.0)) <= 1e-6);
}

BOOST_AUTO_TEST_CASE(comploglog_test) {
  BOOST_CHECK(fabs(nn::comploglog.act(6.6) - 1.00000000) <= 1e-6);
  BOOST_CHECK(fabs(nn::comploglog.act(0) - 0.63212056) <= 1e-6);
  BOOST_CHECK(fabs(nn::comploglog.act(-6.6) - 0.00135944) <= 1e-6);
  BOOST_CHECK(fabs(nn::comploglog.actd(1.0) -
                   nn::derivative(nn::comploglog.act, 1.0)) <= 1e-6);
}

BOOST_AUTO_TEST_CASE(lecuntanh_test) {
  BOOST_CHECK(fabs(nn::lecuntanh.act(6.6) - 1.71538279) <= 1e-6);
  BOOST_CHECK(nn::lecuntanh.act(0) == 0);
  BOOST_CHECK(fabs(nn::lecuntanh.act(-6.6) + 1.71538279) <= 1e-6);
  BOOST_CHECK(fabs(nn::lecuntanh.actd(1.0) -
                   nn::derivative(nn::lecuntanh.act, 1.0)) <= 1e-6);
}

BOOST_AUTO_TEST_CASE(hardtanh_test) {
  BOOST_CHECK(nn::hardtanh.act(6.6) == 1.0);
  BOOST_CHECK(nn::hardtanh.act(0) == 0);
  BOOST_CHECK(nn::hardtanh.act(-6.6) == -1.0);
  BOOST_CHECK(fabs(nn::hardtanh.actd(0.5) -
                   nn::derivative(nn::hardtanh.act, 0.5)) <= 1e-6);
}

BOOST_AUTO_TEST_CASE(absolute_test) {
  BOOST_CHECK(fabs(nn::absolute.act(6.6) - 6.6) <= 1e-6);
  BOOST_CHECK(nn::absolute.act(0) == 0);
  BOOST_CHECK(fabs(nn::absolute.act(-6.6) - 6.6) <= 1e-6);
  BOOST_CHECK(fabs(nn::absolute.actd(1.0) -
                   nn::derivative(nn::absolute.act, 1.0)) <= 1e-6);
}

BOOST_AUTO_TEST_CASE(relu_test) {
  BOOST_CHECK(nn::relu.act(6.6) == 6.6);
  BOOST_CHECK(nn::relu.act(0) == 0.0);
  BOOST_CHECK(nn::relu.act(-6.6) == 0.0);
  BOOST_CHECK(fabs(nn::lecuntanh.actd(1.0) -
                   nn::derivative(nn::lecuntanh.act, 1.0)) <= 1e-6);
}

BOOST_AUTO_TEST_CASE(relu_cos_test) {
  BOOST_CHECK(fabs(nn::relucos.act(6.6) - 7.55023259) <= 1e-6);
  BOOST_CHECK(nn::relucos.act(0) == 1.0);
  BOOST_CHECK(fabs(nn::relucos.act(-6.6) - 0.95023259) <= 1e-6);
  BOOST_CHECK(fabs(nn::relucos.actd(1.0) -
                   nn::derivative(nn::relucos.act, 1.0)) <= 1e-6);
}

BOOST_AUTO_TEST_CASE(relu_sin_test) {
  BOOST_CHECK(fabs(nn::relusin.act(6.6) - 6.91154136) <= 1e-6);
  BOOST_CHECK(nn::relusin.act(0) == 0.0);
  BOOST_CHECK(fabs(nn::relusin.act(-6.6) + 0.31154136) <= 1e-6);
  BOOST_CHECK(fabs(nn::relusin.actd(1.0) -
                   nn::derivative(nn::relusin.act, 1.0)) <= 1e-6);
}

BOOST_AUTO_TEST_CASE(smooth_relu_test) {
  BOOST_CHECK(fabs(nn::smoothrelu.act(6.6) - 6.6) <= 1e-2);
  BOOST_CHECK(fabs(nn::smoothrelu.act(0) - 0.69314718) <= 1e-6);
  BOOST_CHECK(fabs(nn::smoothrelu.act(-6.6) - 0.00135944) <= 1e-6);
  BOOST_CHECK(fabs(nn::smoothrelu.actd(1.0) -
                   nn::derivative(nn::smoothrelu.act, 1.0)) <= 1e-6);
}
