#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <QApplication>
#include <armadillo>
#include <boost/program_options.hpp>

#include "NerdNet/convert.h"
#include "NerdNet/layers.h"
#include "NerdNet/nerd_net.h"
#include "NerdNet/trainer/gradient_descent_trainer.h"
#include "NerdNet/util.h"
#include "NerdNet/util/qt/plot_window.h"

using nerd::nn::VariableShape;
using nerd::nn::NerdNet;
using nerd::nn::FCLayer;
using nerd::nn::SigmoidLayer;
using nerd::nn::SoftmaxCrossEntropy;
using nerd::nn::Tensor;
using nerd::nn::trainer::GradientDescentTrainer;
using nerd::nn::ui::PlotWindow;
using arma::Mat;
using arma::Row;

void GenerateSpiralData(Mat<float>& data, Mat<float>& label, int data_size,
                        float noise) {
  std::mt19937 gen(
      std::chrono::system_clock::now().time_since_epoch().count());
  std::normal_distribution<float> dist(0.0f, noise);
  data = arma::Mat<float>(data_size, 2);
  label = arma::Mat<float>(data_size, 3);
  label.zeros();

  for (int i = 0; i < data_size / 3; ++i) {
    float val = static_cast<float>(i) / data_size * 3;
    data(i, 0) = (std::exp(val) - 1) * std::cos(val) + dist(gen);
    data(i, 1) = (std::exp(val) - 1) * std::sin(val) + dist(gen);
    label(i, 0) = 1.0;
  }

  for (int i = data_size / 3; i < 2 * data_size / 3; ++i) {
    float val = static_cast<float>(i - data_size / 3) / data_size * 3;
    data(i, 0) =
        (std::exp(val) - 1) * std::cos(val + M_PI / 3 * 2) + dist(gen);
    data(i, 1) =
        (std::exp(val) - 1) * std::sin(val + M_PI / 3 * 2) + dist(gen);
    label(i, 1) = 1.0;
  }

  for (int i = 2 * data_size / 3; i < data_size; ++i) {
    float val = static_cast<float>(i - 2 * data_size / 3) / data_size * 3;
    data(i, 0) =
        (std::exp(val) - 1) * std::cos(val + M_PI / 3 * 4) + dist(gen);
    data(i, 1) =
        (std::exp(val) - 1) * std::sin(val + M_PI / 3 * 4) + dist(gen);
    label(i, 2) = 1.0;
  }
}

int main(int argc, char** argv) {
  using namespace boost::program_options;
  options_description desc("Spiral Data Example");
  desc.add_options()("help,h", "./spiral");
  desc.add_options()("data-size", value<int>()->default_value(300),
                     "data size");
  desc.add_options()("noise", value<float>()->default_value(0.06),
                     "data noise");
  desc.add_options()("hidden-size", value<int>()->default_value(32),
                     "hidden size");
  desc.add_options()("learning-rate", value<float>()->default_value(1e-3),
                     "learning rate");
  desc.add_options()("max-epoch", value<int>()->default_value(20000),
                     "max epoch to train");

  variables_map vmap;
  store(parse_command_line(argc, argv, desc), vmap);
  notify(vmap);

  if (vmap.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

  int data_size = vmap["data-size"].as<int>();
  float noise = vmap["noise"].as<float>();
  Mat<float> data, label;
  GenerateSpiralData(data, label, data_size, noise);

  int hidden_size = vmap["hidden-size"].as<int>();
  std::shared_ptr<NerdNet> nerdnet = std::make_shared<NerdNet>();
  nerdnet->AddLayer<FCLayer>(VariableShape{2, hidden_size});
  nerdnet->AddLayer<SigmoidLayer>();
  nerdnet->AddLayer<SoftmaxCrossEntropy>(VariableShape{hidden_size, 3});

  Tensor<float> data_tensor, label_tensor;
  Matrix2Tensor(data, data_tensor);
  Matrix2Tensor(label, label_tensor);

  int max_epoch = vmap["max-epoch"].as<int>();
  float learning_rate = vmap["learning-rate"].as<float>();
  GradientDescentTrainer trainer(learning_rate, nerdnet);
  for (int epoch = 0; epoch < max_epoch + 1; ++epoch) {
    float loss = trainer.Train(data_tensor, label_tensor);
    if (epoch % 1000 == 0) {
      Tensor<float> prediction_tensor = nerdnet->Feed(data_tensor);
      std::cout << "\r" << epoch << ". loss: " << loss / data_size
                << ", accuracy: "
                << Accuracy(prediction_tensor, label_tensor) * 100.0 << "%"
                << std::flush;
    }
  }
  std::cout << std::endl;

  QApplication app(argc, argv);
  PlotWindow window;

  window.SetData(data_tensor, label_tensor,
                 {{PlotWindow::RED, PlotWindow::DISC},
                  {PlotWindow::BLUE, PlotWindow::SQUARE},
                  {PlotWindow::GREEN, PlotWindow::TRIANGLE}});

  Tensor<float> grid = window.grid();
  Tensor<float> prediction = nerdnet->Feed(grid);
  window.SetGridBoundary(prediction);

  window.show();
  return app.exec();
}
