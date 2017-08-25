#include "NerdNet/util/qt/plot_window.h"

#include <armadillo>
#include <iostream>
#include <chrono>
#include <random>

#include "NerdNet/convert.h"
#include "NerdNet/tensor.h"

using nerd::nn::ui::PlotWindow;
using nerd::nn::Tensor;
using arma::Mat;

void GenerateClusterData(
    Tensor<float>& data_tensor, Tensor<float>& label_tensor, int data_size,
    float mean, float stddev,
    const std::vector<std::pair<PlotWindow::Color, PlotWindow::Shape>>&
        labeling) {
  std::mt19937 gen(
      std::chrono::system_clock::now().time_since_epoch().count());
  std::normal_distribution<float> dist(mean, stddev);
  std::uniform_int_distribution<int> random_labeling(0, labeling.size() - 1);

  Mat<float> data(data_size, 2), label(data_size, labeling.size());
  label.zeros();
  for (int i = 0; i < data_size; ++i) {
    data(i, 0) = dist(gen);
    data(i, 1) = dist(gen);
    label(i, random_labeling(gen)) = 1.0f;
  }
  Matrix2Tensor(data, data_tensor);
  Matrix2Tensor(label, label_tensor);
}

int main(int argc, char** argv) {
  QApplication app(argc, argv);
  PlotWindow window;
  window.show();

  std::vector<std::pair<PlotWindow::Color, PlotWindow::Shape>> labeling = {
      {PlotWindow::RED, PlotWindow::CROSS},
      {PlotWindow::GREEN, PlotWindow::CIRCLE},
      {PlotWindow::BLUE, PlotWindow::SQUARE}};

  arma::arma_rng::set_seed_random();
  Tensor<float> data, label;
  GenerateClusterData(data, label, 300, 0.0f, 2.0f, labeling);

  window.SetData(data, label, labeling);
  Tensor<float> grid = window.grid();
  std::cout << grid << std::endl;
  return app.exec();
}
