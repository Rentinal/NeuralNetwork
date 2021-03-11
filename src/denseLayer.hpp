#pragma once
#include "utils.hpp"

class denseLayer
{
private:
  nc::NdArray<double> m_weights;
  nc::NdArray<double> m_biases;
  nc::NdArray<double> m_outputs;

public:
  denseLayer(uint32_t numInputs, uint32_t numNeurons);
  ~denseLayer() = default;
  denseLayer(const denseLayer &) = default;
  denseLayer &operator=(denseLayer const &) = default;
  denseLayer(denseLayer &&) = default;
  denseLayer &operator=(denseLayer &&) = default;

  //sum(weights * input) + bias
  void forward(const nc::NdArray<double> &inputs);

  [[nodiscard]] const nc::NdArray<double> &weights() const;

  [[nodiscard]] const nc::NdArray<double> &biases() const;

  void addToWeights(const nc::NdArray<double> &weights);

  void addToBiases(const nc::NdArray<double> &biases);

  void setWeights(const nc::NdArray<double> &weights);

  void setBiases(const nc::NdArray<double> &biases);

  [[nodiscard]] const nc::NdArray<double> &output() const;
};
