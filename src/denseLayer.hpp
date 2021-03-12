#pragma once
#include "utils.hpp"

class denseLayer
{
private:
  //Forward Pass
  dMatrix m_weights;
  dMatrix m_biases;
  dMatrix m_outputs;
  dMatrix m_inputs;
  //Backward Pass
  dMatrix m_dWeights;
  dMatrix m_dBiases;
  dMatrix m_dInputs;
  //Optimization SGD
  dMatrix m_weightMomentums;
  dMatrix m_biasMomentums;
  //Optimization AdaGrad
  dMatrix m_weightCache;
  dMatrix m_biasCache;

public:
  denseLayer(uint32_t numInputs, uint32_t numNeurons);
  denseLayer(const dMatrix &weights, const dMatrix &biases);
  ~denseLayer() = default;
  denseLayer(const denseLayer &) = default;
  denseLayer &operator=(denseLayer const &) = default;
  denseLayer(denseLayer &&) = default;
  denseLayer &operator=(denseLayer &&) = default;

  //Forward Pass
  void forward(const dMatrix &inputs);
  //Backward Pass
  void backward(const dMatrix &dValues);

  void addToWeights(const dMatrix &weights);

  void addToBiases(const dMatrix &biases);

  void addToWeightCache(const dMatrix &weights);

  void addToBiasCache(const dMatrix &biases);

  void setWeights(const dMatrix &weights);

  void setBiases(const dMatrix &biases);

  void setWeightMomentums(const dMatrix &weights);

  void setBiasMomentums(const dMatrix &biases);

  [[nodiscard]] const dMatrix &output() const { return m_outputs; }
  [[nodiscard]] const dMatrix &weights() const { return m_weights; }
  [[nodiscard]] const dMatrix &biases() const { return m_biases; }

  [[nodiscard]] const dMatrix &dInput() const { return m_dInputs; }
  [[nodiscard]] const dMatrix &dWeights() const { return m_dWeights; }
  [[nodiscard]] const dMatrix &dBiases() const { return m_dBiases; }

  [[nodiscard]] const dMatrix &weightMomentums() const { return m_weightMomentums; }
  [[nodiscard]] const dMatrix &biasMomentums() const { return m_biasMomentums; }

  [[nodiscard]] const dMatrix &weightCache() const { return m_weightCache; }
  [[nodiscard]] const dMatrix &biasCache() const { return m_biasCache; }
};
