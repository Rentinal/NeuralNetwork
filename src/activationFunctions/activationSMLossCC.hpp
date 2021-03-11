#pragma once
#include "activationSoftmax.hpp"
#include "./lossFunctions/categoricalLossEntropy.hpp"

class activationSMLossCC
{
private:
  dMatrix m_output;
  dMatrix m_dInput;
  activationSoftmax m_activation;
  categoricalCrossEntropy m_loss;

public:
  activationSMLossCC();
  ~activationSMLossCC() = default;
  activationSMLossCC(const activationSMLossCC &) = default;
  activationSMLossCC &operator=(activationSMLossCC const &) = default;
  activationSMLossCC(activationSMLossCC &&) = default;
  activationSMLossCC &operator=(activationSMLossCC &&) = default;

  double forward(const dMatrix &inputs, const uiMatrix &yTrue);
  void backward(const dMatrix &dValues, const uiMatrix &yTrue);

  [[nodiscard]] const dMatrix &dInput() const { return m_dInput; }
  [[nodiscard]] const dMatrix &output() const { return m_output; }
};
