#pragma once
#include "optimizer.h"

class stochasticGradientDescent : public optimizer
{
private:
  double m_momentum;

public:
  explicit stochasticGradientDescent(double learningRate, double decay, double momentum = 0.0);
  ~stochasticGradientDescent() override = default;
  stochasticGradientDescent(const stochasticGradientDescent &) = default;
  stochasticGradientDescent &operator=(stochasticGradientDescent const &) = default;
  stochasticGradientDescent(stochasticGradientDescent &&) = default;
  stochasticGradientDescent &operator=(stochasticGradientDescent &&) = default;

  void updateParams(denseLayer &layer) const override;
};
