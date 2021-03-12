#pragma once
#include "optimizers/optimizer.h"

class adaptiveGradient : public optimizer
{
private:
  double m_epsilon;

public:
  explicit adaptiveGradient(double learningRate, double decay, double epsilon = 1e-7);
  ~adaptiveGradient() override = default;
  adaptiveGradient(const adaptiveGradient &) = default;
  adaptiveGradient &operator=(adaptiveGradient const &) = default;
  adaptiveGradient(adaptiveGradient &&) = default;
  adaptiveGradient &operator=(adaptiveGradient &&) = default;

  void updateParams(denseLayer &layer) const override;
};
