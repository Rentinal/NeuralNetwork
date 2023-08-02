#pragma once
#include "optimizer.hpp"

class adaptiveMomentum : public optimizer
{
private:
  double m_epsilon;
  double m_beta1;
  double m_beta2;

public:
  explicit adaptiveMomentum(double learningRate = 0.001, double decay = 0.0, double epsilon = 1e-7, double beta1 = 0.9, double beta2 = 0.999);
  ~adaptiveMomentum() override = default;
  adaptiveMomentum(const adaptiveMomentum &) = default;
  adaptiveMomentum &operator=(adaptiveMomentum const &) = default;
  adaptiveMomentum(adaptiveMomentum &&) = default;
  adaptiveMomentum &operator=(adaptiveMomentum &&) = default;

  void updateParams(denseLayer &layer) const override;
};
