#pragma once
#include "optimizer.hpp"

class rootMeanSquareProp : public optimizer
{
private:
  double m_epsilon;
  double m_rho;

public:
  explicit rootMeanSquareProp(double learningRate = 0.001, double decay = 0.0, double epsilon = 1e-7, double rho = 0.9);
  ~rootMeanSquareProp() override = default;
  rootMeanSquareProp(const rootMeanSquareProp &) = default;
  rootMeanSquareProp &operator=(rootMeanSquareProp const &) = default;
  rootMeanSquareProp(rootMeanSquareProp &&) = default;
  rootMeanSquareProp &operator=(rootMeanSquareProp &&) = default;

  void updateParams(denseLayer &layer) const override;
};
