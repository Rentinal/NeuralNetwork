#pragma once
class denseLayer;

class stochasticGradientDescent
{
private:
  double m_learningRate;

public:
  stochasticGradientDescent(const double learningRate = 1.0);
  ~stochasticGradientDescent() = default;
  stochasticGradientDescent(const stochasticGradientDescent &) = default;
  stochasticGradientDescent &operator=(stochasticGradientDescent const &) = default;
  stochasticGradientDescent(stochasticGradientDescent &&) = default;
  stochasticGradientDescent &operator=(stochasticGradientDescent &&) = default;

  void updateParams(denseLayer &layer);
};
