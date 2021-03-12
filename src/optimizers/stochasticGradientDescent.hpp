#pragma once
class denseLayer;

class stochasticGradientDescent
{
private:
  double m_learningRate;
  double m_currLearningRate;
  double m_decay;
  size_t m_iterations{ 0 };

public:
  explicit stochasticGradientDescent(double learningRate = 1.0, double decay = 0.0);
  ~stochasticGradientDescent() = default;
  stochasticGradientDescent(const stochasticGradientDescent &) = default;
  stochasticGradientDescent &operator=(stochasticGradientDescent const &) = default;
  stochasticGradientDescent(stochasticGradientDescent &&) = default;
  stochasticGradientDescent &operator=(stochasticGradientDescent &&) = default;

  void preUpdateParams();
  void updateParams(denseLayer &layer) const;
  void postUpdateParams();

  double currentLearningRate() const { return m_currLearningRate; }
};
