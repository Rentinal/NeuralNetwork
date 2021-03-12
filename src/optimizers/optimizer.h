#pragma once
class denseLayer;

class optimizer
{
private:
  double m_learningRate;
  double m_currLearningRate;
  double m_decay;
  size_t m_iterations{ 0 };

protected:
  void setLearningRate(const double lr) { m_learningRate = lr; }
  void setCurrentLearningRate(const double lr) { m_currLearningRate = lr; }
  void setDecay(const double decay) { m_decay = decay; }

  [[nodiscard]] double getLearningRate() const { return m_learningRate; }
  [[nodiscard]] double getCurrentLearningRate() const { return m_currLearningRate; }
  [[nodiscard]] double getDecay() const { return m_decay; }

public:
  explicit optimizer(const double learningRate = 1.0, const double decay = 1e-3)
    : m_learningRate(learningRate), m_currLearningRate(m_learningRate), m_decay(decay) {}
  virtual ~optimizer() = default;
  optimizer(const optimizer &) = default;
  optimizer &operator=(optimizer const &) = default;
  optimizer(optimizer &&) = default;
  optimizer &operator=(optimizer &&) = default;

  void preUpdateParams()
  {
    if (m_decay > 0.0) {
      m_currLearningRate = m_learningRate * (1.0 / (1.0 + m_decay * static_cast<double>(m_iterations)));
    }
  }

  virtual void updateParams(denseLayer &layer) const = 0;

  void postUpdateParams()
  {
    m_iterations++;
  }

  [[nodiscard]] double currentLearningRate() const { return m_currLearningRate; }
};
