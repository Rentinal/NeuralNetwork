#include "nnfspch.h"
#include "stochasticGradientDescent.hpp"
#include "denseLayer.hpp"

stochasticGradientDescent::stochasticGradientDescent(const double learningRate, const double decay)
  : m_learningRate(learningRate), m_currLearningRate(m_learningRate), m_decay(decay)
{
}

void stochasticGradientDescent::preUpdateParams()
{
  if (m_decay > 0.0) {
    m_currLearningRate = m_learningRate * (1.0 / (1.0 + m_decay * static_cast<double>(m_iterations)));
  }
}

void stochasticGradientDescent::updateParams(denseLayer &layer) const
{
  layer.addToWeights(-m_currLearningRate * layer.dWeights());
  layer.addToBiases(-m_currLearningRate * layer.dBiases());
}

void stochasticGradientDescent::postUpdateParams()
{
  m_iterations++;
}
