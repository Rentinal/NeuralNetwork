#include "nnfspch.h"
#include "stochasticGradientDescent.hpp"
#include "denseLayer.hpp"

stochasticGradientDescent::stochasticGradientDescent(const double learningRate)
{
  m_learningRate = learningRate;
}

void stochasticGradientDescent::updateParams(denseLayer &layer)
{
  layer.addToWeights(-1.0 * m_learningRate * layer.dWeights());
  layer.addToBiases(-1.0 * m_learningRate * layer.dBiases());
}
