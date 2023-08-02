#include "nnfspch.h"
#include "rootMeanSquareProp.hpp"
#include "denseLayer.hpp"

rootMeanSquareProp::rootMeanSquareProp(double learningRate, double decay, double epsilon, double rho)
  : optimizer(learningRate, decay), m_epsilon(epsilon), m_rho(rho)
{
}

void rootMeanSquareProp::updateParams(denseLayer &layer) const
{
  //Update Cache with squared current gradients
  layer.setWeightCache(m_rho * layer.weightCache() + (1.0 - m_rho) * nc::square(layer.dWeights()));
  layer.setBiasCache(m_rho * layer.biasCache() + (1.0 - m_rho) * nc::square(layer.dBiases()));

  //Vanilla SGD + normalization with square rooted cache
  layer.addToWeights(-getCurrentLearningRate() * layer.dWeights() / (nc::sqrt(layer.weightCache()) + m_epsilon));
  layer.addToBiases(-getCurrentLearningRate() * layer.dBiases() / (nc::sqrt(layer.biasCache()) + m_epsilon));
}
