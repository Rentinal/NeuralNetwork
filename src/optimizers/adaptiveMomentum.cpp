#include "nnfspch.h"
#include "adaptiveMomentum.hpp"
#include "denseLayer.hpp"

adaptiveMomentum::adaptiveMomentum(double learningRate, double decay, double epsilon, double beta1, double beta2)
  : optimizer(learningRate, decay), m_epsilon(epsilon), m_beta1(beta1), m_beta2(beta2)
{
}

void adaptiveMomentum::updateParams(denseLayer &layer) const
{
  //Update momentum with current gradients
  layer.setWeightMomentums(m_beta1 * layer.weightMomentums() + (1.0 - m_beta1) * layer.dWeights());
  layer.setBiasMomentums(m_beta1 * layer.biasMomentums() + (1.0 - m_beta1) * layer.dBiases());

  double denom = 1.0 - std::pow(m_beta1, static_cast<double>(getIterations() + 1));
  //Get corrected momentum
  //Iteration at 0 on first pass
  dMatrix weightMomentumsCorrected = layer.weightMomentums() / denom;
  dMatrix biasMomentumsCorrected = layer.biasMomentums() / denom;

  //Update cache with squared current gradients
  layer.setWeightCache(m_beta2 * layer.weightCache() + (1.0 - m_beta2) * nc::square(layer.dWeights()));
  layer.setBiasCache(m_beta2 * layer.biasCache() + (1.0 - m_beta2) * nc::square(layer.dBiases()));

  //Get corrected Cache
  denom = 1.0 - std::pow(m_beta2, static_cast<double>(getIterations() + 1));
  dMatrix weightCacheCorrected = layer.weightCache() / denom;
  dMatrix biasCacheCorrected = layer.biasCache() / denom;

  //Vanilla SGD + normalization with square rooted cache
  layer.addToWeights(-getCurrentLearningRate() * weightMomentumsCorrected / (nc::sqrt(weightCacheCorrected) + m_epsilon));
  layer.addToBiases(-getCurrentLearningRate() * biasMomentumsCorrected / (nc::sqrt(biasCacheCorrected) + m_epsilon));
}
