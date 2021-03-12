#include "nnfspch.h"
#include "adaptiveGradient.hpp"
#include "denseLayer.hpp"

adaptiveGradient::adaptiveGradient(const double learningRate, const double decay, const double epsilon)
  : optimizer(learningRate, decay), m_epsilon(epsilon)
{
}

void adaptiveGradient::updateParams(denseLayer &layer) const
{
  dMatrix weights = layer.dWeights();
  dMatrix biases = layer.dBiases();

  //Update Cache with squared Gradient
  layer.addToWeightCache(weights * weights);
  layer.addToBiasCache(biases * biases);

  //Vanilla SGD + normalization with square rooted cache
  layer.addToWeights(-getCurrentLearningRate() * weights / (nc::sqrt(layer.weightCache()) + m_epsilon));
  layer.addToBiases(-getCurrentLearningRate() * biases / nc::sqrt(layer.biasCache()) + m_epsilon);
}
