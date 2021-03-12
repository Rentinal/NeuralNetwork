#include "nnfspch.h"
#include "stochasticGradientDescent.hpp"
#include "denseLayer.hpp"

stochasticGradientDescent::stochasticGradientDescent(const double learningRate, const double decay, const double momentum)
  : optimizer(learningRate, decay), m_momentum(momentum)
{
}

void stochasticGradientDescent::updateParams(denseLayer &layer) const
{
  dMatrix weightUpdates;
  dMatrix biasUpdates;
  //With momentum
  if (m_momentum > 0.0) {
    //Weight updates with momentum
    //previous update multiplied by retain factor and update it with current gradients
    weightUpdates = m_momentum * layer.weightMomentums()
                    - getCurrentLearningRate() * layer.dWeights();
    layer.setWeightMomentums(weightUpdates);

    biasUpdates = m_momentum * layer.biasMomentums()
                  - getCurrentLearningRate() * layer.dBiases();
    layer.setBiasMomentums(biasUpdates);
    //Vanilla SGD update
  } else {
    weightUpdates = -getCurrentLearningRate() * layer.dWeights();

    biasUpdates = -getCurrentLearningRate() * layer.dBiases();
  }

  //Either Add vanilla oder momentum updates
  layer.addToBiases(biasUpdates);
  layer.addToWeights(weightUpdates);
}
