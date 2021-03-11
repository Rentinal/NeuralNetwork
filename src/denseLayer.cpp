#include "nnfspch.h"
#include "denseLayer.hpp"

denseLayer::denseLayer(const uint32_t numInputs, const uint32_t numNeurons)
{
  constexpr double RANGE = 0.01;
  //Random values between -0.01 and 0.01
  m_weights = RANGE * nc::random::randN<double>({ numInputs, numNeurons });
  m_biases = nc::zeros<double>({ 1, numNeurons });
}

denseLayer::denseLayer(const dMatrix &weights, const dMatrix &biases)
{
  m_weights = weights.copy();
  m_biases = biases.copy();
}

void denseLayer::forward(const dMatrix &inputs)
{
  m_inputs = inputs;
  m_outputs = utils::addVectorToEveryRow(m_inputs.dot(m_weights), m_biases);
}

void denseLayer::backward(const dMatrix &dValues)
{
  //Gradient on Parameters
  m_dWeights = m_inputs.transpose().dot(dValues);
  m_dBiases = nc::sum(dValues, nc::Axis::ROW);
  //Gradient on Values
  m_dInputs = dValues.dot(m_weights.transpose());
}


void denseLayer::addToWeights(const dMatrix &weights)
{
  m_weights += weights;
}

void denseLayer::addToBiases(const dMatrix &biases)
{
  m_biases += biases;
}

void denseLayer::setWeights(const dMatrix &weights)
{
  m_weights = weights;
}

void denseLayer::setBiases(const dMatrix &biases)
{
  m_biases = biases;
}
