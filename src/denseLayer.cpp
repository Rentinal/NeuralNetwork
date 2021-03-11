#include "nnfspch.h"
#include "denseLayer.hpp"

denseLayer::denseLayer(const uint32_t numInputs, const uint32_t numNeurons)
{
  constexpr double RANGE = 0.01;
  //Random values between -0.01 and 0.01
  m_weights = RANGE * nc::random::randN<double>({ numInputs, numNeurons });
  m_biases = nc::zeros<double>({ 1, numNeurons });
}

void denseLayer::forward(const nc::NdArray<double> &inputs)
{
  m_outputs = utils::addVectorToEveryRow(inputs.dot(m_weights), m_biases);
}

const nc::NdArray<double> &denseLayer::weights() const
{
  return m_weights;
}

const nc::NdArray<double> &denseLayer::biases() const
{
  return m_biases;
}

void denseLayer::addToWeights(const nc::NdArray<double> &weights)
{
  m_weights += weights;
}

void denseLayer::addToBiases(const nc::NdArray<double> &biases)
{
  m_biases += biases;
}

void denseLayer::setWeights(const nc::NdArray<double> &weights)
{
  m_weights = weights;
}

void denseLayer::setBiases(const nc::NdArray<double> &biases)
{
  m_biases = biases;
}

const nc::NdArray<double> &denseLayer::output() const
{
  return m_outputs;
}
