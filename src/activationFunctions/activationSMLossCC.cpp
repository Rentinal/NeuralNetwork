#include "nnfspch.h"
#include "activationSMLossCC.hpp"

activationSMLossCC::activationSMLossCC()
{
  m_activation = activationSoftmax();
  m_loss = categoricalCrossEntropy();
}


double activationSMLossCC::forward(const dMatrix &inputs, const uiMatrix &yTrue)
{
  //Output Layer Activation Function
  m_activation.forward(inputs);
  //Save output
  m_output = m_activation.output();
  //Calculate and return loss
  return m_loss.calculate(m_output, yTrue);
}
void activationSMLossCC::backward(const dMatrix &dValues, const uiMatrix &yTrue)
{
  uint32_t numSamples = dValues.numRows();
  uiMatrix indices = yTrue.copy();

  if (indices.numRows() > 2) {
    indices = nc::argmax(indices, nc::Axis::COL);
  }

  m_dInput = dValues.copy();
  //Calculate Gradient
  for (int32_t i = 0; i < static_cast<int32_t>(m_dInput.numRows()); i++) {
    m_dInput.at(i, static_cast<int32_t>(yTrue.at(0, i)))--;
  }

  //Normalize Gradient
  std::transform(m_dInput.begin(), m_dInput.end(), m_dInput.begin(), [&numSamples](double val1) {
    return val1 / numSamples;
  });
}
