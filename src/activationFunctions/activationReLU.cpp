#include "nnfspch.h"
#include "activationReLU.hpp"

void activationReLU::forward(const dMatrix &inputs)
{
  m_input = inputs;
  m_output = nc::maximum(nc::zeros<double>(inputs.shape()), inputs);
}


void activationReLU::backward(const dMatrix &dValues)
{
  m_dInput = dValues.copy();

  std::transform(m_dInput.begin(), m_dInput.end(), m_input.begin(), m_dInput.begin(), [](double val1, double val2) {
    return val2 <= 0 ? 0 : val1;
  });
}
