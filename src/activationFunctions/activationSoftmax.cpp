#include "nnfspch.h"
#include "activationSoftmax.hpp"

void activationSoftmax::forward(const dMatrix &inputs)
{
  //Retrieve Maximum Value of Inputs
  auto max = inputs.max().at(0, 0);
  //Subtract maximum Value from every Input ->
  //Prevent exploding Values in Exponential Function
  auto expValues = nc::exp(inputs - max);

  //Sum over every Row
  auto sum = nc::sum(expValues, nc::Axis::COL).transpose();
  //Normalize every Input Row with its according Sum
  m_output = utils::normalizeInputData(expValues, sum);
}


void activationSoftmax::backward(const dMatrix &dValues)
{
  m_dInput = nc::empty_like(dValues);
  dMatrix values = dValues.copy();

  for (int32_t i = 0; i < int32_t(m_output.numRows()); i++) {
    //Flatten output Array
    dMatrix singleOutput = m_output.row(i).reshape(-1, 1);
    //Calculate Jacobi Matrix of Output
    dMatrix jacobianMatrix = nc::diagflat(singleOutput) - singleOutput.dot(singleOutput.transpose());
    //Sample-wise gradient
    dMatrix outputVector = values.row(i).dot(jacobianMatrix);
    for (int32_t j = 0; j < int32_t(outputVector.numCols()); j++) {
      m_dInput.put(i, j, outputVector.at(0, j));
    }
  }
}
