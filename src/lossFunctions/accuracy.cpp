#include "nnfspch.h"
#include "accuracy.hpp"


void accuracy::calculate(const dMatrix &output, const uiMatrix &targets)
{
  m_predictions = nc::argmax(output, nc::Axis::COL);
  uiMatrix tempTargets = targets.copy();
  if (tempTargets.numRows() > 1) {
    tempTargets = nc::argmax(tempTargets, nc::Axis::COL);
  }

  uiMatrix accuracy = nc::zeros<uint32_t>(m_predictions.shape());
  std::transform(m_predictions.begin(), m_predictions.end(), tempTargets.begin(), accuracy.begin(), [](uint32_t first, uint32_t sec) {
    return first == sec;
  });

  m_accuracy = nc::mean(accuracy).at(0, 0);
}
