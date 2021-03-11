#include "nnfspch.h"
#include "accuracy.hpp"


void accuracy::calculate(const nc::NdArray<double> &output, const nc::NdArray<uint32_t> &targets)
{
  nc::NdArray<uint32_t> predictions = nc::argmax(output, nc::Axis::COL);
  nc::NdArray<uint32_t> tempTargets = targets.copy();
  if (tempTargets.numRows() > 1) {
    tempTargets = nc::argmax(tempTargets, nc::Axis::COL);
  }

  nc::NdArray<uint32_t> accuracy = nc::zeros<uint32_t>(predictions.shape());
  std::transform(predictions.begin(), predictions.end(), tempTargets.begin(), accuracy.begin(), [](uint32_t first, uint32_t sec) {
    return first == sec;
  });

  m_accuracy = nc::mean(accuracy).at(0, 0);
}
double accuracy::output() const
{
  return m_accuracy;
}
