#include "nnfspch.h"
#include "categoricalLossEntropy.hpp"


dMatrix categoricalCrossEntropy::forward(const dMatrix &yPred, const uiMatrix &yTrue)
{
  constexpr double MIN = 1e-7;
  constexpr double MAX = 1e+7;
  uint32_t numSamples = yPred.numRows();

  //Clip data to prevent division by 0
  //Clip both sides to keep its integrity
  dMatrix yPredClipped = nc::clip(yPred, MIN, MAX);

  dMatrix confidences = nc::zeros<double>({ 1, numSamples });

  //Calculate confidences
  if (yTrue.numRows() > 1) {
    for (int32_t i = 0; i < static_cast<int32_t>(yPredClipped.numRows()); i++) {
      double confidence = 0.0;
      for (int32_t j = 0; j < static_cast<int32_t>(yPredClipped.numCols()); j++) {
        confidence += yPredClipped.at(i, j) * yTrue.at(i, j);
      }
      confidences.put(0, i, confidence);
    }
  } else {
    std::for_each(yTrue.begin(), yTrue.end(), [&yPredClipped, idx = 0, &confidences](uint32_t max) mutable {
      confidences.put(0, idx, yPredClipped.at(idx, static_cast<int32_t>(max)));
      idx++;
    });
  }

  //Calculate losses
  return -1.0 * nc::log(confidences);
}


void categoricalCrossEntropy::backward(const dMatrix &dValues, const uiMatrix &yTrue)
{
  uint32_t numSamples = dValues.numRows();
  uint32_t numLables = dValues.numCols();

  dMatrix indices = nc::zeros<double>({ numLables, numLables });
  for (int32_t i = 0; i < static_cast<int32_t>(yTrue.numCols()); i++) {
    indices.put(i, static_cast<int32_t>(yTrue(0, i)), 1.0);
  }

  //Calculate Gradient
  m_dInput = utils::divideMatrices(-1.0 * indices, dValues);

  //Normalize Gradient
  std::transform(m_dInput.begin(), m_dInput.end(), m_dInput.begin(), [&numSamples](const double &val1) {
    return val1 / numSamples;
  });
}
