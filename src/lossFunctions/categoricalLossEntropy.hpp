#pragma once
#include "loss.hpp"

class categoricalCrossEntropy : public loss
{
public:
  categoricalCrossEntropy() = default;
  ~categoricalCrossEntropy() override = default;
  categoricalCrossEntropy(const categoricalCrossEntropy &) = default;
  categoricalCrossEntropy &operator=(categoricalCrossEntropy const &) = default;
  categoricalCrossEntropy(categoricalCrossEntropy &&) = default;
  categoricalCrossEntropy &operator=(categoricalCrossEntropy &&) = default;

  nc::NdArray<double> forward(const nc::NdArray<double> &yPred, const nc::NdArray<uint32_t> &yTrue) override;
};
