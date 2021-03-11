#pragma once
#include "loss.hpp"

class categoricalCrossEntropy : public loss
{
private:
  dMatrix m_dInput;

public:
  categoricalCrossEntropy() = default;
  ~categoricalCrossEntropy() override = default;
  categoricalCrossEntropy(const categoricalCrossEntropy &) = default;
  categoricalCrossEntropy &operator=(categoricalCrossEntropy const &) = default;
  categoricalCrossEntropy(categoricalCrossEntropy &&) = default;
  categoricalCrossEntropy &operator=(categoricalCrossEntropy &&) = default;

  dMatrix forward(const dMatrix &yPred, const uiMatrix &yTrue) override;
  void backward(const dMatrix &dValues, const uiMatrix &yTrue);

  const dMatrix &dInput() const { return m_dInput; }
};
