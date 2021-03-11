#pragma once
#include "utils.hpp"

class accuracy
{
private:
  double m_accuracy{ 0.0 };
  uiMatrix m_predictions;

public:
  accuracy() = default;
  ~accuracy() = default;
  accuracy(const accuracy &) = default;
  accuracy &operator=(accuracy const &) = default;
  accuracy(accuracy &&) = default;
  accuracy &operator=(accuracy &&) = default;

  void calculate(const dMatrix &, const uiMatrix &);
  [[nodiscard]] double output() const { return m_accuracy; }
  [[nodiscard]] const uiMatrix &prediction() const { return m_predictions; }
};
