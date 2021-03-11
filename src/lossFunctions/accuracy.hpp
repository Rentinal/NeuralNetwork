#pragma once

class accuracy
{
private:
  double m_accuracy{ 0.0 };

public:
  accuracy() = default;
  ~accuracy() = default;
  accuracy(const accuracy &) = default;
  accuracy &operator=(accuracy const &) = default;
  accuracy(accuracy &&) = default;
  accuracy &operator=(accuracy &&) = default;

  void calculate(const nc::NdArray<double> &, const nc::NdArray<uint32_t> &);
  [[nodiscard]] double output() const;
};
