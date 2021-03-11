#pragma once

class loss
{
public:
  loss() = default;
  virtual ~loss() = default;
  loss(const loss &) = default;
  loss &operator=(loss const &) = default;
  loss(loss &&) = default;
  loss &operator=(loss &&) = default;

  virtual nc::NdArray<double> forward(const nc::NdArray<double> &, const nc::NdArray<uint32_t> &) = 0;

  double calculate(const nc::NdArray<double> &output, const nc::NdArray<uint32_t> &y)
  {
    //Calculates the average loss
    return nc::mean(this->forward(output, y)).at(0, 0);
  }
};
