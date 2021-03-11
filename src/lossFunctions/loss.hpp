#pragma once
#include "./utils.hpp"

class loss
{
public:
  loss() = default;
  virtual ~loss() = default;
  loss(const loss &) = default;
  loss &operator=(loss const &) = default;
  loss(loss &&) = default;
  loss &operator=(loss &&) = default;

  virtual dMatrix forward(const dMatrix &, const uiMatrix &) = 0;

  double calculate(const dMatrix &output, const uiMatrix &y)
  {
    //Calculates the average loss
    return nc::mean(this->forward(output, y)).at(0, 0);
  }
};
