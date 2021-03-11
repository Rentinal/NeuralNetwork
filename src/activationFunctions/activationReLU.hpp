#pragma once
#include "./utils.hpp"

class activationReLU
{
private:
  //Forward Pass
  dMatrix m_output;
  dMatrix m_input;
  //Backward Pass
  dMatrix m_dInput;

public:
  activationReLU() = default;
  ~activationReLU() = default;
  activationReLU(const activationReLU &) = default;
  activationReLU &operator=(activationReLU const &) = default;
  activationReLU(activationReLU &&) = default;
  activationReLU &operator=(activationReLU &&) = default;

  //Activation Function that clips all negative Values to 0
  void forward(const dMatrix &inputs);
  void backward(const dMatrix &dValues);

  //non-normalized output
  [[nodiscard]] const dMatrix &output() const { return m_output; }
  [[nodiscard]] const dMatrix &dInput() const { return m_dInput; }
};
