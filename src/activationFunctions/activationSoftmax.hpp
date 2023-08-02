#pragma once
#include "./utils.hpp"

class activationSoftmax
{
private:
  dMatrix m_output;
  dMatrix m_dInput;

public:
  activationSoftmax() = default;
  ~activationSoftmax() = default;
  activationSoftmax(const activationSoftmax &) = default;
  activationSoftmax &operator=(activationSoftmax const &) = default;
  activationSoftmax(activationSoftmax &&) = default;
  activationSoftmax &operator=(activationSoftmax &&) = default;

  //Activation for the output Layer
  void forward(const dMatrix &inputs);
  void backward(const dMatrix &dValues);

  //Normalized output -> determines the output confidence of the network
  [[nodiscard]] const dMatrix &output() const { return m_output; }
  [[nodiscard]] const dMatrix &dInput() const { return m_dInput; }
  void setOutput(const dMatrix &output) { m_output = output; };
};
