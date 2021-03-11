#pragma once
#include "../utils.hpp"

class activationSoftmax
{
private:
  nc::NdArray<double> m_output;

public:
  activationSoftmax() = default;
  ~activationSoftmax() = default;
  activationSoftmax(const activationSoftmax &) = default;
  activationSoftmax &operator=(activationSoftmax const &) = default;
  activationSoftmax(activationSoftmax &&) = default;
  activationSoftmax &operator=(activationSoftmax &&) = default;

  //Activation for the output Layer
  void forward(const nc::NdArray<double> &);

  //Normalized output -> determines the output confidence of the network
  [[nodiscard]] const nc::NdArray<double> &output() const;
};
