#pragma once

class activationReLU
{
private:
  nc::NdArray<double> m_output;

public:
  activationReLU() = default;
  ~activationReLU() = default;
  activationReLU(const activationReLU &) = default;
  activationReLU &operator=(activationReLU const &) = default;
  activationReLU(activationReLU &&) = default;
  activationReLU &operator=(activationReLU &&) = default;

  //Activation Function that clips all negative Values to 0
  void forward(const nc::NdArray<double> &);

  //non-normalized output
  [[nodiscard]] const nc::NdArray<double> &output() const;
};
