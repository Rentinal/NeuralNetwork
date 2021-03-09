#pragma once
#include "../utils.hpp"

class activationReLU {
private:
	nc::NdArray<double> m_output;

public:
	activationReLU() = default;
	~activationReLU() = default;

	//Activation Function that clips all negative Values to 0
	void forward(const nc::NdArray<double>& inputs);

	//non-normalized output
	const nc::NdArray<double>& output() const;
};

