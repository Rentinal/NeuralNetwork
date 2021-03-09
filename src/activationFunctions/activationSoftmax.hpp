#pragma once
#include "../utils.hpp"

class activationSoftmax
{
private:
	nc::NdArray<double> m_output;

public:
	activationSoftmax() = default;
	~activationSoftmax() = default;

	//Activation for the output Layer
	void forward(const nc::NdArray<double>&);

	//Normalized output -> determines the output confidence of the network
	const nc::NdArray<double>& output() const;
};
