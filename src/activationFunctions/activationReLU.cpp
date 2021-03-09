#include "activationReLU.hpp"


void activationReLU::forward(const nc::NdArray<double>& inputs) {
	m_output = nc::maximum(nc::zeros<double>(inputs.shape()), inputs);
}

const nc::NdArray<double>& activationReLU::output() const {
	return m_output;
}