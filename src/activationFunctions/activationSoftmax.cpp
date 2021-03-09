#include "activationSoftmax.hpp"

void activationSoftmax::forward(const nc::NdArray<double>& inputs) {
	//Retrieve Maximum Value of Inputs
	auto max = inputs.max().at(0, 0);
	//Subtract maximum Value from every Input ->
	//Prevent exploding Values in Exponential Function
	auto expValues = nc::exp(inputs - max);

	//Sum over every Row
	auto sum = nc::sum(expValues, nc::Axis::COL).transpose();
	//Normalize every Input Row with its according Sum
	m_output = utils::normalizeInputData(expValues, sum);
}

const nc::NdArray<double>& activationSoftmax::output() const {
	return m_output;
}