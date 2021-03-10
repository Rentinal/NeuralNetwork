#include "accuracy.hpp"

void accuracy::calculate(const nc::NdArray<double>& output, const nc::NdArray<size_t>& targets) {
	nc::NdArray<size_t> predictions = nc::argmax(output, nc::Axis::COL);
	nc::NdArray<size_t> tempTargets = targets;
	if (tempTargets.numRows() > 1) {
		tempTargets = nc::argmax(tempTargets, nc::Axis::COL);
	}
	nc::NdArray<double> accuracy = nc::mean(predictions == tempTargets);
	m_accuracy = accuracy.at(0, 0);
}
double accuracy::output() const {
	return m_accuracy;
}