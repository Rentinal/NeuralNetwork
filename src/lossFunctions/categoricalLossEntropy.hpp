#pragma once
#include "loss.hpp"

class categoricalCrossEntropy : public loss {
public:
	nc::NdArray<double> forward(
		const nc::NdArray<double>& yPred, const nc::NdArray<size_t>& yTrue
	);
};