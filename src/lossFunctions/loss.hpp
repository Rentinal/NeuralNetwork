#pragma once
#include "../utils.hpp"

class loss {
public:
	virtual nc::NdArray<double> forward(const nc::NdArray<double>&, const nc::NdArray<size_t>&) = 0;

	double calculate(const nc::NdArray<double>& output, const nc::NdArray<size_t>& y)
	{
		//Calculates the average loss
		return nc::mean(this->forward(output, y)).at(0, 0);
	}

};