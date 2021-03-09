#pragma once
#include "../utils.hpp"

class loss {
public:
	virtual nc::NdArray<double> forward(const nc::NdArray<double>&, const nc::NdArray<int>&) = 0;

	nc::NdArray<double> calculate(const nc::NdArray<double>& output, const nc::NdArray<int>& y)
	{
		//Calculates the average loss
		return nc::mean(this->forward(output, y));
	}

};