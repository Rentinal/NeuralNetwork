#pragma once
#include "../utils.hpp"

class accuracy {
private:
	double m_accuracy;
public:
	void calculate(const nc::NdArray<double>&, const nc::NdArray<size_t>&);
	double output() const;
};

