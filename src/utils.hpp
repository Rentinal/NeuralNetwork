#pragma once
#include <NumCpp/NumCpp.hpp>

namespace utils {
	double random(const double&, const double&);

	std::tuple<nc::NdArray<double>, nc::NdArray<double>> spiral_data(const size_t&, const size_t&);

	nc::NdArray<double> addVectorToEveryRow(const nc::NdArray<double>&, const nc::NdArray<double>&);

	nc::NdArray<double> divideRowByVector(const nc::NdArray<double>&, const nc::NdArray<double>&);

	nc::NdArray<double> normalizeInputData(const nc::NdArray<double>&, const nc::NdArray<double>&);
}