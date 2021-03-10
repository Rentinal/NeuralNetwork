#include "nnfspch.h"
#include "utils.hpp"

double utils::random(const double& min, const double& max) {
	std::mt19937_64 rng{};
	rng.seed(std::random_device{}());
	return std::uniform_real_distribution<>{min, max}(rng);
}

std::tuple<nc::NdArray<double>, nc::NdArray<size_t>> utils::spiral_data(const size_t& points, const size_t& classes) 
{
	nc::NdArray<double> X = nc::zeros<double>({ points * classes, 2 });
	nc::NdArray<size_t> y = nc::zeros<size_t>({ points * classes, 1 });

	double r, t;
	for (size_t i = 0; i < classes; i++) {
		for (size_t j = 0; j < points; j++) {
			r = double(j) / double(points);
			t = i * 4 + (4 * r);
			nc::NdArray<double> a{ r * cos(t * 2.5), r * sin(t * 2.5) };
			nc::NdArray<double> b{ random(-0.15, 0.15), random(-0.15, 0.15) };
			nc::NdArray<double> value = a + b;
			X.put(i * points + j, 0, value[0]);
			X.put(i * points + j, 1, value[1]);
			y.put(i * points + j, 0, i);
		}
	}

	return std::make_tuple(X, y.flatten());
}



std::tuple<nc::NdArray<double>, nc::NdArray<size_t>> utils::vertical_data(const size_t& points, const size_t& classes) {
	nc::NdArray<double> X = nc::zeros<double>({ points * classes, 2 });
	nc::NdArray<size_t> y = nc::zeros<size_t>({ points * classes, 1 });

	for (size_t i = 0; i < classes; i++)
	{
		nc::NdArray<size_t> values = nc::zeros<size_t>({ points, 1 });
		std::iota(values.begin(), values.end(), i * points);
		for (const auto& k : values) {
			nc::NdArray<double> randomValue = nc::random::randN<double>({ 1,2 });
			X.put(k, 0, randomValue.at(0, 0) * 0.1 + (classes / 3));
			X.put(k, 1, randomValue.at(0, 1) * 0.1 + 0.5);
			y.put(k, 0, i);
		}
	}

	return std::make_tuple(X,y.flatten());
}

//Operator to Add a Vector to every row of a matrix
nc::NdArray<double> utils::addVectorToEveryRow(const nc::NdArray<double>& m, const nc::NdArray<double>& row)
{
	nc::NdArray<double> result(m.numRows(), m.numCols());
	for (size_t j = 0; j < m.numRows(); j++) {
		for (size_t i = 0; i < m.numCols(); i++) {
			result.put(j, i, m.at(j, i) + row[i]);
		}
	}
	return result;
}

//Operator to Add a Vector to every row of a matrix
nc::NdArray<double> utils::divideRowByVector(const nc::NdArray<double>& m, const nc::NdArray<double>& row) 
{
	nc::NdArray<double> result(m.numRows(), m.numCols());
	for (size_t j = 0; j < m.numRows(); j++) {
		for (size_t i = 0; i < m.numCols(); i++) {
			result.put(j, i, m.at(j, i) / row[i]);
		}
	}
	return result;
}

nc::NdArray<double> utils::normalizeInputData(const nc::NdArray<double>& inputs, const nc::NdArray<double>& sum)
{
	nc::NdArray<double> result(inputs.shape());

	for (size_t i = 0; i < result.numRows(); i++)
	{
		for (size_t j = 0; j < result.numCols(); j++)
		{
			result.put(i, j, inputs.at(i, j) / sum.at(i, 0));
		}

	}

	return result;
}


