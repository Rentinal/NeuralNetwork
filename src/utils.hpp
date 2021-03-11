#pragma once
#include "nnfspch.h"

namespace utils {
double random(double min, double max);

std::tuple<nc::NdArray<double>, nc::NdArray<uint32_t>> spiral_data(int32_t points, int32_t classes);

nc::NdArray<double> addVectorToEveryRow(const nc::NdArray<double> &m, const nc::NdArray<double> &row);

nc::NdArray<double> divideRowByVector(const nc::NdArray<double> &m, const nc::NdArray<double> &row);

nc::NdArray<double> normalizeInputData(const nc::NdArray<double> &inputs, const nc::NdArray<double> &sum);

}// namespace utils
