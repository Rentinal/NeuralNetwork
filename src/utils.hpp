#pragma once
#include "nnfspch.h"

using dMatrix = nc::NdArray<double>;
using iMatrix = nc::NdArray<int32_t>;
using uiMatrix = nc::NdArray<uint32_t>;

namespace utils {
double random(double min, double max);

std::tuple<dMatrix, uiMatrix> spiral_data(int32_t points, int32_t classes);

dMatrix addVectorToEveryRow(const dMatrix &m, const dMatrix &row);

dMatrix divideRowByVector(const dMatrix &m, const dMatrix &row);

dMatrix divideMatrices(const dMatrix &m1, const dMatrix &m2);

dMatrix normalizeInputData(const dMatrix &inputs, const dMatrix &sum);

}// namespace utils
