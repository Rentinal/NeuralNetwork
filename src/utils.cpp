#include "nnfspch.h"
#include "utils.hpp"

double utils::random(const double min, const double max)
{
  std::random_device dev;
  std::mt19937_64 rng(dev());
  return std::uniform_real_distribution<>{ min, max }(rng);
}

std::tuple<dMatrix, uiMatrix> utils::spiral_data(const int32_t points, const int32_t classes)
{
  constexpr double OFFSET = 0.15;
  constexpr double RANGE = 2.5;

  dMatrix X = nc::zeros<double>({ uint32_t(points * classes), 2 });
  uiMatrix y = nc::zeros<uint32_t>({ uint32_t(points * classes), 1 });

  double r = 0.0;
  double t = 0.0;

  for (int32_t i = 0; i < classes; i++) {
    for (int32_t j = 0; j < points; j++) {
      r = double(j) / double(points);
      t = i * 4 + (4 * r);
      dMatrix a{ r * cos(t * RANGE), r * sin(t * RANGE) };
      dMatrix b{ random(-OFFSET, OFFSET), random(-OFFSET, OFFSET) };
      dMatrix value = a + b;
      X.put(i * points + j, 0, value[0]);
      X.put(i * points + j, 1, value[1]);
      y.put(i * points + j, 0, uint32_t(i));
    }
  }

  return std::make_tuple(X, y.flatten());
}

//Operator to Add a Vector to every row of a matrix
dMatrix utils::addVectorToEveryRow(const dMatrix &m, const dMatrix &row)
{
  dMatrix result(m.numRows(), m.numCols());

  for (int32_t j = 0; j < int32_t(m.numRows()); j++) {
    for (int32_t i = 0; i < int32_t(m.numCols()); i++) {
      result.put(j, i, m.at(j, i) + row[i]);
    }
  }
  return result;
}

//Operator to Add a Vector to every row of a matrix
dMatrix utils::divideRowByVector(const dMatrix &m, const dMatrix &row)
{
  dMatrix result(m.numRows(), m.numCols());
  for (int32_t j = 0; j < int32_t(m.numRows()); j++) {
    for (int32_t i = 0; i < int32_t(m.numCols()); i++) {
      result.put(j, i, m.at(j, i) / row[i]);
    }
  }
  return result;
}


dMatrix utils::divideMatrices(const dMatrix &m1, const dMatrix &m2)
{
  dMatrix result(m1.shape());
  std::transform(m1.begin(), m1.end(), m2.begin(), result.begin(), [](double val1, double val2) {
    return val1 / val2;
  });
  return result;
}

dMatrix utils::normalizeInputData(const dMatrix &inputs, const dMatrix &sum)
{
  dMatrix result(inputs.shape());

  for (int32_t i = 0; i < int32_t(result.numRows()); i++) {
    for (int32_t j = 0; j < int32_t(result.numCols()); j++) {
      result.put(i, j, inputs.at(i, j) / sum.at(i, 0));
    }
  }

  return result;
}
