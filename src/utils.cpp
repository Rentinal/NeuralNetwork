#include <tuple>
#include <random>
#include "utils.hpp"

    double utils::random(const double& min, const double& max) {
        std::mt19937_64 rng{};
        rng.seed(std::random_device{}());
        return std::uniform_real_distribution<>{min, max}(rng);
    }

    std::tuple<nc::NdArray<double>, nc::NdArray<double>> utils::spiral_data(const size_t& points, const size_t& classes) {
        nc::NdArray<double> X = nc::zeros<double>({ points * classes, 2 });
        nc::NdArray<double> y = nc::zeros<double>({ points * classes, 1 });

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


        return std::make_tuple(X, y);
    }

    //Operator to Add a Vector to every row of a matrix
    nc::NdArray<double> utils::addVectorToEveryRow(const nc::NdArray<double>& m, const nc::NdArray<double>& row) {
        nc::NdArray<double> result(m.numRows(), m.numCols());
        for (size_t j = 0; j < m.numRows(); j++) {
            for (size_t i = 0; i < m.numCols(); i++) {
                result.put(j, i, m.at(j, i) + row[i]);
            }
        }
        return result;
    }
