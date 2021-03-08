// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "utils.hpp"


class denseLayer {
private:
    nc::NdArray<double> m_weights;
    nc::NdArray<double> m_biases;
public:
    nc::NdArray<double> M_outputs;

public:
    denseLayer(const size_t& numInputs, const size_t& numNeurons) {
        //Random values between -0.01 and 0.01
        m_weights = 0.01 * nc::random::randN<double>({ numInputs, numNeurons });
        m_biases = nc::zeros<double>({ 1, numNeurons });
    }

    ~denseLayer() = default;

    void forward(const nc::NdArray<double>& inputs) {
        M_outputs = utils::addVectorToEveryRow(inputs.dot(m_weights), m_biases);
    }

    void printOutput() const {
        std::cout << "Outputs\n" << M_outputs << '\n';
    }
};

int main()
{
    auto [X, y] = utils::spiral_data(100, 3);

   denseLayer dense1(2, 3);

   dense1.forward(X);

   dense1.printOutput();
}
