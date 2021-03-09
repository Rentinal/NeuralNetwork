// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <math.h>
#include "utils.hpp"

//Euler's number calculated -> double precision
static const double E = std::exp(1.0);

class denseLayer {
private:
	nc::NdArray<double> m_weights;
	nc::NdArray<double> m_biases;
	nc::NdArray<double> m_outputs;

public:
	denseLayer(const size_t& numInputs, const size_t& numNeurons) {
		//Random values between -0.01 and 0.01
		m_weights = 0.01 * nc::random::randN<double>({ numInputs, numNeurons });
		m_biases = nc::zeros<double>({ 1, numNeurons });
	}

	~denseLayer() = default;

	//sum(weights * input) + bias
	void forward(const nc::NdArray<double>& inputs) {
		m_outputs = utils::addVectorToEveryRow(inputs.dot(m_weights), m_biases);
	}

	const nc::NdArray<double>& output() const {
		return m_outputs;
	}

	void printOutput() const {
		std::cout << "Outputs\n" << m_outputs << '\n';
	}
};

//Activation Function that clips all negative Values to 0
//non-normalized output
class activationReLU {
private:
	nc::NdArray<double> m_output;

public:
	activationReLU() = default;
	~activationReLU() = default;

	void forward(const nc::NdArray<double>& inputs) {
		m_output = nc::maximum(nc::zeros<double>(inputs.shape()), inputs);
	}

	const nc::NdArray<double>& output() const {
		return m_output;
	}
};

//Activation for the output Layer
//Normalized output -> determines the output confidence of the network
class activationSoftmax {
private:
	nc::NdArray<double> m_output;

public:
	activationSoftmax() = default;
	~activationSoftmax() = default;

	void forward(const nc::NdArray<double>& inputs) {
		//Retrieve Maximum Value of Inputs
		auto max = inputs.max().at(0,0);
		//Subtract maximum Value from every Input ->
		//Prevent exploding Values in Exponential Function
		auto expValues = nc::exp(inputs - max);

		//Sum over every Row
		auto sum = nc::sum(expValues, nc::Axis::COL).transpose();
		//Normalize every Input Row with its according Sum
		m_output = utils::normalizeInputData(expValues, sum);
	}

	const nc::NdArray<double>& output() const {
		return m_output;
	}
};

int main()
{
	//Create random dataset
	auto [X, y] = utils::spiral_data(5, 3);

	//First Dense Layer with 2 inputs and 3 Neurons(output values)
	denseLayer dense1(2, 3);

	//Create ReLU activation (used in hidden layers)
	activationReLU activation1;

	//Create second Dense Layer with 3 inputs (Num outputs of first layer)
	//and 3 outputs (Num of layers in dataset)
	denseLayer dense2(3, 3);
	
	//Create Softmax Activation (used in output layer)
	activationSoftmax activation2;

	//Pass Dataset through first layer
	dense1.forward(X);

	//Pass Output of first layer through first activation function
	activation1.forward(dense1.output());

	//Pass Output of first activation function through second neuron layer
	dense2.forward(activation1.output());

	//Pass Output of second layer through second activation function
	activation2.forward(dense2.output());

	//Every Output has same confidence
	std::cout << "Output\n" << activation2.output();

}
