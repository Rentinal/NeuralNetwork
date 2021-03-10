// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <math.h>
#include "activationFunctions/activationSoftmax.hpp"
#include "activationFunctions/activationReLU.hpp"
#include "lossFunctions/categoricalLossEntropy.hpp"
#include "lossFunctions/accuracy.hpp"

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

	const nc::NdArray<double>& weights() const {
		return m_weights;
	}

	const nc::NdArray<double>& biases() const {
		return m_biases;
	}

	void addToWeights(const nc::NdArray<double>& weights) {
		m_weights += weights;
	}

	void addToBiases(const nc::NdArray<double>& biases) {
		m_biases += biases;
	}

	void setWeights(const nc::NdArray<double>& weights) {
		m_weights = weights;
	}

	void setBiases(const nc::NdArray<double>& biases) {
		m_biases = biases;
	}

	const nc::NdArray<double>& output() const {
		return m_outputs;
	}

	void printOutput() const {
		std::cout << "Outputs\n" << m_outputs << '\n';
	}
};

int main()
{
	//Create random dataset
	auto [X, y] = utils::vertical_data(100, 3);

		//First Dense Layer with 2 inputs and 3 Neurons(output values)
	denseLayer dense1(2, 3);

	//Create ReLU activation (used in hidden layers)
	activationReLU activation1;

	//Create second Dense Layer with 3 inputs (Num outputs of first layer)
	//and 3 outputs (Num of layers in dataset)
	denseLayer dense2(3, 3);

	//Create Softmax Activation (used in output layer)
	activationSoftmax activation2;

	//Pass Output of second activation function through loss function
	categoricalCrossEntropy lossFunction;
	double lowestLoss = 9999999.0;
	auto bestDense1Weights = dense1.weights().copy();
	auto bestDense1Biases = dense1.biases().copy();
	auto bestDense2Weights = dense2.weights().copy();
	auto bestDense2Biases = dense2.biases().copy();

	for (size_t i = 0; i < 10000; i++)
	{
		dense1.addToWeights(0.05 * nc::random::randN<double>({ 2,3 }));
		dense1.addToBiases(0.05 * nc::random::randN<double>({ 1,3 }));
		dense2.addToWeights(0.05 * nc::random::randN<double>({ 3,3 }));
		dense2.addToBiases(0.05 * nc::random::randN<double>({ 1,3 }));

		//Pass Dataset through first layer
		dense1.forward(utils::Z);

		//Pass Output of first layer through first activation function
		activation1.forward(dense1.output());

		//Pass Output of first activation function through second neuron layer
		dense2.forward(activation1.output());

		//Pass Output of second layer through second activation function
		activation2.forward(dense2.output());

		//Average loss
		double loss = lossFunction.calculate(activation2.output(), y);

		//Calculate accuracy from output of the second activation function layer
		accuracy accuracy;
		accuracy.calculate(activation2.output(), y);

		if (loss < lowestLoss) {
			std::cout << "New Set of weights found, iteration: " << i << " loss: " << loss
				<< " acc: " << accuracy.output() << '\n';
			bestDense1Weights = dense1.weights().copy();
			bestDense1Biases = dense1.biases().copy();
			bestDense2Weights = dense2.weights().copy();
			bestDense2Biases = dense2.biases().copy();
			lowestLoss = loss;
		}
		else {
			dense1.setWeights(bestDense1Weights.copy());
			dense1.setBiases(bestDense1Biases.copy());
			dense2.setWeights(bestDense2Weights.copy());
			dense2.setBiases(bestDense2Biases.copy());
		}
		std::cout << "current Line: " << i << "\r";
	}







}



