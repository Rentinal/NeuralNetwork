#include "categoricalLossEntropy.hpp"

nc::NdArray<double> categoricalCrossEntropy::forward(const nc::NdArray<double>& yPred, const nc::NdArray<int>& yTrue)
{
	size_t numSamples = yPred.numRows();

	//Clip data to prevent division by 0
	//Clip both sides to keep its integrity
	nc::NdArray<double> yPredClipped = nc::clip(yPred, 1e-7, 1e+7);

	nc::NdArray<double> confidences = nc::zeros<double>({ 1, numSamples });

	//Calculate confidences
	if (yTrue.numRows() > 1) {
		for (size_t i = 0; i < yPredClipped.numRows(); i++)
		{
			double confidence = 0.0;
			for (size_t j = 0; j < yPredClipped.numCols(); j++)
			{
				confidence += yPredClipped.at(i, j) * yTrue.at(i, j);
			}
			confidences.put(0, i, confidence);
		}
	}
	else {
		std::for_each(yTrue.begin(), yTrue.end(), [&yPredClipped, idx = 0, &confidences](int const& max) mutable{
			confidences.put(0, idx, yPredClipped.at(idx, max));
			idx++;
		});
	}

	//Calculate losses
	return -1.0 * nc::log(confidences);
}