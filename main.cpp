#include <stdio.h>
#include <iostream>

#include "cublasNN.h"
#include "randinitweights.h"
#include "activations.h"
#include "costfunctions.h"

int main(int argc, char **argv)
{
	/*
	cout << "Neural Network Function Approximation Example" << endl;
	cublasNN *nn = new cublasNN;
	cout << "Read CSV: " << endl;
	float csvTime = 0;
	float csvTime2 = 0;
	vector<vector<float>> xVec = nn->readCSV("../bikeshare/trainP7_1.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> yVec = nn->readCSV("../bikeshare/trainYP2_1.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> xVecValidate = nn->readCSV("../bikeshare/validateP7_1.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> yVecValidate = nn->readCSV("../bikeshare/validateY1.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> xVecPredict = nn->readCSV("../bikeshare/testPF1_1.csv", false, csvTime2);
	csvTime += csvTime2;
	cout << csvTime << " s" << endl;

	for (size_t i = 0; i < xVec.size(); i++)
		xVec[i].erase(xVec[i].begin() + 1);
	for (size_t i = 0; i < xVecValidate.size(); i++)
		xVecValidate[i].erase(xVecValidate[i].begin() + 1);
	for (size_t i = 0; i < xVecPredict.size(); i++)
		xVecPredict[i].erase(xVecPredict[i].begin() + 1);

	// Note for classification setting the layers must be done before setting the data because the dimensions of the y must be known.
	int layers[4] = {10, 40, 160, 1}; //The bias unit is auto added by the class.
	nn->setLayers(layers, 4); //This will random initialise the weights
	nn->setData(xVec, yVec, false);
	nn->setValidateData(xVecValidate, yVecValidate, false);
	nn->setPredictData(xVecPredict);
	nn->normaliseData();
	nn->normaliseValidateData();
	nn->normalisePredictData();
	nn->setIters(30000);
	nn->setDisplay(3000);
	nn->addBiasData();
	nn->addBiasDataValidate();
	nn->addBiasDataPredict();
	nn->copyDataGPU();

	/* Arguments for momentum:
	 * 1. Momentum/viscosity. Set to 0 for gradient descent.
	 * 2. Learning rate.
	 * 3. Hidden layer activation function. See activations.cu for example. You can write your own and add it to activations.h/.cu
	 * 4. Activation function derivative. See activations.cu for example. You can write your own and add it to activations.h/.cu
	 * 5. Number of batches for mini-batch or stochastic. Set this to 1 for full batch or same as the dataset size for stochastic
	 */
	/*float gpuTime = nn->trainFuncApproxMomentum(0.9, 0.003, sigmoidGPU, sigmoidGradGPU, 1);
	cout << "GPU Training " << gpuTime << " s" << endl;*/

	/*
	nn->validateClassify(sigmoidGPU);
	cout << "Write CSV" << endl;
	vector<vector<float>> result = nn->predictFuncApprox(sigmoidGPU);
	cout << nn->writeCSV(result, "result.csv") << " s" << endl;
	cout << "Finished, press enter to end" << endl;
	*/

	// A second example
	cout << "Neural Network Classification Example" << endl;
	cublasNN *nn = new cublasNN;
	cout << "Read CSV: " << endl;
	float csvTime = 0;
	float csvTime2 = 0;
	vector<vector<float>> xVec = nn->readCSV("../MNIST/trainX.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> yVec = nn->readCSV("../MNIST/trainY.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> xVecValidate = nn->readCSV("../MNIST/validateX.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> yVecValidate = nn->readCSV("../MNIST/validateY.csv", false, csvTime2);
	csvTime += csvTime2;
	vector<vector<float>> xVecPredict = nn->readCSV("../MNIST/test.csv", true, csvTime2);
	csvTime += csvTime2;
	cout << csvTime << " s" << endl;

	// Note for classification setting the layers must be done before setting the data because the dimensions of the y must be known.
	int layers[4] = {784, 500, 300, 10}; //The bias unit is auto added by the class.
	nn->setLayers(layers, 4, randInitWeights2GPU); //This will random initialise the weights
	nn->setData(xVec, yVec, true);
	nn->setValidateData(xVecValidate, yVecValidate, true);
	nn->setPredictData(xVecPredict);
	nn->normaliseData();
	nn->normaliseValidateData();
	nn->normalisePredictData();
	nn->setIters(500);
	nn->setDisplay(1);
	nn->addBiasData();
	nn->addBiasDataValidate();
	nn->addBiasDataPredict();
	nn->copyDataGPU();

	/* Arguments for momentum:
	 * 1. Momentum/viscosity. Set to 0 for gradient descent.
	 * 2. Learning rate.
	 * 3. Hidden layer activation function. See activations.cu for example. You can write your own and add it to activations.h/.cu
	 * 4. Activation function derivative. See activations.cu for example. You can write your own and add it to activations.h/.cu
	 * 5. Output layer activation function. See activations.cu for example. You can write your own and add it to activations.h/.cu
	 * 6. Cost function. See costfunctions.cu for example. You can write your own and add it to costfunctions.h/.cu
	 * 7. Number of batches for mini-batch or stochastic. Set this to 1 for full batch or same as the dataset size for stochastic
	 */
	//float gpuTime = nn->trainClassifyMomentum(0.9, 0.075, sigmoidGPU, sigmoidGradGPU, sigmoidOutputGPU, negLnMaxCostGPU, 1);
	float gpuTime = nn->trainClassifyMomentum(0.9, 0.075, tanhGPU, sechSqGPU, softmaxGPU, crossEntropyCostGPU, 1);
	cout << "GPU Training " << gpuTime << " s" << endl;

	//nn->validateClassify(sigmoidGPU, sigmoidOutputGPU, negLnMaxCostGPU);
	nn->validateClassify(tanhGPU, softmaxGPU, crossEntropyCostGPU);

	cout << "Write CSV" << endl;
	//vector<vector<float>> result = nn->predictClassify(sigmoidGPU, sigmoidOutputGPU);
	vector<vector<float>> result = nn->predictClassify(tanhGPU, softmaxGPU);
	cout << nn->writeCSV(result, "result.csv") << " s" << endl;
	cout << "Finished, press enter to end" << endl;

	getchar();
	delete nn;
}
