#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>


#include "cublasNN.h"
#include "randinitweights.h"
#include "activations.h"
#include "costfunctions.h"

void load_file(string filename, vector<float>& result, int& m, int& n)
{
    ifstream in(filename,  ios::in|ios::binary/*|ios::ate*/);
    int dim[2];

    if(!in.is_open())
    {
        cerr << "failed to open file\n";
        return;
    }

    in.read((char*)dim, sizeof(dim));
    cout << "Dimensions: " << dim[0] << "x" << dim[1] << endl;
/*    vector<vector<float>> result;

    for(int i = 0; i < dim[0]; i++)
    {
        vector<float> line(dim[0]);
        in.read((char*)&line[0], dim[1] * sizeof(float));
        result.push_back(line);
    }
    for(int i = dim[0] - 5; i < dim[0]; i++)
    {
        for(int j = dim[1] - 5; j < dim[1]; j++)
            cout << result[i][j] << '\t';
        cout << endl;
    }
    return result;*/
    result = vector<float>(dim[0] * dim[1]);
    in.read((char*)&result[0], dim[0] * dim[1] * sizeof(float));
    m = dim[0];
    n = dim[1];

}

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
	nn->setLayers(layers, 4, randInitWeights1GPU); //This will random initialise the weights
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
	cout << "GPU Training " << gpuTime << " s" << endl;


	nn->validateFuncApprox(sigmoidGPU);
	cout << "Write CSV" << endl;
	vector<vector<float>> result = nn->predictFuncApprox(sigmoidGPU);
	cout << nn->writeCSV(result, "result.csv") << " s" << endl;
	cout << "Finished, press enter to end" << endl;


	// A second example
	/*cout << "Neural Network Classification Example" << endl;
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
	int layers[3] = {784, 50, 10}; //The bias unit is auto added by the class.
	nn->setLayers(layers, 3, randInitWeights2GPU); //This will random initialise the weights
	vector<vector<float>> xAll, yAll;
	xAll.insert(xAll.end(), xVec.begin(), xVec.end());
	xAll.insert(xAll.end(), xVecValidate.begin(), xVecValidate.end());
	yAll.insert(yAll.end(), yVec.begin(), yVec.end());
	yAll.insert(yAll.end(), yVecValidate.begin(), yVecValidate.end());
	//nn->setData(xVec, yVec, true);
	nn->setData(xAll, yAll, true);
	//nn->setValidateData(xVecValidate, yVecValidate, true);
	nn->setPredictData(xVecPredict);
	nn->normaliseData();
	//nn->normaliseValidateData();
	nn->normalisePredictData();
	nn->setIters(150);
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
	/*float gpuTime = nn->trainClassifyMomentum(0.9, 0.075, tanhGPU, sechSqGPU, softmaxGPU, crossEntropyCostGPU, 10);
	cout << "GPU Training " << gpuTime << " s" << endl;

	//nn->validateClassify(sigmoidGPU, sigmoidOutputGPU, negLnMaxCostGPU);
	//nn->validateClassify(tanhGPU, softmaxGPU, crossEntropyCostGPU);

	cout << "Write CSV" << endl;
	//vector<vector<float>> result = nn->predictClassify(sigmoidGPU, sigmoidOutputGPU);
	vector<vector<float>> result = nn->predictClassify(tanhGPU, softmaxGPU);
	cout << nn->writeCSV(result, "result.csv") << " s" << endl;
	cout << "Finished, press enter to end" << endl;*/
	cout << "Neural Network Function Approximation Example" << endl;

	cublasNN *nn = new cublasNN;
	//loadData("/home/henry/Coding/C++/Confidential/Work/hannd/benchmarkDataset/pca64d.bin", train_x);
	//loadData("/home/henry/Coding/C++/Confidential/Work/hannd/benchmarkDataset/xyz3d.bin", train_y);
	cout << "Read CSV:" << endl;

	// Note for classification setting the layers must be done before setting the data because the dimensions of the y must be known.
	int layers[3] = {150, 450, 5}; //The bias unit is auto added by the class.
	nn->setLayers(layers, 3, randInitWeights2GPU);
	vector<float> train_x;
	vector<float> train_y;
	int m, n, nOut;
	load_file("/home/henry/nfs4/csv/binary/x.bin", train_x, m, n);
	load_file("/home/henry/nfs4/csv/binary/y.bin", train_y, m, nOut);
	nn->setData(train_x, train_y, m, n, nOut, false);
	//float dummy_float;
	//vector<vector<float>> train_x = nn->readCSV("/home/henry/nfs4/csv/data/AAL_LDX.TXT", false, dummy_float);
	//vector<vector<float>> train_y = nn->readCSV("/home/henry/nfs4/csv/targets/AAL_LDY.TXT", false, dummy_float);
	//vector<vector<float>> train_y = nn->readCSV("/home/henry/nfs4/csv/targets/AAL_LDY_SINGLE.TXT", false, dummy_float);
	//nn->setData(train_x, train_y, false);
	nn->normaliseData();
	nn->setIters(10000);
	nn->setDisplay(100);
	nn->addBiasData();
	nn->copyDataGPU();

	/* Arguments for momentum:
	 * 1. Momentum/viscosity. Set to 0 for gradient descent.
	 * 2. Learning rate.
	 * 3. Hidden layer activation function. See activations.cu for example. You can write your own and add it to activations.h/.cu
	 * 4. Activation function derivative. See activations.cu for example. You can write your own and add it to activations.h/.cu
	 * 5. Number of batches for mini-batch or stochastic. Set this to 1 for full batch or same as the dataset size for stochastic
	 */
	float gpuTime = nn->trainFuncApproxMomentum(0.9999, std::pow(2, -14), tanhGPU, sechSqGPU, 1);
	cout << "GPU Training " << gpuTime << " s" << endl;

	cout << "Finished, press enter to end" << endl;

	getchar();
	delete nn;
}
